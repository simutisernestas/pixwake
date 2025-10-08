import argparse
import multiprocessing
import os
import time
from multiprocessing import Pipe, Process

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from py_wake.deficit_models.gaussian import (
    NiayifarGaussianDeficit as PyWakeNiayifarGaussianDeficit,
)
from py_wake.site import UniformSite
from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespoHernandez
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import NiayifarGaussianDeficit
from pixwake.turbulence import CrespoHernandez


def generate_turbine_layout(n_turbines, spacing_D, rotor_diameter=120.0):
    """
    Generates a rectangular grid of turbines.
    """

    def _closest_factors(n):
        factors = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        factors.sort(key=lambda x: x[0] + x[1])
        return factors[0]

    fx, fy = _closest_factors(n_turbines)
    assert fx * fy == n_turbines

    spacing_m = spacing_D * rotor_diameter
    x, y = np.meshgrid(
        np.arange(fx) * spacing_m,
        np.arange(fy) * spacing_m,
    )
    return x.flatten(), y.flatten()


def generate_time_series_wind_data(n_hours=8760, weibull_a=8, weibull_k=2):
    """
    Generates a time-series of wind speed and direction data.
    """
    ws = weibull_a * np.random.weibull(weibull_k, n_hours)
    wd = np.random.uniform(0, 360, n_hours)
    return ws, wd


def generate_wind_rose_data(
    ws_res=1.0, wd_res=1.0, weibull_a=8, weibull_k=2, max_ws=30
):
    """
    Generates wind data based on a wind rose.
    """
    from scipy.stats import weibull_min

    ws_bins = np.arange(0, max_ws, ws_res)
    wd_bins = np.arange(0, 360, wd_res)
    ws_probs = weibull_min.pdf(ws_bins, c=weibull_k, scale=weibull_a) * ws_res
    ws_probs /= ws_probs.sum()
    wd_probs = np.ones_like(wd_bins) / len(wd_bins)
    ws_grid, wd_grid = np.meshgrid(ws_bins, wd_bins)
    prob_grid = np.outer(wd_probs, ws_probs).T
    return ws_grid.flatten(), wd_grid.flatten(), prob_grid.flatten()


def get_pywake_n_cpu(n_turbines, max_cpu=32):
    """Scales the number of CPUs for PyWake based on the number of turbines."""
    n_cpu_at_50 = 4
    max_out_at = 200
    n_cpu = n_cpu_at_50 + (max_cpu - n_cpu_at_50) * (n_turbines - 50) / (
        max_out_at - 50
    )
    return max(4, min(max_cpu, int(np.round(n_cpu))))


def monitor_memory(pid, conn, interval=0.1):
    p = psutil.Process(pid)
    max_memory = 0
    running = True
    while running:
        try:
            if conn.poll():
                cmd = conn.recv()
                if cmd == "get_max":
                    conn.send(max_memory / (1024**2))  # Convert bytes to MB
                elif cmd == "reset":
                    max_memory = 0
                    conn.send("reset_done")
                elif cmd == "stop":
                    running = False
                    conn.send("stopped")
                    break
        except:
            running = False
            break

        mem = p.memory_info().rss
        # Add memory of all children processes
        for child in p.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                print("No such process : ) !!!")
                continue

        if mem > max_memory:
            max_memory = mem

        time.sleep(interval)
    conn.close()


def run_benchmark(n_turbines_list, spacings_list):
    """Runs the full performance benchmark."""

    parent_pid = psutil.Process().pid

    def start_monitor():
        parent_conn, child_conn = Pipe()
        proc = Process(target=monitor_memory, args=(parent_pid, child_conn, 0.01))
        proc.start()
        time.sleep(1.0)  # Give monitor time to start
        return parent_conn, proc

    parent_conn, monitor_proc = start_monitor()

    def terminate_monitor():
        nonlocal parent_conn, monitor_proc
        try:
            parent_conn.close()
        except Exception:
            pass
        monitor_proc.join(timeout=2)
        if monitor_proc.is_alive():
            print("[WARN] Monitor process did not terminate, terminating forcefully.")
            monitor_proc.terminate()
        try:
            parent_conn.close()
        except Exception:
            pass
        time.sleep(1.0)

    def safe_send_recv(cmd, timeout=5, attempts=5):
        nonlocal parent_conn, monitor_proc
        for _ in range(attempts):
            try:
                if not monitor_proc.is_alive():
                    terminate_monitor()
                    parent_conn, monitor_proc = start_monitor()

                parent_conn.send(cmd)
                if parent_conn.poll(timeout):
                    return parent_conn.recv()

                raise TimeoutError(f"Timeout waiting for monitor response to '{cmd}'")
            except (EOFError, BrokenPipeError, TimeoutError) as e:
                print(f"[ERROR] Monitor communication failed: {e}. Restarting monitor.")
                try:
                    parent_conn.close()
                except Exception:
                    pass
                parent_conn, monitor_proc = start_monitor()

        raise RuntimeError(
            f"Failed to communicate with monitor after restart for '{cmd}'"
        )

    # fmt: off
    ct_vals = np.array([0.80, 0.79, 0.77, 0.75, 0.72, 0.68, 0.64, 0.62, 0.61, 
                       0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 
                       0.18, 0.15, 0.12, 0.10, 0.10])
    power_vals = np.array([100, 300, 600, 1200, 1800, 2300, 2700, 2900,
                           2950, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 3000, 3000, 0])
    # fmt: on
    ct_pw_ws = np.arange(3.0, 25.0 + 1.0, 1.0)
    power_curve = Curve(wind_speed=ct_pw_ws, values=power_vals)
    ct_curve = Curve(wind_speed=ct_pw_ws, values=ct_vals)
    rotor_diameter = 120.0
    hub_height = 100.0
    results = []

    for spacing in spacings_list:
        for n_turbines in n_turbines_list:
            print(
                f"\nRunning benchmark for {n_turbines} turbines with {spacing}D spacing..."
            )
            x, y = generate_turbine_layout(n_turbines, spacing, rotor_diameter)
            n_turbines_actual = len(x)

            pixwake_turbine = Turbine(
                rotor_diameter=rotor_diameter,
                hub_height=hub_height,
                power_curve=power_curve,
                ct_curve=ct_curve,
            )
            pixwake_model = NiayifarGaussianDeficit(
                use_effective_ws=True,
                use_effective_ti=True,
                turbulence_model=CrespoHernandez(),
            )
            pixwake_sim = WakeSimulation(
                pixwake_model,
                fpi_damp=1.0,
                mapping_strategy="map",
                turbine=pixwake_turbine,
            )

            pywake_power_ct = PowerCtTabular(
                ws=ct_pw_ws, power=power_vals * 1000, power_unit="w", ct=ct_vals
            )
            pywake_turbines = WindTurbines(
                names=[f"WT{i}" for i in range(n_turbines_actual)],
                diameters=[rotor_diameter] * n_turbines_actual,
                hub_heights=[hub_height] * n_turbines_actual,
                powerCtFunctions=[pywake_power_ct] * n_turbines_actual,
            )
            pywake_wfm = All2AllIterative(
                site=UniformSite(),
                windTurbines=pywake_turbines,
                wake_deficitModel=PyWakeNiayifarGaussianDeficit(
                    use_effective_ws=True, use_effective_ti=True
                ),
                turbulenceModel=PyWakeCrespoHernandez(),
            )
            n_cpu = get_pywake_n_cpu(n_turbines_actual)

            ws_ts, wd_ts = generate_time_series_wind_data()

            safe_send_recv("reset")
            max_mem_before = safe_send_recv("get_max")

            start = time.time()
            pywake_wfm(
                x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True, TI=0.1
            ).aep().sum()
            pywake_aep_time_ts = time.time() - start
            start = time.time()
            pywake_wfm.aep_gradients(
                x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True, TI=0.1
            )
            pywake_grad_time_ts = time.time() - start

            max_mem_after = safe_send_recv("get_max")
            memory_usage_pywake_sim = max_mem_after - max_mem_before

            @jax.jit
            def pixwake_aep_ts(xx, yy):
                return pixwake_sim(xx, yy, ws_ts, wd_ts, ti=0.1).aep()

            grad_fn_ts = jax.jit(jax.value_and_grad(pixwake_aep_ts, argnums=(0, 1)))
            _ = pixwake_aep_ts(x, y).block_until_ready()
            _, _ = grad_fn_ts(x, y)

            safe_send_recv("reset")
            max_mem_before = safe_send_recv("get_max")

            start = time.time()
            _ = pixwake_aep_ts(x, y).block_until_ready()
            pixwake_aep_time_ts = time.time() - start
            start = time.time()
            _, (px_dx, px_dy) = grad_fn_ts(x, y)
            px_dx.block_until_ready()
            px_dy.block_until_ready()
            pixwake_grad_time_ts = time.time() - start

            max_mem_after = safe_send_recv("get_max")
            memory_usage_pixwake_sim = max_mem_after - max_mem_before

            results.append(
                {
                    "n_turbines": n_turbines_actual,
                    "spacing": spacing,
                    "pywake_aep_time_ts": pywake_aep_time_ts,
                    "pixwake_aep_time_ts": pixwake_aep_time_ts,
                    "pywake_grad_time_ts": pywake_grad_time_ts,
                    "pixwake_grad_time_ts": pixwake_grad_time_ts,
                    "pywake_aep_time_wr": pywake_aep_time_ts,
                    "pixwake_aep_time_wr": pixwake_aep_time_ts,
                    "pywake_grad_time_wr": pywake_grad_time_ts,
                    "pixwake_grad_time_wr": pixwake_grad_time_ts,
                    "memory_usage_pywake_sim": memory_usage_pywake_sim,
                    "memory_usage_pixwake_sim": memory_usage_pixwake_sim,
                }
            )
            print(
                f"    PixWake Time-series (AEP/Grad): {pixwake_aep_time_ts:.4f}s / {pixwake_grad_time_ts:.4f}s | MMem: {memory_usage_pixwake_sim:.2f}MB"
            )
            print(
                f"    PyWake Time-series (AEP/Grad): {pywake_aep_time_ts:.4f}s / {pywake_grad_time_ts:.4f}s | MMem: {memory_usage_pywake_sim:.2f}MB"
            )

    try:
        safe_send_recv("stop")
    except Exception as e:
        print(f"[WARN] Could not stop monitor cleanly: {e}")
    monitor_proc.join(timeout=2)
    if monitor_proc.is_alive():
        print("[WARN] Monitor process did not terminate, terminating forcefully.")
        monitor_proc.terminate()
    parent_conn.close()

    return results


def plot_results(results):
    """Plots the benchmark results and saves them to files."""
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    spacings = df["spacing"].unique()

    df["pywake_total_time_ts"] = df["pywake_aep_time_ts"] + df["pywake_grad_time_ts"]
    df["pixwake_total_time_ts"] = df["pixwake_aep_time_ts"] + df["pixwake_grad_time_ts"]
    df["pywake_total_time_wr"] = df["pywake_aep_time_wr"] + df["pywake_grad_time_wr"]
    df["pixwake_total_time_wr"] = df["pixwake_aep_time_wr"] + df["pixwake_grad_time_wr"]

    plot_configs = {
        "AEP_Time-Series": ("pywake_aep_time_ts", "pixwake_aep_time_ts"),
        "Gradient_Time-Series": ("pywake_grad_time_ts", "pixwake_grad_time_ts"),
        "Total_Time-Series": ("pywake_total_time_ts", "pixwake_total_time_ts"),
    }

    for name, (pywake_col, pixwake_col) in plot_configs.items():
        plt.figure(figsize=(10, 6))
        for spacing in spacings:
            df_spacing = df[df["spacing"] == spacing].sort_values("n_turbines")
            plt.plot(
                df_spacing["n_turbines"],
                df_spacing[pywake_col],
                "o-",
                label=f"PyWake {spacing}D",
            )
            plt.plot(
                df_spacing["n_turbines"],
                df_spacing[pixwake_col],
                "o-",
                label=f"PixWake {spacing}D",
            )
        plt.xlabel("Number of Turbines")
        plt.ylabel("Runtime (s)")
        plt.title(f"{name.replace('_', ' ')} Runtime Comparison (All Spacings)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")
        filename = f"figures/benchmark_{name.lower()}_all_spacings.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved plot: {filename}")
        plt.close()

    # Plot memory usage
    plt.figure(figsize=(10, 6))
    for spacing in spacings:
        df_spacing = df[df["spacing"] == spacing].sort_values("n_turbines")
        plt.plot(
            df_spacing["n_turbines"],
            df_spacing["memory_usage_pywake_sim"],
            "o-",
            label=f"PyWake {spacing}D",
        )
        plt.plot(
            df_spacing["n_turbines"],
            df_spacing["memory_usage_pixwake_sim"],
            "o-",
            label=f"PixWake {spacing}D",
        )
    plt.xlabel("Number of Turbines")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison (All Spacings)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.yscale("log")
    filename = "figures/benchmark_memory_usage_all_spacings.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Benchmark PixWake vs PyWake.")
    parser.add_argument(
        "--n_turbines",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="List of number of turbines to benchmark.",
    )
    parser.add_argument(
        "--spacings",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="List of turbine spacings (in D) to benchmark.",
    )
    args = parser.parse_args()

    N_TURBINES_LIST = args.n_turbines
    SPACINGS_LIST = args.spacings

    benchmark_results = run_benchmark(N_TURBINES_LIST, SPACINGS_LIST)

    print("\n--- Benchmark Results ---")
    for res in benchmark_results:
        print(res)

    print("\n--- Generating Plots ---")
    plot_results(benchmark_results)
    print("Plots generated successfully.")
