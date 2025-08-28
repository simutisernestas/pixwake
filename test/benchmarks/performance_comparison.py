import numpy as np
import multiprocessing
import time

import jax
import pandas as pd
import matplotlib.pyplot as plt
from jax import config as jcfg
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.site import UniformSite
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, NOJModel, Turbine, WakeSimulation

jcfg.update("jax_enable_x64", True)  # need float64 to match pywake


def generate_turbine_layout(n_turbines, spacing_D, rotor_diameter=120.0):
    """
    Generates a square grid of turbines.
    """
    n_side = int(np.sqrt(n_turbines))
    if n_side**2 != n_turbines:
        print(
            f"Warning: The requested number of turbines ({n_turbines}) is not a"
            f" perfect square. Using the largest possible square grid with "
            f"{n_side**2} turbines."
        )

    spacing_m = spacing_D * rotor_diameter
    x, y = np.meshgrid(
        np.arange(n_side) * spacing_m,
        np.arange(n_side) * spacing_m,
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
    max_out_at = 400
    n_cpu = n_cpu_at_50 + (max_cpu - n_cpu_at_50) * (n_turbines - 50) / (
        max_out_at - 50
    )
    return max(4, min(max_cpu, int(np.round(n_cpu))))


def run_benchmark(n_turbines_list, spacings_list):
    """Runs the full performance benchmark."""
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
    wake_expansion_k = 0.1
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
            pixwake_model = NOJModel(k=wake_expansion_k)
            pixwake_sim = WakeSimulation(pixwake_model, fpi_damp=1.0)

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
                wake_deficitModel=NOJDeficit(k=wake_expansion_k),
            )
            n_cpu = get_pywake_n_cpu(n_turbines_actual)

            ws_ts, wd_ts = generate_time_series_wind_data()
            start = time.time()
            pywake_wfm(x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True).aep().sum()
            pywake_aep_time_ts = time.time() - start
            start = time.time()
            pywake_wfm.aep_gradients(
                x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True
            )
            pywake_grad_time_ts = time.time() - start

            @jax.jit
            def pixwake_aep_ts(xx, yy):
                return pixwake_sim(xx, yy, ws_ts, wd_ts, pixwake_turbine).aep()

            grad_fn_ts = jax.jit(jax.value_and_grad(pixwake_aep_ts, argnums=(0, 1)))
            _ = pixwake_aep_ts(x, y).block_until_ready()
            _, _ = grad_fn_ts(x, y)
            start = time.time()
            _ = pixwake_aep_ts(x, y).block_until_ready()
            pixwake_aep_time_ts = time.time() - start
            start = time.time()
            _, (px_dx, px_dy) = grad_fn_ts(x, y)
            px_dx.block_until_ready()
            px_dy.block_until_ready()
            pixwake_grad_time_ts = time.time() - start

            # ws_wr, wd_wr, probs_wr = generate_wind_rose_data()
            # start = time.time()
            # pywake_wfm(x=x, y=y, wd=wd_wr, ws=ws_wr, n_cpu=n_cpu, time=True).aep(
            #     freq=probs_wr
            # ).sum()
            # pywake_aep_time_wr = time.time() - start
            # start = time.time()
            # pywake_wfm.aep_gradients(
            #     x=x, y=y, wd=wd_wr, ws=ws_wr, n_cpu=n_cpu, freq=probs_wr, time=True
            # )
            # pywake_grad_time_wr = time.time() - start

            # @jax.jit
            # def pixwake_aep_wr(xx, yy):
            #     return pixwake_sim(xx, yy, ws_wr, wd_wr, pixwake_turbine).aep(
            #         probabilities=probs_wr
            #     )

            # grad_fn_wr = jax.jit(jax.value_and_grad(pixwake_aep_wr, argnums=(0, 1)))
            # _ = pixwake_aep_wr(x, y).block_until_ready()
            # _, _ = grad_fn_wr(x, y)
            # start = time.time()
            # _ = pixwake_aep_wr(x, y).block_until_ready()
            # pixwake_aep_time_wr = time.time() - start
            # start = time.time()
            # _, (px_dx, px_dy) = grad_fn_wr(x, y)
            # px_dx.block_until_ready()
            # px_dy.block_until_ready()
            # pixwake_grad_time_wr = time.time() - start

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
                }
            )
            print(
                f"    PixWake Time-series (AEP/Grad): {pixwake_aep_time_ts:.4f}s / {pixwake_grad_time_ts:.4f}s"
            )
            print(
                f"    PyWake Time-series (AEP/Grad): {pywake_aep_time_ts:.4f}s / {pywake_grad_time_ts:.4f}s"
            )

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
        "AEP_Wind-Rose": ("pywake_aep_time_wr", "pixwake_aep_time_wr"),
        "Gradient_Wind-Rose": ("pywake_grad_time_wr", "pixwake_grad_time_wr"),
        "Total_Wind-Rose": ("pywake_total_time_wr", "pixwake_total_time_wr"),
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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    N_TURBINES_LIST = [50, 100, 250, 500]
    SPACINGS_LIST = [3, 5, 7]

    # For testing, run a smaller set
    # N_TURBINES_LIST = [49, 100]
    # SPACINGS_LIST = [3, 5]

    benchmark_results = run_benchmark(N_TURBINES_LIST, SPACINGS_LIST)

    print("\n--- Benchmark Results ---")
    for res in benchmark_results:
        print(res)

    print("\n--- Generating Plots ---")
    plot_results(benchmark_results)
    print("Plots generated successfully.")
