import argparse
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
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


def get_turbine_curves():
    """Returns power and CT curves for the turbine."""
    # fmt: off
    ct_vals = np.array([0.80, 0.79, 0.77, 0.75, 0.72, 0.68, 0.64, 0.62, 0.61, 
                       0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 
                       0.18, 0.15, 0.12, 0.10, 0.10])
    power_vals = np.array([100, 300, 600, 1200, 1800, 2300, 2700, 2900,
                           2950, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 3000, 3000, 0])
    # fmt: on
    ct_pw_ws = np.arange(3.0, 25.0 + 1.0, 1.0)
    return ct_pw_ws, power_vals, ct_vals


def benchmark_pywake(
    ws_ts, wd_ts, n_turbines, spacing, rotor_diameter=120.0, hub_height=100.0
):
    """Benchmarks PyWake for a given number of turbines and spacing."""
    x, y = generate_turbine_layout(n_turbines, spacing, rotor_diameter)
    n_turbines = len(x)

    ct_pw_ws, power_vals, ct_vals = get_turbine_curves()

    pywake_power_ct = PowerCtTabular(
        ws=ct_pw_ws, power=power_vals * 1000, power_unit="w", ct=ct_vals
    )
    pywake_turbines = WindTurbines(
        names=[f"WT{i}" for i in range(n_turbines)],
        diameters=[rotor_diameter] * n_turbines,
        hub_heights=[hub_height] * n_turbines,
        powerCtFunctions=[pywake_power_ct] * n_turbines,
    )
    pywake_wfm = All2AllIterative(
        site=UniformSite(),
        windTurbines=pywake_turbines,
        wake_deficitModel=PyWakeNiayifarGaussianDeficit(
            use_effective_ws=True, use_effective_ti=True
        ),
        turbulenceModel=PyWakeCrespoHernandez(),
    )

    def get_pywake_n_cpu(n_turbines, max_cpu=32):
        """Scales the number of CPUs for PyWake based on the number of turbines."""
        available_cpu_cores = int(
            os.environ.get("LSB_DJOB_NUMPROC") or os.cpu_count() or 1
        )
        n_cpu_at_50 = 4
        max_out_at = 200
        # linear scaling between 50 and max_out_at turbines
        n_cpu = n_cpu_at_50 + (max_cpu - n_cpu_at_50) * (n_turbines - 50) / (
            max_out_at - 50
        )
        return min(
            max(
                1,
                min(
                    max_cpu,
                    int(np.round(n_cpu)),
                ),
            ),
            available_cpu_cores,
        )

    n_cpu = get_pywake_n_cpu(n_turbines)

    start = time.time()
    pywake_wfm(x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True, TI=0.1).aep().sum()
    pywake_aep_time_ts = time.time() - start

    start = time.time()
    pywake_wfm.aep_gradients(
        x=x, y=y, wd=wd_ts, ws=ws_ts, n_cpu=n_cpu, time=True, TI=0.1
    )
    pywake_grad_time_ts = time.time() - start

    return pywake_aep_time_ts, pywake_grad_time_ts


def benchmark_pixwake(
    ws_ts, wd_ts, n_turbines, spacing, rotor_diameter=120.0, hub_height=100.0
):
    """Benchmarks PixWake for a given number of turbines and spacing."""
    import jax

    x, y = generate_turbine_layout(n_turbines, spacing, rotor_diameter)

    ct_pw_ws, power_vals, ct_vals = get_turbine_curves()

    power_curve = Curve(wind_speed=ct_pw_ws, values=power_vals)
    ct_curve = Curve(wind_speed=ct_pw_ws, values=ct_vals)

    pixwake_turbine = Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=hub_height,
        power_curve=power_curve,
        ct_curve=ct_curve,
    )
    pixwake_model = NiayifarGaussianDeficit(
        use_effective_ws=True,
        use_effective_ti=True,
    )
    pixwake_sim = WakeSimulation(
        pixwake_turbine,
        pixwake_model,
        turbulence=CrespoHernandez(),
        fpi_damp=1.0,
    )

    @jax.jit
    def pixwake_aep_ts(xx, yy):
        return pixwake_sim(xx, yy, ws_ts, wd_ts, ti=0.1).aep()

    grad_fn_ts = jax.jit(jax.value_and_grad(pixwake_aep_ts, argnums=(0, 1)))

    # Warmup
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

    return pixwake_aep_time_ts, pixwake_grad_time_ts


class BenchTarget:
    PYWAKE = "pywake"
    PIXWAKE = "pixwake"


@dataclass
class BenchmarkResult:
    n_turbines: int
    spacing: float
    aep_time_ts: float
    grad_time_ts: float
    mem_usage: list[float]


def run_benchmark(bench_target: BenchTarget, n_turbines_list, spacings_list):
    """Runs the full performance benchmark."""
    rotor_diameter = 120.0
    hub_height = 100.0
    ws_ts, wd_ts = generate_time_series_wind_data(n_hours=100)
    results = []

    for spacing in spacings_list:
        for n_turbines in n_turbines_list:
            print(f"\nBENCH {n_turbines} turbines with {spacing}D spacing...")

            if bench_target == BenchTarget.PYWAKE:
                bench_func = benchmark_pywake
            else:
                bench_func = benchmark_pixwake

            (mem_usage, (t_aep, t_grad)) = memory_usage(
                (
                    bench_func,
                    (ws_ts, wd_ts, n_turbines, spacing, rotor_diameter, hub_height),
                ),
                interval=0.01,
                max_usage=False,
                retval=True,
                include_children=True,
            )

            results.append(
                BenchmarkResult(
                    n_turbines=n_turbines,
                    spacing=spacing,
                    aep_time_ts=t_aep,
                    grad_time_ts=t_grad,
                    mem_usage=mem_usage,
                )
            )
            print(f"  Benchmark results ({bench_target}):")
            print(f"  Time-series (AEP/Grad): {t_aep:.4f}s / {t_grad:.4f}s")
            print(f"  Memory usage (max): {max(mem_usage):.2f} MiB")

    return results


def plot_results(run_id: str):
    """Plots the benchmark results and saves them to files."""
    import glob

    import xarray as xr

    files = glob.glob(f"benchout/{run_id}/benchmark_*.nc")
    if not files:
        print(f"No benchmark files found in 'benchout/{run_id}/'.")
        return

    try:
        datasets = [xr.open_dataset(f) for f in files]
        ds = xr.concat(datasets, dim="target")
    except ValueError as e:
        print(f"Could not merge datasets: {e}")
        print("This might be because the coordinates are not consistent across files.")
        print(
            "Please ensure that all benchmark runs were performed with the same settings."
        )
        return

    df = ds.to_dataframe().reset_index()
    df["total_time"] = df["aep_time"] + df["grad_time"]

    targets = df["target"].unique()
    if len(targets) < 2:
        print(
            "Both 'pywake' and 'pixwake' benchmark results are required for plotting."
        )
        print(f"Found targets: {targets}")
        return

    spacings = sorted(df["spacing"].unique())

    _, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot runtime
    ax = axes[0]
    for target in targets:
        for spacing in spacings:
            subset = df[(df["target"] == target) & (df["spacing"] == spacing)]
            subset = subset.sort_values("n_turbines")
            if not subset.empty:
                ax.plot(
                    subset["n_turbines"],
                    subset["total_time"],
                    "o-",
                    label=f"{target.capitalize()} {spacing}D",
                )

    ax.set_ylabel("Total Runtime (s)")
    ax.set_title("Benchmark: Runtime Comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    ax.set_yscale("log")

    # Plot memory usage
    ax = axes[1]
    for target in targets:
        for spacing in spacings:
            subset = df[(df["target"] == target) & (df["spacing"] == spacing)]
            subset = subset.sort_values("n_turbines")
            if not subset.empty:
                ax.plot(
                    subset["n_turbines"],
                    subset["mem_usage_max"],
                    "o-",
                    label=f"{target.capitalize()} {spacing}D",
                )

    ax.set_xlabel("Number of Turbines")
    ax.set_ylabel("Max Memory Usage (MiB)")
    ax.set_title("Benchmark: Memory Usage Comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    ax.set_yscale("log")

    plt.tight_layout()
    filename = f"benchout/{run_id}/benchmark_comparison_{int(time.time())}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()


if __name__ == "__main__":
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
    parser.add_argument(
        "--target",
        type=str,
        choices=[BenchTarget.PYWAKE, BenchTarget.PIXWAKE],
        default=None,
        help="Benchmark target: 'pywake' or 'pixwake'.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier for output files.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot benchmark results from 'benchout' folder.",
    )
    args = parser.parse_args()

    if args.plot:
        plot_results(args.run_id)
        exit()

    if not args.target:
        raise ValueError("Either --plot or --target must be specified.")

    benchmark_results = run_benchmark(args.target, args.n_turbines, args.spacings)

    import xarray as xr

    targets = [args.target]
    n_turbines_vals = args.n_turbines  # preserve input order
    spacing_vals = args.spacings  # preserve input order

    shape = (len(targets), len(n_turbines_vals), len(spacing_vals))
    data_arrays = {
        "aep_time": np.full(shape, np.nan),
        "grad_time": np.full(shape, np.nan),
        "mem_usage_max": np.full(shape, np.nan),
    }

    for r in benchmark_results:
        ti = targets.index(args.target)
        ni = n_turbines_vals.index(r.n_turbines)
        si = spacing_vals.index(r.spacing)

        data_arrays["aep_time"][ti, ni, si] = r.aep_time_ts
        data_arrays["grad_time"][ti, ni, si] = r.grad_time_ts
        if r.mem_usage:
            data_arrays["mem_usage_max"][ti, ni, si] = max(r.mem_usage)

    ds = xr.Dataset(
        {k: (("target", "n_turbines", "spacing"), v) for k, v in data_arrays.items()},
        coords={
            "target": targets,
            "n_turbines": n_turbines_vals,
            "spacing": spacing_vals,
        },
    )
    print("\n", ds)

    os.makedirs("benchout", exist_ok=True)
    os.makedirs(f"benchout/{args.run_id}", exist_ok=True)
    ds.to_netcdf(
        f"benchout/{args.run_id}/benchmark_{args.target}_{int(time.time())}.nc"
    )
