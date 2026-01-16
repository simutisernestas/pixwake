import argparse
import time
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from py_wake.site.shear import PowerShear
from py_wake.site.xrsite import XRSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

from pixwake import Turbine, Curve, WakeSimulation
from pixwake.plot import plot_flow_map
from pixwake.deficit import TurboGaussianDeficit, SelfSimilarityBlockageDeficit2020
from pixwake.rotor_avg import GaussianOverlapAvgModel
from pixwake.superposition import SquaredSum
from pixwake.utils import ct2a_mom1d

# --- TopFarm Imports ---
from topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.constraint_aggregation import (
    DistanceConstraintAggregation,
)
from topfarm.easy_drivers import EasySGDDriver
from topfarm.plotting import XYPlotComp


# --- Helper functions ---
def load_farm_boundaries():
    """Load pre-defined boundary coordinates for all farms in the study."""

    # fmt: off
    def _parse(coords):
        return np.array(coords).reshape((-1, 2)).T.tolist()
    dk1d_tender_9 = _parse([
        695987.10, 6201357.42,  693424.66, 6200908.39,  684555.85, 6205958.06,
        683198.50, 6228795.09,  696364.70, 6223960.96,  697335.32, 6204550.03,
    ])
    dk0z_tender_5 = _parse([
        696839.73, 6227384.49,  683172.62, 6237928.36, 681883.78, 6259395.21,
        695696.30, 6254559.16,
    ])
    dk0w_tender_3 = _parse([
        706694.39, 6224158.53,  703972.08, 6226906.60,  702624.63, 6253853.54,
        712771.62, 6257704.93,  715639.34, 6260664.68,  721593.24, 6257907.00,
    ])
    dk0v_tender_1 = _parse([
        695387.98, 6260724.98,  690623.95, 6265534.10,  689790.35, 6282204.66,
        706966.93, 6287633.44,  708324.28, 6264796.40,  696034.30, 6260723.06,
    ])
    dk0y_tender_4 = _parse([
        688971.22, 6289970.32,  699859.69, 6313455.88,  706084.61, 6313894.00,
        711981.42, 6312278.14,  712492.24, 6310678.30,  705728.34, 6295172.01,
        703484.11, 6292667.06,  695423.00, 6290179.44,
    ])
    dk0x_tender_2 = _parse([
        715522.10, 6271624.87,  714470.02, 6296972.62, 735902.87, 6290515.57,  
        726238.52, 6268396.34,
    ])
    dk1a_tender_6 = _parse([
        741993.80, 6285017.51,  754479.42, 6280870.40,  755733.10, 6260088.64,
        753546.86, 6256441.88,  738552.85, 6267674.69,  738130.35, 6276124.15,
    ])
    dk1b_tender_7 = _parse([
        730392.02, 6258565.79,  741435.03, 6261729.53,  743007.82, 6238891.85,
        741806.53, 6237068.79,  729493.72, 6233452.17,  729032.39, 6255601.55,
    ])
    dk1c_tender_8 = _parse([
        719322.37, 6234395.78,  730063.15, 6226372.25,  720738.33, 6206078.65,
        712391.75, 6209300.13,  709646.60, 6212504.92,
    ])
    dk1e_tender_10 = _parse([
        705363.69, 6203384.47,  716169.94, 6202667.31, 705315.73, 6178496.66,
        693580.72, 6176248.30,
    ])
    # fmt: on
    return [
        dk0w_tender_3,  # target farm
        dk1d_tender_9,
        dk0z_tender_5,
        dk0v_tender_1,
        dk0y_tender_4,
        dk0x_tender_2,
        dk1a_tender_6,
        dk1b_tender_7,
        dk1c_tender_8,
        dk1e_tender_10,
    ]


def get_turbine_types():
    """Generate turbine types with gradually increasing size and power"""
    RPs = np.arange(10, 16).astype(int)  # Rated power range from 10 to 15 MW
    Ds = (240 * np.sqrt(RPs / 15)).astype(int)  # Scale by sqrt of power ratio
    hub_heights = (Ds / 240 * 150).astype(int)  # Scale hub height proportionally

    # Create list of turbine types
    turbines = []

    for i, (rp, d, h) in enumerate(zip(RPs, Ds, hub_heights)):
        ws = np.arange(0.1, 30, 0.1)
        pw_turbine = GenericWindTurbine(f"WT_{i}", d, h, rp * 1e3)
        px_turbine = Turbine(
            rotor_diameter=pw_turbine.diameter(),
            hub_height=pw_turbine.hub_height(),
            power_curve=Curve(
                ws=jnp.array(ws), values=jnp.array(pw_turbine.power(ws)) / 1e3
            ),
            ct_curve=Curve(ws=jnp.array(ws), values=jnp.array(pw_turbine.ct(ws))),
            type_id=i,
        )
        turbines.append(px_turbine)

    return turbines


# --- Configuration ---
H5_LAYOUT_FILE = "sandbox/re_precomputed_layouts.h5"
SITE_FILE = "sandbox/ref_site.nc"
TIME_SERIES_FILE = "sandbox/energy_island_10y_daily_av_wind.csv"

TARGET_FARM_IDX = 0
TARGET_TYPE_IDX = 5
N_NEIGHBOR_FARMS = 9


def get_layout_from_h5(farm_idx, type_idx, seed):
    """Helper to load a pre-computed layout."""
    config_key = f"farm{farm_idx}_t{type_idx}_s{seed}"
    try:
        with h5py.File(H5_LAYOUT_FILE, "r") as f:
            if config_key in f:
                return f[config_key]["layout"][:]
            return None, None
    except Exception:
        return None, None


def create_wfm(site, wind_turbines):
    """Factory function to create a clean, identical WFM instance."""
    _ = site  # TODO: not used... shear should be taken into account
    wake_deficit_model = TurboGaussianDeficit(
        ct2a=ct2a_mom1d,
        ctlim=0.96,
        rotor_avg_model=GaussianOverlapAvgModel(),
        superposition=SquaredSum(),
        use_effective_ws=False,
        use_radius_mask=False,
    )
    return WakeSimulation(
        wind_turbines,
        wake_deficit_model,
        blockage=SelfSimilarityBlockageDeficit2020(),
        mapping_strategy="vmap",
    )


__liberal_aep_jax_func_cache = None


def liberal_aep_jac(x, y, *, wfm_pristine, ts, n_mc_samples, target_type_idx):
    """Expected gradient for the target farm in isolation."""
    global __liberal_aep_jax_func_cache

    sample_ts = ts.sample(n_mc_samples, replace=True)

    if __liberal_aep_jax_func_cache is None:

        def __aep_func(_x, _y, _ws, _wd):
            return wfm_pristine(
                _x, _y, _ws, _wd, ti_amb=0.1, wt_types=[target_type_idx] * len(x)
            ).aep()

        __liberal_aep_jax_func_cache = jax.jit(jax.grad(__aep_func, argnums=(0, 1)))

    dx, dy = __liberal_aep_jax_func_cache(
        jnp.array(x),
        jnp.array(y),
        jnp.array(sample_ts["WS_150"].values),
        jnp.array(sample_ts["WD_150"].values),
    )
    jac = np.array([dx, dy])
    return jac * 1e6


__conservative_aep_jax_func_cache = None


def conservative_aep_jac(
    x,
    y,
    *,
    wfm_pristine,
    ts,
    target_type_idx,
    n_mc_samples,
    x_neighbors,
    y_neighbors,
    wt_types_neighbors,
):
    """Gradient for conservative case, compute with neighbor farms always present."""
    global __conservative_aep_jax_func_cache

    indices = np.random.randint(0, len(ts), n_mc_samples)
    sample_ts = ts.iloc[indices]

    _wt_types_concat = np.concatenate(
        [np.array([target_type_idx] * len(x)), wt_types_neighbors], dtype=int
    ).tolist()

    if __conservative_aep_jax_func_cache is None:

        def __aep_func(_x, _y, _ws, _wd):
            _x_concat = jnp.concatenate([_x, x_neighbors])
            _y_concat = jnp.concatenate([_y, y_neighbors])

            return wfm_pristine(
                _x_concat,
                _y_concat,
                _ws,
                _wd,
                ti_amb=0.1,
                wt_types=_wt_types_concat,
            ).aep()

        __conservative_aep_jax_func_cache = jax.jit(
            jax.grad(__aep_func, argnums=(0, 1))
        )

    dx, dy = __conservative_aep_jax_func_cache(
        jnp.array(x),
        jnp.array(y),
        jnp.array(sample_ts["WS_150"].values),
        jnp.array(sample_ts["WD_150"].values),
    )
    jac = np.array([dx, dy])
    return jac * 1e6


def placeholder_cost_func(x, y, **kwargs):
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind Farm Layout Optimizer")
    parser.add_argument(
        "--mode",
        default="conservative",
        choices=["conservative", "liberal"],
    )
    parser.add_argument(
        "--conservative_farm_id",
        type=int,
        default=None,
        help="For 'conservative' mode: specify a single farm ID (1-9) to use. If None, uses all 9 farms.",
    )
    parser.add_argument("--output", default="sandbox/optimized_layout.h5")
    parser.add_argument("--n_mc_samples", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Run N gradient evaluations to measure runtime, then exit.",
    )
    parser.add_argument(
        "--plot-flow-map",
        action="store_true",
        help="Plot the flow map and exit.",
    )
    parser.add_argument(
        "--flow-map-ws",
        type=float,
        default=9.0,
        help="Wind speed for flow map plot (default: 9.0 m/s).",
    )
    parser.add_argument(
        "--flow-map-wd",
        type=float,
        default=270.0,
        help="Wind direction for flow map plot (default: 270 deg).",
    )
    parser.add_argument(
        "--flow-map-output",
        type=str,
        default=None,
        help="Output file path for flow map plot (e.g., 'flow_map.png'). If not set, displays interactively.",
    )
    parser.add_argument(
        "--flow-map-height-mode",
        type=str,
        choices=["mean", "specific", "average"],
        default="average",
        help="Height mode for flow map: 'mean' (mean hub height), 'specific' (use --flow-map-height), "
        "or 'average' (average across rotor-swept heights). Default: 'average'.",
    )
    parser.add_argument(
        "--flow-map-height",
        type=float,
        default=None,
        help="Specific height for flow map when --flow-map-height-mode=specific.",
    )
    parser.add_argument(
        "--flow-map-n-heights",
        type=int,
        default=5,
        help="Number of height samples when --flow-map-height-mode=average. Default: 5.",
    )
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        print(f"🔒 NumPy random seed set to {args.seed}.")

    print("--- Pre-loading data ---")

    shear_model = PowerShear(h_ref=150, alpha=0.1)
    site_pristine = XRSite.load(SITE_FILE, shear=shear_model)
    turbine_types = get_turbine_types()
    wfm_pristine = create_wfm(site_pristine, turbine_types)
    print("-> WFM initialized")

    ts_data = pd.read_csv(TIME_SERIES_FILE, sep=";")
    ts_data = ts_data[ts_data.WS_150 > 5]
    ts_data = ts_data[ts_data.WS_150 < 10]
    ts_data = ts_data.reset_index()
    ts_data.index = np.arange(len(ts_data))

    x0, y0 = get_layout_from_h5(TARGET_FARM_IDX, TARGET_TYPE_IDX, seed=0)
    np.random.seed(2)
    x0 = np.random.uniform(x0.min(), x0.max(), x0.size)
    y0 = np.random.uniform(y0.min(), y0.max(), y0.size)
    if x0 is None:
        raise SystemExit("Fatal: Initial layout for target farm could not be loaded.")
    n_wt_target = len(x0)

    farm_boundaries = load_farm_boundaries()
    target_boundary = np.array(farm_boundaries[TARGET_FARM_IDX]).T.tolist()

    ss_seeds = np.random.randint(1, 50, size=N_NEIGHBOR_FARMS)
    x_neighbors, y_neighbors, types_neighbors = [], [], []
    for n in range(N_NEIGHBOR_FARMS):
        if n == TARGET_FARM_IDX:
            continue
        type_idx = n % 6
        seed = ss_seeds[n]
        xn, yn = get_layout_from_h5(farm_idx=n, type_idx=type_idx, seed=seed)
        if xn is not None:
            x_neighbors.append(xn)
            y_neighbors.append(yn)
            types_neighbors.append(np.full(len(xn), type_idx))
        else:
            raise SystemExit(
                f"Fatal: Could not load layout for neighbor farm {n}, type {type_idx}, seed {seed}."
            )

    x_neighbors = np.concatenate(x_neighbors)
    y_neighbors = np.concatenate(y_neighbors)
    wt_types_neighbors = np.concatenate(types_neighbors)

    if args.plot_flow_map:
        height_mode_desc = {
            "mean": "mean hub height",
            "specific": f"height={args.flow_map_height}m",
            "average": f"averaged over {args.flow_map_n_heights} heights",
        }
        print(
            f"\n--- Plotting flow map (ws={args.flow_map_ws} m/s, wd={args.flow_map_wd} deg, "
            f"{height_mode_desc[args.flow_map_height_mode]}) ---"
        )

        if args.mode == "conservative":
            all_x = np.concatenate([x0, x_neighbors])
            all_y = np.concatenate([y0, y_neighbors])
            all_types = (
                np.concatenate([np.full(len(x0), TARGET_TYPE_IDX), wt_types_neighbors])
                .astype(int)
                .tolist()
            )
        else:
            all_x = x0
            all_y = y0
            all_types = [TARGET_TYPE_IDX] * len(x0)

        # Define grid bounds with margin around turbines
        margin = 2000
        x_min, x_max = all_x.min() - margin, all_x.max() + margin
        y_min, y_max = all_y.min() - margin, all_y.max() + margin
        grid_density = 400
        grid_x, grid_y = np.mgrid[
            x_min : x_max : grid_density * 1j,
            y_min : y_max : grid_density * 1j,
        ]

        flow_map_data, (fm_x, fm_y) = wfm_pristine.flow_map(
            jnp.array(all_x),
            jnp.array(all_y),
            fm_x=jnp.array(grid_x.ravel()),
            fm_y=jnp.array(grid_y.ravel()),
            fm_z=args.flow_map_height,
            ws=args.flow_map_ws,
            wd=args.flow_map_wd,
            ti=0.1,
            wt_types=all_types,
            height_mode=args.flow_map_height_mode,
            n_height_samples=args.flow_map_n_heights,
        )

        ax = plot_flow_map(
            fm_x, fm_y, flow_map_data[0], jnp.array(all_x), jnp.array(all_y), show=False
        )
        ax.set_title(
            f"Flow Map ({args.mode} mode, ws={args.flow_map_ws} m/s, wd={args.flow_map_wd}°, "
            f"{height_mode_desc[args.flow_map_height_mode]})"
        )

        if args.flow_map_output:
            plt.savefig(args.flow_map_output, dpi=150, bbox_inches="tight")
            print(f"Flow map saved to {args.flow_map_output}")
        else:
            plt.show()

        raise SystemExit(0)

    print(f"--- Configuring optimizer for '{args.mode}' mode ---")

    base_kwargs = {
        "wfm_pristine": wfm_pristine,
        "ts": ts_data,
        "n_mc_samples": args.n_mc_samples,
        "target_type_idx": TARGET_TYPE_IDX,
    }

    if args.mode == "liberal":
        grad_func = partial(liberal_aep_jac, **base_kwargs)

    elif args.mode == "conservative":
        conservative_kwargs = {
            **base_kwargs,
            "x_neighbors": x_neighbors,
            "y_neighbors": y_neighbors,
            "wt_types_neighbors": wt_types_neighbors,
        }
        grad_func = partial(conservative_aep_jac, **conservative_kwargs)

    else:
        raise SystemExit(f"Fatal: Unknown mode '{args.mode}' specified.")

    if args.dry_run > 0:
        print(f"\n--- Dry run: timing {args.dry_run} gradient evaluation(s) ---")
        # Warmup run (JIT compilation)
        print("Warmup (JIT compilation)...", end=" ", flush=True)
        t0 = time.time()
        _ = grad_func(x0, y0)
        warmup_time = time.time() - t0
        print(f"{warmup_time:.2f}s")

        # Timed runs
        times = []
        for i in range(args.dry_run):
            t0 = time.time()
            _ = grad_func(x0, y0)
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  Run {i + 1}: {elapsed:.3f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(
            f"\nResults ({args.mode} mode, {len(x0)} turbines, {args.n_mc_samples} MC samples):"
        )
        print(f"  Average: {avg_time:.3f}s (+/- {std_time:.3f}s)")
        print(f"  Total:   {np.sum(times):.3f}s")
        print("--- Dry run finished ---")
        raise SystemExit(0)

    cost_comp = CostModelComponent(
        input_keys=["x", "y"],
        n_wt=n_wt_target,
        cost_function=placeholder_cost_func,
        cost_gradient_function=grad_func,
        objective=True,
        maximize=True,
    )

    boundary_comp = XYBoundaryConstraint(target_boundary, "polygon")

    problem = TopFarmProblem(
        design_vars={"x": x0, "y": y0},
        cost_comp=cost_comp,
        constraints=[
            DistanceConstraintAggregation(
                boundary_comp,
                n_wt_target,
                3 * turbine_types[TARGET_TYPE_IDX].rotor_diameter,
                None,
            )
        ],
        driver=EasySGDDriver(
            maxiter=args.iterations,
            learning_rate=args.lr,
            speedupSGD=True,
            sgd_thresh=0.03,
        ),
        plot_comp=XYPlotComp(save_plot_per_iteration=True, plot_initial=False),
    )

    print(f"\n--- Starting Optimization ---")
    start_time = time.time()
    _, state, _ = problem.optimize()
    end_time = time.time()
    print(f"--- Optimization Finished in {end_time - start_time:.2f} seconds ---")

    print(f"Saving final layout to {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("layout", data=np.array([state["x"], state["y"]]))
        f.attrs["mode"] = args.mode
        f.attrs["seed"] = args.seed if args.seed is not None else "None"
        if args.mode == "conservative":
            f.attrs["conservative_farm_id"] = (
                args.conservative_farm_id
                if args.conservative_farm_id is not None
                else "all"
            )

    print("--- Script Finished ---")
