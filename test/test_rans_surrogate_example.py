import json
import sys

import pytest

from pixwake.rotor_avg import CGIRotorAvg

if sys.version_info >= (3, 14):
    pytest.skip("Flax not compatible with Python 3.14+", allow_module_level=True)
try:
    import flax.linen as nn
    from flax import serialization
    from flax.struct import field
except ImportError:
    pytest.skip("Flax not installed", allow_module_level=True)

import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp
from scipy.ndimage import gaussian_filter1d

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit.base import WakeDeficit
from pixwake.turbulence.base import WakeTurbulence

asarray_method = np.asarray
from py_wake.examples.data.dtu10mw import DTU10MW

np.asarray = asarray_method


class WakeDeficitModelFlax(nn.Module):
    """A Flax module for the wake deficit model."""

    _scale_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
        )
    )
    _mean_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                3.34995157e1,
                3.63567130e-04,
                2.25024289e-02,
                1.43747711e-01,
                -1.45229452e-03,
                6.07149107e-01,
            ]
        )
    )
    _scale_y: jnp.ndarray = field(default_factory=lambda: jnp.array([0.02168894]))
    _mean_y: jnp.ndarray = field(default_factory=lambda: jnp.array([0.00614207]))

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the wake deficit model to the input."""
        x = (x - self._mean_x) / self._scale_x
        x = nn.tanh(nn.Dense(70)(x))
        x = nn.sigmoid(nn.Dense(102)(x))
        x = nn.sigmoid(nn.Dense(102)(x))
        x = nn.sigmoid(nn.Dense(102)(x))
        x = nn.Dense(1)(x)
        return (x * self._scale_y) + self._mean_y


class WakeAddedTIModelFlax(nn.Module):
    """A Flax module for the wake-added turbulence intensity model."""

    _scale_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
        )
    )
    _mean_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                3.34995157e1,
                3.63567130e-04,
                2.25024289e-02,
                1.43747711e-01,
                -1.45229452e-03,
                6.07149107e-01,
            ]
        )
    )
    _scale_y: jnp.ndarray = field(default_factory=lambda: jnp.array([0.00571155]))
    _mean_y: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0014295]))

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the wake-added TI model to the input."""
        x = (x - self._mean_x) / self._scale_x
        x = nn.sigmoid(nn.Dense(118)(x))
        x = nn.sigmoid(nn.Dense(118)(x))
        x = nn.sigmoid(nn.Dense(118)(x))
        x = nn.sigmoid(nn.Dense(118)(x))
        x = nn.Dense(1)(x)
        return (x * self._scale_y) + self._mean_y


def _load_model(model_class: type[nn.Module], filename: str) -> tuple[nn.Module, Any]:
    """Loads a single Flax model from a file."""
    model = model_class()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), "rb"
    ) as f:
        bytes_data = f.read()
    return model, serialization.from_bytes(variables, bytes_data)


def _predict(
    model: nn.Module,
    params: Any,
    ti: float | jnp.ndarray,
    x_d,
    y_d,
    ct_eff,
) -> jnp.ndarray:
    """A helper function to run predictions with the Flax models."""
    md_input = jnp.stack(
        [
            x_d,  # normalized x distance
            y_d,  # normalized y distance
            jnp.zeros_like(x_d),  # (z - h_hub) / D; evaluating at hub height
            jnp.full_like(x_d, ti),  # turbulence intensity
            jnp.zeros_like(x_d),  # yaw
            jnp.broadcast_to(ct_eff, x_d.shape),  # thrust coefficient
        ],
        axis=-1,
    ).reshape(-1, 6)
    return jnp.array(model.apply(params, md_input)).reshape(x_d.shape)


class RANSDeficit(WakeDeficit):
    """A RANS surrogate model for wake prediction. This model uses two pre-trained
    neural networks to predict the wake deficit and added turbulence intensity. It
    is based on high-fidelity RANS CFD simulations.
    """

    def __init__(self, use_effective_ws=True, use_effective_ti=False, **kwargs) -> None:
        """Initializes the RANSDeficit model."""
        super().__init__(use_radius_mask=False, **kwargs)
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti
        self.deficit_model, self.deficit_weights = _load_model(
            WakeDeficitModelFlax,
            "./data/rans_deficit_surrogate.msgpack",
        )

    def _deficit(self, ws_eff, ti_eff, ctx):
        """Computes the wake deficit using the RANS surrogate model.

        This method calculates the velocity deficit and added turbulence
        intensity for each turbine in the wind farm.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.
            xs_r: An array of x-coordinates for each receiver point (optional).
            ys_r: An array of y-coordinates for each receiver point (optional).
            ti_eff: An array of effective turbulence intensities at each turbine.

        Returns:
            A tuple containing the updated effective wind speeds and turbulence
            intensities at each turbine.

        Raises:
            ValueError: If ctx.ti is None - turbulence intensity is required.
        """
        if ctx.ti is None and ti_eff is None:
            raise ValueError(
                "RANSDeficit requires turbulence intensity (ti) to be provided. "
                "Pass ti parameter when calling WakeSimulation."
            )

        x_d = ctx.dw / ctx.turbine.rotor_diameter
        y_d = ctx.cw / ctx.turbine.rotor_diameter
        ct_eff = ctx.turbine.ct(ws_eff)
        in_domain_mask = (
            (x_d < 70)
            & (x_d > -3)
            & (jnp.abs(y_d) < 6)
            & ((x_d > 1e-6) | (x_d < -1e-6))
            & ((y_d > 1e-6) | (y_d < -1e-6))
        )
        ti_input = ti_eff if self.use_effective_ti else ctx.ti

        deficit_fraction = _predict(
            self.deficit_model,
            self.deficit_weights,
            ti_input,
            x_d,
            y_d,
            ct_eff,
        )
        deficit_fraction = jnp.where(in_domain_mask, deficit_fraction, 0.0)

        ws_reference = ws_eff[None, :]
        ws_reference = (
            ws_reference
            if self.use_effective_ws
            else jnp.full_like(ws_reference, ctx.ws)
        )
        return deficit_fraction * ws_reference

    def _wake_radius(self, ws_eff, ti_eff, ctx) -> jnp.ndarray:
        return ctx.turbine.rotor_diameter * 6.0


class RANSTurbulence(WakeTurbulence):
    """A RANS surrogate model for wake-added turbulence intensity prediction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.turbulence_model, self.ti_weights = _load_model(
            WakeAddedTIModelFlax,
            "./data/rans_addedti_surrogate.msgpack",
        )

    def _added_turbulence(self, ws_eff, ti_eff, ctx):
        x_d = ctx.dw / ctx.turbine.rotor_diameter
        y_d = ctx.cw / ctx.turbine.rotor_diameter
        ct_eff = ctx.turbine.ct(ws_eff)
        in_domain_mask = (
            (x_d < 70)
            & (x_d > -3)
            & (jnp.abs(y_d) < 6)
            & ((x_d > 1e-6) | (x_d < -1e-6))
            & ((y_d > 1e-6) | (y_d < -1e-6))
        )
        ti_input = ti_eff if ti_eff is not None else ctx.ti
        # ti_input = ctx.ti

        added_turbulence = _predict(
            self.turbulence_model,
            self.ti_weights,
            ti_input,
            x_d,
            y_d,
            ct_eff,
        )
        return jnp.where(in_domain_mask, added_turbulence, 0.0)


# could add to curve object or implementation from pywake ?
def smooth_curve(ws, values, sigma=0.5):
    """Apply Gaussian smoothing to make curve differentiable"""
    smoothed = gaussian_filter1d(values, sigma=sigma, mode="nearest")
    return ws, smoothed


def build_dtu10mw_wt(smooth=False) -> Turbine:
    pywake_turbine = DTU10MW(method="linear")  # smoothing is done here
    ws = jnp.linspace(0, 30.0, 30).tolist()

    # Smooth the curves
    if smooth:
        ws_power, power = smooth_curve(ws, pywake_turbine.power(ws), sigma=1.0)
        ws_ct, ct = smooth_curve(ws, pywake_turbine.ct(ws), sigma=0.5)
    else:
        ws_power = ws_ct = ws
        power = pywake_turbine.power(ws)
        ct = pywake_turbine.ct(ws)

    pixwake_turbine = Turbine(
        rotor_diameter=pywake_turbine.diameter(),
        hub_height=pywake_turbine.hub_height(),
        power_curve=Curve(
            ws=jnp.array(ws_power),
            values=jnp.array(power),
        ),
        ct_curve=Curve(
            ws=jnp.array(ws_ct),
            values=jnp.array(ct),
        ),
    )

    # import matplotlib.pyplot as plt
    # ws = jnp.linspace(0, 30, 1000)
    # ct = pixwake_turbine.ct(ws)
    # ct_ws_grad_func = jax.vmap(jax.grad(pixwake_turbine.ct))
    # ct_grad = ct_ws_grad_func(ws)
    # plt.figure()
    # plt.plot(ws, ct, label="Ct Curve")
    # plt.xlabel("Wind Speed (m/s)")
    # plt.ylabel("Thrust Coefficient (Ct)")
    # plt.title("DTU 10MW Turbine Thrust Coefficient Curve")
    # plt.figure()
    # plt.plot(ws, ct_grad, label="dCt/dWs", color="orange")
    # plt.xlabel("Wind Speed (m/s)")
    # plt.ylabel("dCt/dWs")
    # plt.title("DTU 10MW Turbine Thrust Coefficient Gradient")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # exit()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # pywake_turbine.plot_power_ct()
    # from pixwake.plot import plot_power_and_thrust_curve
    # plot_power_and_thrust_curve(pixwake_turbine, show=True)
    # exit()
    return pixwake_turbine


def block_all(res):
    if isinstance(res, tuple):
        return tuple(block_all(r) for r in res)
    else:
        return res.block_until_ready()


def load_opt_site_and_reference_resource():
    import yaml

    wind_resource = yaml.load(
        open("Wind_Resource.yaml"),
        Loader=yaml.FullLoader,
    )["wind_resource"]
    A = wind_resource["weibull_a"]
    k = wind_resource["weibull_k"]
    freq = wind_resource["sector_probability"]
    wd = wind_resource["wind_direction"]
    ws = wind_resource["wind_speed"]
    TI = wind_resource["turbulence_intensity"]["data"]
    import xarray as xr
    from py_wake.site.xrsite import XRSite

    site = XRSite(
        ds=xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", freq["data"]),
                "Weibull_A": ("wd", A["data"]),
                "Weibull_k": ("wd", k["data"]),
                "TI": (wind_resource["turbulence_intensity"]["dims"][0], TI),
            },
            coords={"wd": wd, "ws": ws},
        )
    )
    return (site, np.array(ws), np.array(wd))


def test_rans_surrogate_aep():
    CUTOUT_WS = 25.0
    CUTIN_WS = 4.0

    # site, site_ws, site_wd = load_opt_site_and_reference_resource()
    # site_wd = jnp.arange(0, 360, 1)
    # pix_ws, pix_wd = jnp.meshgrid(site_ws, site_wd)
    # pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()
    # P_ilk = site.local_wind(ws=site_ws, wd=site_wd).P_ilk
    # pix_probs = P_ilk.reshape((1, pix_wd.size)).T
    # # normalize
    # # pix_probs /= jnp.sum(pix_probs)

    # layout = np.load(
    #     "./IEA_ModelChoice.AWAKEN_OptDriver.SGD_seed149_initial_pos.npy",
    #     allow_pickle=True,
    # ).item()
    # lx, ly = layout["x"], layout["y"]

    seed = int(time.time()) % 142
    print(f"Using seed: {seed}")
    onp.random.seed(seed)
    T = 100
    WSS = jnp.asarray(onp.random.uniform(CUTIN_WS, CUTOUT_WS, T))
    WDS = jnp.asarray(onp.random.uniform(0, 360, T))
    # WSS = jnp.asarray([4.96990966796875])
    # WDS = jnp.asarray([58.039665])
    # print(WSS)
    # print(WDS)
    # WSS = pix_ws
    # WDS = pix_wd

    turbine = build_dtu10mw_wt()
    wi, le = 10, 8
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * turbine.rotor_diameter, wi),
        jnp.linspace(0, le * 3 * turbine.rotor_diameter, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    # add some noise to positions
    # xs += onp.random.normal(0, 0.1 * turbine.rotor_diameter, xs.shape)
    # ys += onp.random.normal(0, 0.1 * turbine.rotor_diameter, ys.shape)
    # assert xs.shape[0] == (wi * le), xs.shape

    model = RANSDeficit()
    turbulence = RANSTurbulence()
    sim = WakeSimulation(turbine, model, turbulence)

    # flow_map, (fx, fy) = sim.flow_map(lx, ly, ti=0.06, wd=270, ws=10.0)
    # from pixwake.plot import plot_flow_map
    # plot_flow_map(fx, fy, flow_map, show=False)
    # import matplotlib.pyplot as plt
    # plt.savefig("rans_surrogate_flow_map_example.png")

    def aep(xx, yy):
        return sim(xx, yy, WSS, WDS, 0.06).aep()  # probabilities=pix_probs

    # # jit_aep = jax.jit(aep)
    # aep_value = aep(jnp.array(xs), jnp.array(ys))  # warm-up
    # print(aep_value)
    # # aep_value = jit_aep(jnp.array(xs), jnp.array(ys))  # warm-up
    # # print(aep_value)
    # # exit()
    # return

    aep_and_grad = jax.jit(
        aep,
    )  # argnums=(0, 1)
    grad_func = jax.jit(jax.grad(aep, argnums=(0, 1)))

    res = aep_and_grad(xs, ys)
    grad = grad_func(xs, ys)
    block_all(res)
    print("\nRunning JIT; Warm-up complete...\n")
    s = time.time()
    res = aep_and_grad(xs, ys)
    block_all(res)
    print(f"AEP: {res} in {time.time() - s:.3f} seconds")
    s = time.time()
    grad = grad_func(xs, ys)
    block_all(grad)
    print(f"Gradients computed in {time.time() - s:.3f} seconds")

    # print(f"AEP: {res[0]} in {time.time() - s:.3f} seconds")

    # assert jnp.isfinite(res[0]).all(), "AEP should be finite"
    # assert jnp.isfinite(res[1][0]).all(), "Gradient of x should be finite"
    # assert jnp.isfinite(res[1][1]).all(), "Gradient of y should be finite"


def run_opt(seed=0):
    from pathlib import Path

    import shapely
    from topfarm import TopFarmProblem
    from topfarm.constraint_components.boundary import XYBoundaryConstraint
    from topfarm.constraint_components.constraint_aggregation import (
        DistanceConstraintAggregation,
    )
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.cost_models.cost_model_wrappers import CostModelComponent
    from topfarm.easy_drivers import EasySGDDriver
    from topfarm.plotting import XYPlotComp

    seed += 42
    print(f"Using seed: {seed}")
    onp.random.seed(seed)

    site, site_ws, site_wd = load_opt_site_and_reference_resource()
    site_wd = jnp.arange(0, 360, 1)
    pix_ws, pix_wd = jnp.meshgrid(site_ws, site_wd)
    pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()

    # fmt: off
    def aep(): return 0.0
    # fmt: on

    TI_VALUE = 0.06
    # wind resource
    WDS = np.linspace(0, 360 - 1, 360)
    dirs = WDS  # SITE.ds["wd"][:-1]
    repeat = WDS.shape[0] // site.ds["Sector_frequency"][:-1].shape[0]
    freqs = np.repeat(site.ds["Sector_frequency"][:-1], repeat).values
    freqs /= freqs.sum()
    As = np.repeat(site.ds["Weibull_A"], repeat).values
    ks = np.repeat(site.ds["Weibull_k"], repeat).values
    samps = 75  # number of samples

    turbine = build_dtu10mw_wt()
    model = RANSDeficit()
    turbulence = RANSTurbulence()
    sim = WakeSimulation(turbine, model, turbulence)
    grad_func = jax.jit(
        jax.grad(
            lambda xx, yy, wsx, wdx: sim(xx, yy, wsx, wdx, ti_amb=TI_VALUE).aep(),
            argnums=(0, 1),
        )
    )

    def sampling():
        idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
        wd = dirs[idx]
        A = As[idx]
        k = ks[idx]
        ws = A * np.random.weibull(k)
        return wd, ws

    def aep_jac(x, y):
        wd, ws = sampling()
        dx, dy = grad_func(jnp.array(x), jnp.array(y), jnp.array(ws), jnp.array(wd))
        jac = np.array([np.atleast_2d(dx), np.atleast_2d(dy)]) * 1e3
        return jac

    layout = np.load(
        "./IEA_ModelChoice.AWAKEN_OptDriver.SGD_seed149_initial_pos.npy",
        allow_pickle=True,
    ).item()
    lx, ly = layout["x"], layout["y"]

    aep_jac(lx, ly)

    MAX_ITER = 1e5
    min_spacing_m = 4 * turbine.rotor_diameter
    bpath = "./IEA_740_10_scaled.json"
    boundaries = json.load(open(bpath))
    boundaries = (
        np.array([boundaries["boundaries_x"], boundaries["boundaries_y"]]).T
        * turbine.rotor_diameter
    )
    constraint_comp = XYBoundaryConstraint(boundaries, "polygon")
    nwt = 80
    boundary_poly = shapely.geometry.Polygon(boundaries)

    minx, miny, maxx, maxy = boundary_poly.bounds
    points = []
    it = 0
    while len(points) < nwt and it < int(MAX_ITER):
        # Generate random points within the bounding box
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = shapely.geometry.Point(x, y)
        min_distance_to_existing_ones = np.min(
            [point.distance(p) for p in points] if points else [np.inf]
        )
        if (
            boundary_poly.contains(point)
            and min_distance_to_existing_ones > min_spacing_m
        ):
            points.append(point)
        it += 1
    if it >= int(MAX_ITER):
        raise RuntimeError(
            "Could not generate enough points within the boundary. "
            "Increase the number of iterations or reduce the minimum spacing."
        )
    x0 = np.array([p.x for p in points])
    y0 = np.array([p.y for p in points])
    constraints = [SpacingConstraint(2), constraint_comp]
    cost_comp = CostModelComponent(
        input_keys=["x", "y"],
        n_wt=nwt,
        cost_function=lambda x, y: 0.0,
        objective=True,
        cost_gradient_function=aep_jac,
        maximize=True,
    )
    n_iter = 2000

    driver = EasySGDDriver(
        maxiter=n_iter,
        speedupSGD=True,
        learning_rate=0.1 * turbine.rotor_diameter,
        additional_constant_lr_iterations=100,
    )
    constraints = DistanceConstraintAggregation(
        constraint_comp,
        n_wt=nwt,
        min_spacing_m=2,
        windTurbines=None,
    )

    print("Running optimization...")
    OPTDIR = "./opt_results/"
    os.makedirs(OPTDIR, exist_ok=True)
    layout_name = "IEA"

    tf = TopFarmProblem(
        design_vars={"x": x0, "y": y0},
        cost_comp=cost_comp,
        constraints=constraints,
        driver=driver,
        plot_comp=XYPlotComp(
            folder_name=Path(OPTDIR).joinpath(Path(f"{layout_name}_{seed}_iters")),
            save_plot_per_iteration=True,
            memory=3,
            plot_initial=False,
            delay=0,
        ),
        reports=False,
    )
    _, state, rec = tf.optimize()

    optimized_positions = {"x": state["x"], "y": state["y"]}
    np.save(
        Path(OPTDIR).joinpath(
            Path(
                f"{layout_name}_{str('pixwake_surrogate')}_{'SGD'}_seed{seed}_opt_pos.npy"
            )
        ),
        optimized_positions,
    )
    np.save(
        Path(OPTDIR).joinpath(
            Path(
                f"{layout_name}_{str('pixwake_surrogate')}_{'SGD'}_seed{seed}_initial_pos.npy"
            )
        ),
        {"x": x0, "y": y0},
    )


import matplotlib.pyplot as plt
from glob import glob
import xarray as xr


def run_flow_cases():
    turbine = build_dtu10mw_wt()
    # rotor_avg_model=CGIRotorAvg(4)
    model = RANSDeficit()
    turbulence = RANSTurbulence()
    wfm = WakeSimulation(turbine, model, turbulence)

    RANS_DATA_PATHS = [
        y for y in glob(os.path.join("rans-data", "*.nc")) if "aero" not in y
    ]

    # flow_map, (fx, fy) = sim.flow_map(lx, ly, ti=0.06, wd=270, ws=10.0)
    # from pixwake.plot import plot_flow_map
    # plot_flow_map(fx, fy, flow_map, show=False)
    # import matplotlib.pyplot as plt
    # plt.savefig("rans_surrogate_flow_map_example.png")

    @jax.jit
    def _process_flow_case(ws, wd, ti, wtx, wty, fm_x, fm_y):
        effective_ws, _ = wfm.flow_map(
            wtx,
            wty,
            wd=270,
            ws=ws,
            ti=ti,
            fm_x=fm_x,
            fm_y=fm_y,
        )
        return {
            "wd": wd,
            "TI": ti,
            "ws": ws,
            "ws_eff": effective_ws,
        }

    for flowdb_path in RANS_DATA_PATHS:
        rans_dataset = xr.load_dataset(flowdb_path)
        pywake_res: xr.DataArray = rans_dataset.copy()
        pywake_res["U"] = pywake_res["U"] * 0
        pywake_res = pywake_res.sel(z=turbine.hub_height)

        flow_cases = [
            (ws, wd, ti)
            for ti in rans_dataset["TI"].values
            for wd in rans_dataset["wd"].values
            for ws in rans_dataset["ws"].values
        ]
        fm_x, fm_y = jnp.meshgrid(
            rans_dataset["x"].values,
            rans_dataset["y"].values,
        )
        fm_x, fm_y = fm_x.ravel(), fm_y.ravel()

        def block(x):
            x["ws_eff"].block_until_ready()
            return x

        results = [
            block(
                _process_flow_case(
                    ws,
                    wd,
                    ti,
                    rans_dataset.sel(wd=wd)["wt_x"].values,
                    rans_dataset.sel(wd=wd)["wt_y"].values,
                    fm_x,
                    fm_y,
                )
            )
            for ws, wd, ti in flow_cases
        ]

        for res in results:
            pywake_res["U"].loc[
                {
                    "ws": float(res["ws"]),
                    "wd": float(res["wd"]),
                    "TI": round(float(res["TI"]), 3),
                }
            ] = (
                res["ws_eff"]
                .reshape((rans_dataset["y"].size, rans_dataset["x"].size))
                .T
            )

        # pywake_res.to_netcdf(
        #     Path(args.prefix).joinpath(Path(flowdb_path.split("/")[-1]))
        # )

        plotws = 14.0
        plt.figure()
        fig_name = flowdb_path.split("/")[-1].split(".")[0]
        pywake_res["U"].sel(TI=0.1, wd=270, ws=plotws).plot(x="x", y="y")
        plt.axis("equal")
        plt.savefig(f"{fig_name}_pywake.png")

        plt.figure()
        (
            rans_dataset["U"].sel(TI=0.1, wd=270, ws=plotws, z=turbine.hub_height)
            * plotws
        ).plot(x="x", y="y")
        plt.axis("equal")
        plt.savefig(f"{fig_name}_rans.png")

        plt.figure()
        (
            pywake_res["U"].sel(TI=0.1, wd=270, ws=plotws)
            - (
                rans_dataset["U"].sel(TI=0.1, wd=270, ws=plotws, z=turbine.hub_height)
                * plotws
            )
        ).plot(x="x", y="y", cmap="RdBu_r")
        plt.axis("equal")
        plt.savefig(f"{fig_name}_error.png")
        # plt.close("all")
        # plt.show()

        rans_dataset["U"] = rans_dataset["U"] * rans_dataset["ws"]
        print(
            f"MAE for {flowdb_path.split('/')[-1]}: {np.abs(pywake_res['U'].values - rans_dataset['U'].sel(z=turbine.hub_height).values).mean()}"
        )


if __name__ == "__main__":
    # run_flow_cases()
    # exit()
    # for _ in range(100):
    #     test_rans_surrogate_aep()
    #     # exit()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_opt(seed=args.seed - 1)
    exit()

    for i in range(16):
        path = (
            "./IEA_ModelChoice.AWAKEN_OptDriver.SGD_seed149_opt_pos.npy"
            if i == 0
            else f"./opt_results/IEA_pixwake_surrogate_SGD_seed{42 + i - 1}_opt_pos.npy"
        )
        eval_layouts = np.load(
            path,
            allow_pickle=True,
        ).item()
        lx, ly = eval_layouts["x"], eval_layouts["y"]
        site, site_ws, site_wd = load_opt_site_and_reference_resource()
        site_wd = jnp.arange(0, 360 - 1, 1)
        pix_ws, pix_wd = jnp.meshgrid(site_ws, site_wd)
        pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()
        P_ilk = site.local_wind(ws=site_ws, wd=site_wd).P_ilk
        pix_probs = P_ilk.reshape((1, pix_wd.size)).T.squeeze()
        TI_VALUE = 0.06

        turbine = build_dtu10mw_wt()
        model = RANSDeficit()
        turbulence = RANSTurbulence()
        sim = WakeSimulation(turbine, model, turbulence)

        aep, _ = sim.aep_gradients_chunked(
            jnp.array(lx),
            jnp.array(ly),
            pix_ws,
            pix_wd,
            ti_amb=TI_VALUE,
            probabilities=pix_probs,
            chunk_size=128,
        )
        with open("evallog.txt", "a") as file:
            print(f"Seed {i} -> AEP: {aep}", file=file)
            print(f"Seed {i} -> AEP: {aep}")
