import sys

import pytest

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
import numpy as onp
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit.base import WakeDeficit
from pixwake.turbulence.base import WakeTurbulence


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

    def __init__(self, use_effective_ws=True, use_effective_ti=True, **kwargs) -> None:
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
            & ((x_d > 1e-3) | (x_d < -1e-3))
        )
        ti_input = ti_eff if ti_eff is not None else ctx.ti

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
        in_domain_mask = (x_d < 70) & (x_d > -3) & (jnp.abs(y_d) < 6)
        ti_input = ti_eff if ti_eff is not None else ctx.ti

        added_turbulence = _predict(
            self.turbulence_model,
            self.ti_weights,
            ti_input,
            x_d,
            y_d,
            ct_eff,
        )
        return jnp.where(in_domain_mask, added_turbulence, 0.0)


from scipy.ndimage import gaussian_filter1d


# hahah it actually works...
def smooth_curve(ws, values, sigma=0.5):
    """Apply Gaussian smoothing to make curve differentiable"""
    smoothed = gaussian_filter1d(values, sigma=sigma, mode="nearest")
    return ws, smoothed


def build_dtu10mw_wt(smooth=True) -> Turbine:
    pywake_turbine = DTU10MW()
    ws = jnp.linspace(0, 30, 301).tolist() + [100.0]

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


def test_rans_surrogate_aep():
    CUTOUT_WS = 25.0
    CUTIN_WS = 4.0

    onp.random.seed(42)
    T = 100
    WSS = jnp.asarray(onp.random.uniform(CUTIN_WS, CUTOUT_WS, T))
    WDS = jnp.asarray(onp.random.uniform(0, 360, T))

    turbine = build_dtu10mw_wt()
    wi, le = 3, 3
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 2 * turbine.rotor_diameter, wi),
        jnp.linspace(0, le * 2 * turbine.rotor_diameter, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    # add some noise to positions
    xs += onp.random.normal(0, turbine.rotor_diameter, xs.shape)
    ys += onp.random.normal(0, turbine.rotor_diameter, ys.shape)

    assert xs.shape[0] == (wi * le), xs.shape

    model = RANSDeficit()
    turbulence = RANSTurbulence()
    sim = WakeSimulation(
        turbine,
        model,
        turbulence,
        fpi_damp=1.0,
        fpi_tol=1e-6,
    )

    # flow_map, (fx, fy) = sim.flow_map(xs, ys, ti=0.1, wd=270)  # warm-up
    # from pixwake.plot import plot_flow_map
    # plot_flow_map(fx, fy, flow_map, show=True)
    # exit()

    def aep(xx, yy):
        return sim(xx, yy, WSS, WDS, 0.1).aep()

    aep_and_grad = jax.jit(jax.value_and_grad(aep, argnums=(0, 1)))

    res = aep_and_grad(xs, ys)
    block_all(res)
    s = time.time()
    res = aep_and_grad(xs, ys)
    block_all(res)
    print(f"AEP: {res[0]} in {time.time() - s:.3f} seconds")

    assert jnp.isfinite(res[0]).all(), "AEP should be finite"
    assert jnp.isfinite(res[1][0]).all(), "Gradient of x should be finite"
    assert jnp.isfinite(res[1][1]).all(), "Gradient of y should be finite"


if __name__ == "__main__":
    test_rans_surrogate_aep()
