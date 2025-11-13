import pytest

# TODO: could revive this as an example or generic surrogate model
pytest.skip("Specific case, not generic wake model", allow_module_level=True)

import os
import time
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from flax import serialization
from flax.struct import field
from jax.test_util import check_grads
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.base import WakeDeficit
from pixwake.core import SimulationContext


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


def load_rans_models() -> tuple[nn.Module, Any, nn.Module, Any]:
    """Loads the pre-trained RANS surrogate models.

    Returns:
        A tuple containing the deficit model, deficit weights, turbulence
        model, and turbulence weights.
    """

    def _load_model(
        model_class: type[nn.Module], filename: str
    ) -> tuple[nn.Module, Any]:
        """Loads a single Flax model from a file."""
        model = model_class()
        variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))
        with open(filename, "rb") as f:
            bytes_data = f.read()
        return model, serialization.from_bytes(variables, bytes_data)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    deficit_model, deficit_weights = _load_model(
        WakeDeficitModelFlax,
        os.path.join(file_dir, "./data/rans_deficit_surrogate.msgpack"),
    )
    turbulence_model, ti_weights = _load_model(
        WakeAddedTIModelFlax,
        os.path.join(file_dir, "./data/rans_addedti_surrogate.msgpack"),
    )
    return deficit_model, deficit_weights, turbulence_model, ti_weights


class RANSDeficit(WakeDeficit):
    """A RANS surrogate model for wake prediction.

    This model uses two pre-trained neural networks to predict the wake deficit
    and added turbulence intensity. The model is based on high-fidelity RANS
    CFD simulations.

    Note: This model requires turbulence intensity to be provided in the simulation
    context (ctx.ti). The TI value must be passed when calling the WakeSimulation.
    """

    def __init__(self, use_effective: bool = True) -> None:
        """Initializes the RANSDeficit.

        Args:
            use_effective: A boolean flag to control the deficit calculation.
                - If True (default), the deficit is calculated as an absolute
                  reduction in wind speed, proportional to the effective wind
                  speed at the waking turbine. This is more physically realistic.
                - If False, the deficit is calculated as a fractional reduction
                  relative to the free-stream wind speed.
        """
        super().__init__()
        self.use_effective = use_effective
        self.use_effective_ti = True
        (
            self.deficit_model,
            self.deficit_weights,
            self.turbulence_model,
            self.ti_weights,
        ) = load_rans_models()

    def compute(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        if ctx.ti is None:
            raise ValueError(
                "RANSDeficit requires turbulence intensity (ti) to be provided. "
                "Pass ti parameter when calling WakeSimulation."
            )

        if xs_r is None:
            xs_r = ctx.dw
        if ys_r is None:
            ys_r = ctx.cw

        x_d, y_d = self.get_downwind_crosswind_distances(
            ctx.dw, ctx.cw, xs_r, ys_r, ctx.wd
        )
        x_d /= ctx.turbine.rotor_diameter
        y_d /= ctx.turbine.rotor_diameter
        ct_eff = ctx.turbine.ct(ws_eff)
        mask_off_diag = ~jnp.eye(x_d.shape[0], dtype=bool)
        in_domain_mask = (x_d < 70) & (x_d > -3) & (jnp.abs(y_d) < 6) & mask_off_diag

        def _predict(
            model: nn.Module, params: Any, ti: float | jnp.ndarray
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
            nn_out = jnp.array(model.apply(params, md_input)).reshape(x_d.shape)
            return jnp.where(in_domain_mask, nn_out, 0.0).sum(axis=1)

        # Use effective TI if available, otherwise use ambient TI from context
        ti_input = ti_eff if ti_eff is not None else ctx.ti

        added_ti = _predict(self.turbulence_model, self.ti_weights, ti_input)

        # Ambient TI comes from context
        new_effective_ti = ctx.ti + added_ti

        deficit = _predict(self.deficit_model, self.deficit_weights, ti_input)

        if self.use_effective:
            deficit *= ws_eff
            new_ws_eff = jnp.maximum(0.0, ctx.ws - deficit)
        else:
            new_ws_eff = jnp.maximum(0.0, ctx.ws * (1.0 - deficit))

        return new_ws_eff, new_effective_ti


def get_rans_dependencies():
    turbine = DTU10MW()
    ct_xp = turbine.powerCtFunction.ws_tab
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]
    D = turbine.diameter()
    return ct_xp, ct_fp, pw_fp, D


def test_rans_surrogate_aep():
    ct_xp, ct_fp, pw_fp, D = get_rans_dependencies()
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    onp.random.seed(42)
    T = 10
    WSS = jnp.asarray(onp.random.uniform(CUTIN_WS, CUTOUT_WS, T))
    WDS = jnp.asarray(onp.random.uniform(0, 360, T))

    wi, le = 10, 8
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    turbine = Turbine(
        rotor_diameter=D,
        hub_height=100.0,
        power_curve=Curve(ws=ct_xp, values=pw_fp),
        ct_curve=Curve(ws=ct_xp, values=ct_fp),
    )

    model = RANSDeficit()
    sim = WakeSimulation(
        model, turbine, mapping_strategy="map", fpi_damp=0.8, fpi_tol=1e-3
    )

    def aep(xx, yy):
        return sim(xx, yy, WSS, WDS, ti=0.1).aep()

    aep_and_grad = jax.jit(jax.value_and_grad(aep, argnums=(0, 1)))

    def block_all(res):
        if isinstance(res, tuple):
            return tuple(block_all(r) for r in res)
        else:
            return res.block_until_ready()

    res = aep_and_grad(xs, ys)
    block_all(res)
    s = time.time()
    res = aep_and_grad(xs, ys)
    block_all(res)
    print(f"AEP: {res[0]} in {time.time() - s:.3f} seconds")

    assert jnp.isfinite(res[0]).all(), "AEP should be finite"
    assert jnp.isfinite(res[1][0]).all(), "Gradient of x should be finite"
    assert jnp.isfinite(res[1][1]).all(), "Gradient of y should be finite"


def test_rans_surrogate_gradients():
    ct_xp, ct_fp, pw_fp, D = get_rans_dependencies()
    ws = 9.0
    wd = 90.0
    wi, le = 3, 2
    xs, ys = jnp.meshgrid(
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()

    turbine = Turbine(
        rotor_diameter=D,
        hub_height=100.0,
        power_curve=Curve(ws=ct_xp, values=pw_fp),
        ct_curve=Curve(ws=ct_xp, values=ct_fp),
    )

    model = RANSDeficit()
    simulation = WakeSimulation(model, turbine, fpi_damp=0.8, fpi_tol=1e-3)

    def sim(x, y):
        return simulation(
            x, y, jnp.full_like(x, ws), jnp.full_like(x, wd), ti=0.1
        ).effective_ws.sum()

    check_grads(sim, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2, eps=10)
