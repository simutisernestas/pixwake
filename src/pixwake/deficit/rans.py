import os
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization
from flax.struct import field

from ..core import SimulationContext
from .base import WakeDeficitModel


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


class RANSDeficit(WakeDeficitModel):
    """A RANS surrogate model for wake prediction.

    This model uses two pre-trained neural networks to predict the wake deficit
    and added turbulence intensity. The model is based on high-fidelity RANS
    CFD simulations.
    """

    def __init__(self, ambient_ti: float) -> None:
        """Initializes the RANSDeficit.

        Args:
            ambient_ti: The ambient turbulence intensity.
        """
        super().__init__()
        self.ambient_ti = ambient_ti
        (
            self.deficit_model,
            self.deficit_weights,
            self.turbulence_model,
            self.ti_weights,
        ) = load_rans_models()

    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        use_effective: bool = True,
    ) -> jnp.ndarray:
        """Computes the wake deficit using the RANS surrogate model.

        This method calculates the velocity deficit and added turbulence
        intensity for each turbine in the wind farm. It offers two modes of
        operation controlled by the `use_effective` parameter.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.
            use_effective: A boolean flag to control the deficit calculation.
                - If True (default), the deficit is calculated as an absolute
                  reduction in wind speed, proportional to the effective wind
                  speed at the waking turbine. This is more physically realistic.
                - If False, the deficit is calculated as a fractional reduction
                  relative to the free-stream wind speed.

        Returns:
            An array of updated effective wind speeds at each turbine.
        """
        x_d, y_d = self.get_downwind_crosswind_distances(ctx.xs, ctx.ys, ctx.wd)
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

        effective_ti = self.ambient_ti + _predict(
            self.turbulence_model, self.ti_weights, self.ambient_ti
        )

        deficit = _predict(self.deficit_model, self.deficit_weights, effective_ti)

        if use_effective:
            deficit *= ws_eff
            return jnp.maximum(0.0, ctx.ws - deficit)  # (N,)

        return jnp.maximum(0.0, ctx.ws * (1.0 - deficit))

    def flow_map(self, ws_eff: jnp.ndarray, ctx: SimulationContext) -> jnp.ndarray:
        if ctx.x is None or ctx.y is None:
            raise ValueError("x and y coordinates must be provided for flow map.")

        x_d, y_d = self._get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, ctx.x, ctx.y, ctx.wd
        )

        x_d /= ctx.turbine.rotor_diameter
        y_d /= ctx.turbine.rotor_diameter
        ct_eff = ctx.turbine.ct(ws_eff)

        in_domain_mask = (x_d < 70) & (x_d > -3) & (jnp.abs(y_d) < 6)

        def _predict(
            model: nn.Module, params: Any, ti: float | jnp.ndarray
        ) -> jnp.ndarray:
            """A helper function to run predictions with the Flax models."""
            md_input = jnp.stack(
                [
                    x_d,
                    y_d,
                    jnp.zeros_like(x_d),
                    jnp.full_like(x_d, ti),
                    jnp.zeros_like(x_d),
                    jnp.broadcast_to(ct_eff, x_d.shape),
                ],
                axis=-1,
            ).reshape(-1, 6)
            nn_out = jnp.array(model.apply(params, md_input)).reshape(x_d.shape)
            return jnp.where(in_domain_mask, nn_out, 0.0).sum(axis=1)

        effective_ti = self.ambient_ti + _predict(
            self.turbulence_model, self.ti_weights, self.ambient_ti
        )

        deficit = _predict(self.deficit_model, self.deficit_weights, effective_ti)

        return jnp.maximum(0.0, ctx.ws * (1.0 - deficit))
