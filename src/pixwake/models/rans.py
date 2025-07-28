import os

import flax.linen as fnn
import jax
import jax.numpy as jnp
from flax import serialization

from .base import WakeModel


class WakeDeficitModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.02168894])
    mean_y = jnp.array([0.00614207])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.tanh(fnn.Dense(70)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


class WakeAddedTIModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.00571155])
    mean_y = jnp.array([0.0014295])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


def load_rans_models():
    def _load_model(model_class, filename):
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


class RANSModel(WakeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        (
            self.deficit_model,
            self.deficit_weights,
            self.turbulence_model,
            self.ti_weights,
        ) = load_rans_models()

    def compute_deficit(self, ws_eff, state, use_effective=True):
        x_d, y_d = self.get_downwind_crosswind_distances(state.xs, state.ys, state.wd)
        x_d /= state.turbine.rotor_diameter
        y_d /= state.turbine.rotor_diameter
        ct = jnp.interp(
            ws_eff, state.turbine.ct_curve.wind_speed, state.turbine.ct_curve.values
        )
        mask_off_diag = ~jnp.eye(x_d.shape[0], dtype=bool)
        in_domain_mask = (x_d < 70) & (x_d > -3) & (jnp.abs(y_d) < 6) & mask_off_diag

        def _predict(model, params, ti):
            md_input = jnp.stack(
                [
                    x_d,  # normalized x distance
                    y_d,  # normalized y distance
                    jnp.zeros_like(x_d),  # (z - h_hub) / D; evaluating at hub height
                    jnp.full_like(x_d, ti),  # turbulence intensity
                    jnp.zeros_like(x_d),  # yaw
                    jnp.broadcast_to(ct, x_d.shape),  # thrust coefficient
                ],
                axis=-1,
            ).reshape(-1, 6)
            output = model.apply(params, md_input).reshape(x_d.shape)
            return jnp.where(in_domain_mask, output, 0.0).sum(axis=1)

        effective_ti = self.ambient_ti + _predict(
            self.turbulence_model, self.ti_weights, self.ambient_ti
        )

        deficit = _predict(self.deficit_model, self.deficit_weights, effective_ti)

        if use_effective:
            deficit *= ws_eff
            return jnp.maximum(0.0, state.ws - deficit)  # (N,)

        return jnp.maximum(0.0, state.ws * (1.0 - deficit))
