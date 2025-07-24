import jax.numpy as jnp

from ..utils import get_eps
from .base import WakeModel


class NOJModel(WakeModel):
    def compute_deficit(self, ws_eff, state):
        x_d, y_d = self.get_downwind_crosswind_distances(state.xs, state.ys, state.wd)
        wake_rad = (state.turbine.rotor_diameter / 2) + self.k * x_d

        # mask upstream turbines within wake cone
        mask = (x_d > 0) & (jnp.abs(y_d) < wake_rad)

        # interpolate CT curve
        ct = jnp.interp(
            ws_eff, state.turbine.ct_curve.wind_speed, state.turbine.ct_curve.values
        )

        # wake deficit formulation
        a_coef = ct * (0.2460 + ct * (0.0586 + ct * 0.0883))
        term = (
            2
            * a_coef
            * ((state.turbine.rotor_diameter / 2) / jnp.maximum(wake_rad, get_eps()))
            ** 2
        )

        # combine deficits in quadrature
        deficits = jnp.sqrt(jnp.sum(jnp.where(mask, term**2, 0.0), axis=1) + get_eps())

        # new effective speed
        return jnp.maximum(0.0, state.ws * (1.0 - deficits))
