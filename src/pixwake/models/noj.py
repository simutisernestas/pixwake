import jax.numpy as jnp

from ..core import SimulationState
from ..utils import get_eps
from .base import WakeModel


class NOJModel(WakeModel):
    """A Jensen NOJ (N.O. Jensen) wake model.

    This is a simple analytical model that assumes a linearly expanding wake.
    """

    def __init__(self, k: float) -> None:
        """Initializes the NOJModel.

        Args:
            k: The wake expansion coefficient.
        """
        super().__init__()
        self.k = k

    def compute_deficit(
        self, ws_eff: jnp.ndarray, state: SimulationState
    ) -> jnp.ndarray:
        """Computes the wake deficit using the NOJ model.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            state: The state of the simulation.

        Returns:
            An array of updated effective wind speeds at each turbine.
        """
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
