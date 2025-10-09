from typing import Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficit


class NOJDeficit(WakeDeficit):
    """A Jensen NOJ (N.O. Jensen) wake model.

    This is a simple analytical model that assumes a linearly expanding wake.
    """

    def __init__(self, k: float = 0.1, ct2a: Callable = ct2a_madsen) -> None:
        """Initializes the NOJDeficit.

        Args:
            k: The wake expansion coefficient.
        """
        super().__init__()
        self.k = k
        self.ct2a = ct2a

    def compute_deficit(
        self, ws_eff: jnp.ndarray, ti_eff: jnp.ndarray | None, ctx: SimulationContext
    ) -> jnp.ndarray:
        """Computes the wake deficit using the NOJ model.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.
            xs_r: An array of x-coordinates for each receiver point (optional).
            ys_r: An array of y-coordinates for each receiver point (optional).
            ti_eff: An array of effective turbulence intensities (optional, not used).

        Returns:
            An array of updated effective wind speeds at each turbine.
        """
        wake_rad = (ctx.turbine.rotor_diameter / 2) + self.k * ctx.dw
        # mask upstream turbines within wake cone
        mask = (ctx.dw > 0) & (jnp.abs(ctx.cw) < wake_rad)

        eps = get_float_eps()

        deficit_matrix = (
            2
            * self.ct2a(ctx.turbine.ct(ws_eff))
            * ((ctx.turbine.rotor_diameter / 2) / jnp.maximum(wake_rad, eps)) ** 2
        )
        deficit_matrix = jnp.where(mask, deficit_matrix, 0.0)

        return deficit_matrix
