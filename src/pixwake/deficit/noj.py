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

    def __init__(
        self,
        k: float = 0.1,
        ct2a: Callable = ct2a_madsen,
        use_radius_mask: bool = True,
    ) -> None:
        """Initializes the NOJDeficit.

        Args:
            k: The wake expansion coefficient.
        """
        super().__init__(use_radius_mask)
        self.k = k
        self.ct2a = ct2a

    def compute(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        _ = ti_eff  # unused

        wt = ctx.turbine
        rr = wt.rotor_diameter / 2
        wake_radius = (rr) + self.k * ctx.dw
        all2all_deficit_matrix = (
            2 * self.ct2a(wt.ct(ws_eff))
            * (rr / jnp.maximum(wake_radius, get_float_eps())) ** 2
        )  # fmt: skip
        return ctx.ws * all2all_deficit_matrix, wake_radius
