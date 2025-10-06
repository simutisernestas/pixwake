from typing import Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficitModel


class NOJDeficit(WakeDeficitModel):
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
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Computes the wake deficit using the NOJ model.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.
            xs_r: An array of x-coordinates for each receiver point (optional).
            ys_r: An array of y-coordinates for each receiver point (optional).

        Returns:
            An array of updated effective wind speeds at each turbine.
        """
        if xs_r is None:
            xs_r = ctx.xs
        if ys_r is None:
            ys_r = ctx.ys

        x_d, y_d = self.get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, xs_r, ys_r, ctx.wd
        )
        wake_rad = (ctx.turbine.rotor_diameter / 2) + self.k * x_d

        # mask upstream turbines within wake cone
        mask = (x_d > 0) & (jnp.abs(y_d) < wake_rad)

        ct_eff = ctx.turbine.ct(ws_eff)

        # wake deficit formulation
        a_coef = self.ct2a(ct_eff)
        term = (
            2
            * a_coef
            * (
                (ctx.turbine.rotor_diameter / 2)
                / jnp.maximum(wake_rad, get_float_eps())
            )
            ** 2
        )

        # combine deficits in quadrature
        deficits = jnp.sqrt(
            jnp.sum(jnp.where(mask, term**2, 0.0), axis=1) + get_float_eps()
        )

        # new effective speed
        return jnp.maximum(0.0, ctx.ws * (1.0 - deficits))
