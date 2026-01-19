from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from pixwake.core import SimulationContext
from pixwake.jax_utils import NUMERICAL_FLOOR
from pixwake.turbulence.base import WakeTurbulence

from ..utils import ct2a_madsen


@dataclass
class CrespoHernandez(WakeTurbulence):
    """Implements the Crespo-Hernandez wake-added turbulence model.

    This is an empirical model for calculating the turbulence intensity added
    by a wind turbine's wake. The model is based on the thrust coefficient,
    ambient turbulence intensity, and the downwind distance from the turbine.

    Reference:
        Crespo, A., & Hernández, J. (1996). Turbulence characteristics in
        wind-turbine wakes. Journal of Wind Engineering and Industrial
        Aerodynamics, 61(1), 71-85.
    """

    c: tuple[float, float, float, float] = (0.73, 0.8325, -0.0325, -0.32)
    ct2a: Callable = ct2a_madsen

    def _added_turbulence(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Calculates wake-added turbulence using the Crespo-Hernandez model."""
        _ = ti_eff  # unused

        # Convert to induction factor with numerical safeguard
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        induction_factor = jnp.maximum(self.ct2a(ct), NUMERICAL_FLOOR)
        # Safeguard downwind distance for power law
        dw_safe = jnp.maximum(ctx.dw, NUMERICAL_FLOOR)
        # Normalized downwind distance
        distance_normalized = dw_safe / ctx.turbine.rotor_diameter
        # Apply Crespo-Hernandez formula using ambient ti Eq (21) in paper
        c0, c1, c2, c3 = self.c
        ti_ambient = ctx.ti  # scalar
        ti_added = (
            c0
            * induction_factor[None, :] ** c1
            * ti_ambient**c2  # type: ignore
            * distance_normalized**c3
        )
        return ti_added
