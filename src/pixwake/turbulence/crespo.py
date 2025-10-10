from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp

from pixwake.core import SimulationContext
from pixwake.turbulence.base import WakeTurbulence

from ..utils import ct2a_madsen


@dataclass
class CrespoHernandez(WakeTurbulence):
    """Implements the Crespo-Hernandez wake-added turbulence model.

    This is an empirical model for calculating the turbulence intensity added
    by a wind turbine's wake. The model is based on the thrust coefficient,
    ambient turbulence intensity, and the downwind distance from the turbine.

    Attributes:
        c: A list of four empirical coefficients `[c0, c1, c2, c3]` used in the
           turbulence formula: `TI_add = c0 * a^c1 * TI_amb^c2 * (x/D)^c3`.
        ct2a: A callable that converts the thrust coefficient (`Ct`) to the
            induction factor (`a`).

    Reference:
        Crespo, A., & Hernández, J. (1996). Turbulence characteristics in
        wind-turbine wakes. Journal of Wind Engineering and Industrial
        Aerodynamics, 61(1), 71-85.
    """

    c: list[float] = field(default_factory=lambda: [0.73, 0.8325, -0.0325, -0.32])
    ct2a: Callable = ct2a_madsen

    def added_turbulence(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Calculates wake-added turbulence using the Crespo-Hernandez model.

        Note: This model uses the ambient turbulence intensity in its
        calculation, not the effective TI, as per the original formulation.

        Args:
            ws_eff: The effective wind speeds at the source turbines.
            ti_eff: The effective turbulence intensities at the source turbines
                (not used in this model).
            ctx: The simulation context.

        Returns:
            A JAX numpy array of the added turbulence intensity.
        """
        _ = ti_eff  # unused

        # Convert to induction factor with numerical safeguard
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        induction_factor = jnp.maximum(self.ct2a(ct), 1e-10)
        # Safeguard downwind distance for power law
        dw_safe = jnp.maximum(ctx.dw, 1e-10)
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
