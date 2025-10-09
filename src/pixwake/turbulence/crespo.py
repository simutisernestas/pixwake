from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp

from pixwake.core import SimulationContext
from pixwake.turbulence.base import TurbulenceModel

from ..utils import ct2a_madsen


@dataclass
class CrespoHernandez(TurbulenceModel):
    """Crespo-Hernandez wake-added turbulence model.

    Empirical model for wake-added turbulence intensity based on thrust
    coefficient and distance downstream. The model uses ambient turbulence
    intensity (not effective TI) as per the original formulation.

    Reference:
        Crespo, A., & Hernández, J. (1996). Turbulence characteristics in
        wind-turbine wakes. Journal of Wind Engineering and Industrial
        Aerodynamics, 61(1), 71-85.

    Attributes:
        c: Four empirical coefficients [c0, c1, c2, c3] for the turbulence
           formula: TI_add = c0 * a^c1 * TI_ambient^c2 * (x/D)^c3.
        ct2a: Function to convert thrust coefficient to induction factor.
    """

    c: list[float] = field(default_factory=lambda: [0.73, 0.8325, -0.0325, -0.32])
    ct2a: Callable = ct2a_madsen

    def calc_added_turbulence(
        self,
        ctx: SimulationContext,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        wake_radius: jnp.ndarray,
        ct: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Calculate wake-added turbulence using Crespo-Hernandez formula.

        Note: This model uses AMBIENT turbulence intensity in the formula,
        not effective TI, as per the original formulation and PyWake implementation.

        Args:
            ctx: Simulation context.
            ws_eff: Effective wind speed at sources (n_sources,).
            dw: Downwind distances (n_receivers, n_sources).
            cw: Crosswind distances (n_receivers, n_sources).
            ti_eff: Effective TI at sources (unused in this model).
            wake_radius: Wake radius (n_receivers, n_sources).
            ct: Thrust coefficients at sources (n_sources,). Computed if not provided.

        Returns:
            Added turbulence intensity (n_receivers, n_sources).
        """
        # Convert to induction factor with numerical safeguard
        induction_factor = jnp.maximum(self.ct2a(ct), 1e-10)

        # Safeguard downwind distance for power law
        dw_safe = jnp.maximum(dw, 1e-10)

        # Normalized downwind distance
        distance_normalized = dw_safe / ctx.turbine.rotor_diameter

        # Apply Crespo-Hernandez formula using AMBIENT TI
        c0, c1, c2, c3 = self.c
        ti_ambient = ctx.ti  # scalar
        assert ti_ambient is not None
        ti_added = (
            c0
            * induction_factor[None, :] ** c1
            * ti_ambient**c2
            * distance_normalized**c3
        )

        # Apply spatial mask: inside wake and downstream only
        is_inside_wake = jnp.abs(cw) < wake_radius
        is_downstream = dw > 0
        mask = is_inside_wake & is_downstream

        return jnp.where(mask, ti_added, 0.0)
