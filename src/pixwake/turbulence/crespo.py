from dataclasses import dataclass, field

import jax.numpy as jnp

from pixwake.core import SimulationContext
from pixwake.turbulence.base import TurbulenceModel

from ..utils import ct2a_madsen


@dataclass
class CrespoHernandez(TurbulenceModel):
    """
    Crespo-Hernandez turbulence model implementation, adapted from PyWake.

    This model calculates added turbulence based on the formulation by
    A. Crespo and J. Hernández in "Turbulence characteristics in wind-turbine
    wakes," J. of Wind Eng. and Industrial Aero. 61 (1996) 71-85.

    Attributes
    ----------
    c : list[float]
        A list of four coefficients used in the turbulence calculation.
        Defaults to [0.73, 0.8325, -0.0325, -0.32].
    ct2a : callable
        A function to convert thrust coefficient (Ct) to induction factor (a).
        Defaults to `ct2a_madsen`.
    """

    c: list[float] = field(default_factory=lambda: [0.73, 0.8325, -0.0325, -0.32])
    ct2a: callable = ct2a_madsen  # TODO: repeated from deficit models...

    def calc_added_turbulence(
        self,
        ctx: SimulationContext,
        ws_eff: jnp.ndarray,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        ti_amb: jnp.ndarray,
        wake_radius: jnp.ndarray,
        ct: jnp.ndarray,  # TODO: only added to match py_wake...
    ) -> jnp.ndarray:
        """
        Calculates the added turbulence intensity (TI) using the Crespo-Hernandez model.

        Parameters
        ----------
        ctx : SimulationContext
            The simulation context.
        ws_eff : jnp.ndarray
            The effective wind speed at each turbine.
        dw : jnp.ndarray
            The downwind distance between all pairs of turbines.
        cw : jnp.ndarray
            The crosswind distance between all pairs of turbines.
        ti_amb : jnp.ndarray
            The ambient turbulence intensity at each source turbine.

        Returns
        -------
        jnp.ndarray
            An array representing the added turbulence intensity at each
            turbine from each other turbine.
        """
        if ct is None:
            ct = ctx.turbine.ct(ws_eff)
        a = self.ct2a(ct)

        # Ensure induction factor 'a' is not too small to avoid NaN in gradients
        a = jnp.maximum(a, 1e-10)

        # Ensure downwind distance is positive to avoid issues with power laws
        dw_gt0 = jnp.maximum(dw, 1e-10)

        # Crespo-Hernandez formula for added turbulence (Eq. 21 in the paper)
        # The formula is applied for each source turbine's effect on each destination turbine.
        ti_add = (
            self.c[0]
            * a[None, :] ** self.c[1]
            * ti_amb[None, :] ** self.c[2]
            * (dw_gt0 / ctx.turbine.rotor_diameter) ** self.c[3]
        )

        # Turbulence is only added inside the wake and for downwind positions
        is_inside_wake = jnp.abs(cw) < wake_radius
        is_downwind = dw > 0
        ti_add_filtered = jnp.where(
            jnp.logical_and(is_inside_wake, is_downwind), ti_add, 0
        )

        return ti_add_filtered
