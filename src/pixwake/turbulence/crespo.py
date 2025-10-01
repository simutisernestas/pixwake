from dataclasses import dataclass, field

import jax.numpy as jnp

from pixwake.core import SimulationContext
from pixwake.deficit.utils import ct2a_madsen
from pixwake.turbulence.base import TurbulenceModel


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
    ct2a: callable = ct2a_madsen

    def calc_added_turbulence(
        self,
        ctx: SimulationContext,
        ws_eff: jnp.ndarray,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        ti_amb: jnp.ndarray,
        wake_radius: jnp.ndarray,
        ct: jnp.ndarray,  # TODO:
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


if __name__ == "__main__":
    turbulence_model = CrespoHernandez()

    from pixwake.core import Curve, Turbine

    # redundant basically for the turbulence compute; could just pass in CT array ?
    turbine = Turbine(
        rotor_diameter=8.0,
        hub_height=10.0,
        ct_curve=Curve(jnp.array([0, 25]), jnp.array([8 / 9, 0])),
        power_curve=Curve(jnp.array([0, 25]), jnp.array([0, 1])),
    )
    ctx = SimulationContext(
        xs=jnp.array([0.0, 200.0]),
        ys=jnp.array([0.0, 0.0]),
        ws=jnp.array([8.0, 8.0]),
        wd=jnp.array([0.0, 0.0]),
        turbine=turbine,
    )

    from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespo

    py_wake_model = PyWakeCrespo()

    n_turbines = 2
    import numpy as np

    # Define inputs similar to the PyWake call
    dw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    cw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    D_src_il = np.ones((n_turbines, 1)) * 8.0
    ct_ilk = np.ones((n_turbines, 1, 1)) * 8.0 / 9.0
    TI_ilk = np.ones((n_turbines, 1, 1)) * 0.1
    wake_radius_ijlk = np.ones((n_turbines, n_turbines, 1, 1)) * 4.0

    pywake_ti_res = py_wake_model.calc_added_turbulence(
        dw_ijlk=dw_ijlk,
        cw_ijlk=cw_ijlk,
        D_src_il=D_src_il,
        ct_ilk=ct_ilk,
        TI_ilk=TI_ilk,
        D_dst_ijl=None,
        wake_radius_ijlk=wake_radius_ijlk,
    ).squeeze()

    # Convert inputs to JAX arrays
    dw = jnp.array(dw_ijlk[:, :, 0, 0])
    cw = jnp.array(cw_ijlk[:, :, 0, 0])
    ti_amb = jnp.array(TI_ilk[:, 0, 0])
    wake_radius = jnp.array(wake_radius_ijlk[:, :, 0, 0])

    pixwake_ti_res = turbulence_model.calc_added_turbulence(
        ctx,
        ws_eff=jnp.array([0.1]),
        dw=dw,
        cw=cw,
        ti_amb=ti_amb,
        wake_radius=wake_radius,
        ct=jnp.array([8.0 / 9.0, 8.0 / 9.0]),
    )

    np.testing.assert_allclose(pixwake_ti_res, pywake_ti_res, rtol=1e-5, atol=1e-5)
