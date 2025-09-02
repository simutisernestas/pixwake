from typing import Callable

import jax.numpy as jnp

from ..core import SimulationState
from ..utils import get_eps
from .base import WakeModel


def ct2a_madsen(
    ct: jnp.ndarray, ct2ap: tuple = (0.2460, 0.0586, 0.0883)
) -> jnp.ndarray:
    """
    BEM axial induction approximation by
    Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.: Implementation
    of the blade element momentum model on a polar grid and its aeroelastic load impact,
    Wind Energ. Sci., 5, 1–27, https://doi.org/10.5194/wes-5-1-2020, 2020.
    """
    # TODO: should unify between the NOJ and usage here !!!

    # Evaluate with Horner's rule.
    # ct2a_ilk = ct2ap[2] * ct_ilk**3 + ct2ap[1] * ct_ilk**2 + ct2ap[0] * ct_ilk
    return ct * (ct2ap[0] + ct * (ct2ap[1] + ct * ct2ap[2]))


class BastankhahGaussianDeficit(WakeModel):
    """A Bastankhah-Gaussian wake model.

    This model is based on the work of Bastankhah and Porte-Agel (2014),
    and the implementation in PyWake.
    """

    def __init__(
        self,
        k: float,
        use_effective_ws: bool = False,
        ceps: float = 0.2,
        ctlim: float = 0.899,
        ct2a: Callable = ct2a_madsen,
    ) -> None:
        """Initializes the BastankhahGaussianDeficit model.

        Args:
            k: The wake expansion coefficient.
            use_effective_ws: A boolean indicating whether to use the effective
                              wind speed in the deficit calculation.
        """
        super().__init__()
        self.k = k
        self.use_effective_ws = use_effective_ws
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a

    def compute_deficit(
        self, ws_eff: jnp.ndarray, state: SimulationState
    ) -> jnp.ndarray:
        """Computes the wake deficit using the Bastankhah-Gaussian model.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            state: The state of the simulation.

        Returns:
            An array of updated effective wind speeds at each turbine.
        """
        x_d, y_d = self.get_downwind_crosswind_distances(state.xs, state.ys, state.wd)

        # Get CT
        ct = jnp.interp(
            ws_eff, state.turbine.ct_curve.wind_speed, state.turbine.ct_curve.values
        )

        # Mask for upstream turbines
        mask = x_d > 0

        # Small epsilon to avoid division by zero and sqrt of negative numbers
        eps = get_eps()

        # According to the PyWake implementation:
        # beta = 1/2 * (1 + sqrt(1-ct)) / sqrt(1-ct)
        sqrt_1_minus_ct = jnp.sqrt(jnp.maximum(eps, 1.0 - jnp.minimum(self.ctlim, ct)))
        beta = 0.5 * (1.0 + sqrt_1_minus_ct) / sqrt_1_minus_ct

        # sigma_sqr = (k * dw / D + ceps * sqrt(beta))**2
        # In pixwake, D_src is a scalar.
        D_src = state.turbine.rotor_diameter

        # x_d is (n_turbines, n_turbines), D_src is a scalar.
        epsilon_ilk = self.ceps * jnp.sqrt(beta)
        sigma_term = self.k * x_d / D_src + epsilon_ilk
        sigma_sqr = sigma_term**2

        # ct_eff = ct / (8 * sigma_sqr)
        # ct needs to be from the source turbine, so ct[:, None]
        ct_eff = ct / (8.0 * sigma_sqr + eps)

        # deficit_centre = ws_ref * 2 * ct2a(ct_eff)
        # ws_ref is a scalar in the single-case simulation.
        deficit_centre = jnp.minimum(1.0, 2.0 * self.ct2a(ct_eff))

        # sigma_dimensional_sqr = sigma_sqr * D_src**2
        sigma_dimensional_sqr = sigma_sqr * (D_src**2)

        # exponent = -1 / (2 * sigma_dimensional_sqr) * cw**2
        exponent = -1.0 / (2.0 * sigma_dimensional_sqr + eps) * (y_d**2)

        # deficit = deficit_centre * exp(exponent)
        deficit_matrix = deficit_centre * jnp.exp(exponent)

        # Apply mask
        deficit_matrix = jnp.where(mask, deficit_matrix, 0.0)

        # Combine deficits in quadrature
        total_deficit = jnp.sqrt(jnp.sum(deficit_matrix**2, axis=1) + eps)

        # PyWake uses effective wind speed for the reference wind speed if the
        # use_effective_ws flag is set. In pixwake, inside the single-case
        # simulation, ws is a scalar.
        ws_ref = ws_eff if self.use_effective_ws else state.ws

        # New effective wind speed
        return jnp.maximum(0.0, ws_ref * (1.0 - total_deficit))
