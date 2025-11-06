from typing import Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficit


class BastankhahGaussianDeficit(WakeDeficit):
    """Implements the Bastankhah-Gaussian wake deficit model.

    This model, proposed by Bastankhah and Porte-Agel (2014), describes the
    wake deficit using a Gaussian radial profile. The expansion of the wake is
    proportional to the turbulence intensity.

    Reference:
        Bastankhah, M., & Porté-Agel, F. (2014). A new analytical model for
        wind-turbine wakes. Renewable Energy, 70, 116-123.
    """

    def __init__(
        self,
        k: float = 0.0324555,
        ceps: float = 0.2,
        ctlim: float = 0.899,
        ct2a: Callable = ct2a_madsen,
        use_effective_ws: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the `BastankhahGaussianDeficit` model.

        Args:
            k: The wake expansion coefficient.
            ceps: The near-wake coefficient for initial wake expansion.
            ctlim: The maximum thrust coefficient for numerical stability.
            ct2a: A callable to convert thrust coefficient to induction factor.
            use_effective_ws: If `True`, use the effective wind speed as the
                reference for deficit calculation.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.k = k
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a
        self.use_effective_ws = use_effective_ws

    def __wake_params(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ):
        # Compute wake parameters for all source-receiver pairs
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        diameter = ctx.turbine.rotor_diameter

        # Beta coefficient from Bastankhah formulation
        ct_limited = jnp.minimum(self.ctlim, ct)  # less than 1.0
        sqrt_term = jnp.sqrt(1.0 - ct_limited)
        beta = 0.5 * (1.0 + sqrt_term) / sqrt_term

        # Initial wake expansion (near-wake)
        w_epsilon = self.ceps * jnp.sqrt(beta)  # (n_sources,)

        # Wake width parameter (normalized by diameter)
        k_expansion = jnp.asarray(self.wake_expansion_coefficient(ctx.ti, ti_eff))

        w_epsilon = w_epsilon[None, :]
        ct = ct[None, :]

        sigma_normalized = (
            k_expansion * ctx.dw / diameter + w_epsilon
        )  # (n_receivers, n_sources)

        return sigma_normalized, ct

    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        feps = get_float_eps()
        sigma_normalized, ct = self.__wake_params(ws_eff, ti_eff, ctx)

        # Effective thrust coefficient accounting for wake expansion
        ct_effective = ct / (8.0 * sigma_normalized**2 + feps)

        # Centerline deficit (as fraction of reference wind speed)
        centerline_deficit = jnp.minimum(1.0, 2.0 * self.ct2a(ct_effective))

        # Gaussian radial decay
        sigma_dimensional = sigma_normalized * ctx.turbine.rotor_diameter
        radial_factor = jnp.exp(-(ctx.cw**2) / (2.0 * sigma_dimensional**2 + feps))

        # Total deficit as fraction
        deficit_fraction = centerline_deficit * radial_factor

        # Convert to absolute deficit (m/s)
        ws_reference = ws_eff[None, :]
        ws_reference = (
            ws_reference
            if self.use_effective_ws
            else jnp.full_like(ws_reference, ctx.ws)
        )
        return deficit_fraction * ws_reference

    def wake_expansion_coefficient(
        self, ti_amb: jnp.ndarray | None, ti_eff: jnp.ndarray | None
    ) -> jnp.ndarray | float:
        """Returns the wake expansion coefficient.

        For the `BastankhahGaussianDeficit` model, this is a constant value.
        """
        _ = (ti_amb, ti_eff)  # unused
        return self.k

    def _wake_radius(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        sigma_normalized, _ = self.__wake_params(ws_eff, ti_eff, ctx)
        # Dimensional wake radius (2*sigma per Niayifar)
        return 2.0 * sigma_normalized * ctx.turbine.rotor_diameter


class NiayifarGaussianDeficit(BastankhahGaussianDeficit):
    """Implements the Niayifar-Gaussian wake deficit model.

    This model extends the `BastankhahGaussianDeficit` by making the wake
    expansion coefficient dependent on the turbulence intensity, following the
    formulation `k = a[0] * TI + a[1]`.

    Reference:
        Niayifar, A., & Porté-Agel, F. (2016). Analytical modeling of wind farms:
        A new approach for power prediction. Energies, 9(9), 741.
    """

    def __init__(
        self,
        a: tuple[float, float] = (0.38, 4e-3),
        use_effective_ti: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the `NiayifarGaussianDeficit` model.

        Args:
            a: A tuple `(a0, a1)` for the wake expansion formula.
            use_effective_ti: If `True`, use the effective turbulence intensity
                for the wake expansion calculation.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.a = a
        self.use_effective_ti = use_effective_ti

    def wake_expansion_coefficient(
        self,
        ti: jnp.ndarray | None,
        ti_eff: jnp.ndarray | None,
    ) -> jnp.ndarray | float:
        """Calculates the TI-dependent wake expansion coefficient."""
        ti = ti_eff if self.use_effective_ti else ti
        assert ti is not None
        a0, a1 = self.a
        return a0 * ti + a1
