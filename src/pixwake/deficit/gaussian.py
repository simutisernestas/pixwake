from typing import TYPE_CHECKING, Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficit

if TYPE_CHECKING:
    from ..rotor_avg import GaussianOverlapAvgModel


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
        rotor_avg_model: "GaussianOverlapAvgModel | None" = None,
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
            rotor_avg_model: An optional rotor averaging model. If a
                GaussianOverlapAvgModel is provided, its `deficit_model`
                attribute is automatically set to this instance.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(rotor_avg_model=rotor_avg_model, **kwargs)
        self.k = k
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a
        self.use_effective_ws = use_effective_ws

        # Set back-reference for GaussianOverlapAvgModel
        if rotor_avg_model is not None and hasattr(rotor_avg_model, "deficit_model"):
            rotor_avg_model.deficit_model = self

    def __wake_params(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def sigma(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Returns the dimensional wake width sigma.

        This method is used by the GaussianOverlapAvgModel for rotor averaging.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The dimensional wake width sigma (in meters).
        """
        sigma_normalized, _ = self.__wake_params(ws_eff, ti_eff, ctx)
        return sigma_normalized * ctx.turbine.rotor_diameter

    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
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
        rotor_avg_model: "GaussianOverlapAvgModel | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the `NiayifarGaussianDeficit` model.

        Args:
            a: A tuple `(a0, a1)` for the wake expansion formula.
            use_effective_ti: If `True`, use the effective turbulence intensity
                for the wake expansion calculation.
            rotor_avg_model: An optional rotor averaging model. If a
                GaussianOverlapAvgModel is provided, its `deficit_model`
                attribute is automatically set to this instance.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(rotor_avg_model=rotor_avg_model, **kwargs)
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


class TurboGaussianDeficit(WakeDeficit):
    """Implements the TurbOPark Gaussian wake deficit model.

    This model, implemented similar to Ørsted's TurbOPark model, uses a more
    sophisticated wake expansion formula that accounts for turbulence intensity
    effects through calibrated coefficients.

    Reference:
        https://github.com/OrstedRD/TurbOPark/blob/main/TurbOPark%20description.pdf
    """

    def __init__(
        self,
        A: float = 0.04,
        cTI: tuple[float, float] = (1.5, 0.8),
        ceps: float = 0.25,
        ctlim: float = 0.999,
        ct2a: Callable = ct2a_madsen,
        use_effective_ws: bool = False,
        use_effective_ti: bool = False,
        rotor_avg_model: "GaussianOverlapAvgModel | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the `TurboGaussianDeficit` model.

        Args:
            A: The wake expansion amplitude coefficient.
            cTI: A tuple `(c1, c2)` for the TI-dependent wake expansion.
            ceps: The near-wake coefficient for initial wake expansion.
            ctlim: The maximum thrust coefficient for numerical stability.
            ct2a: A callable to convert thrust coefficient to induction factor.
            use_effective_ws: If `True`, use the effective wind speed as the
                reference for deficit calculation.
            use_effective_ti: If `True`, use the effective turbulence intensity
                for the wake expansion calculation.
            rotor_avg_model: An optional rotor averaging model. If a
                GaussianOverlapAvgModel is provided, its `deficit_model`
                attribute is automatically set to this instance.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(rotor_avg_model=rotor_avg_model, **kwargs)
        self.A = A
        self.cTI = cTI
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti

        # Set back-reference for GaussianOverlapAvgModel
        if rotor_avg_model is not None and hasattr(rotor_avg_model, "deficit_model"):
            rotor_avg_model.deficit_model = self

    def _epsilon(self, ct: jnp.ndarray) -> jnp.ndarray:
        """Computes the near-wake epsilon parameter from thrust coefficient.

        Args:
            ct: The thrust coefficient array.

        Returns:
            The epsilon parameter for near-wake expansion.
        """
        ct_limited = jnp.minimum(ct, self.ctlim)
        sqrt_term = jnp.sqrt(1.0 - ct_limited)
        beta = 0.5 * (1.0 + sqrt_term) / sqrt_term
        return self.ceps * jnp.sqrt(beta)

    def __wake_params(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes wake parameters using TurbOPark formulation.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            A tuple of (sigma_normalized, ct) arrays.
        """
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        diameter = ctx.turbine.rotor_diameter

        # TI reference (either ambient or effective)
        ti_ref = ti_eff if self.use_effective_ti else ctx.ti
        assert ti_ref is not None, "TI must be provided for TurboGaussianDeficit"
        ti_ref = jnp.atleast_1d(jnp.asarray(ti_ref))

        c1, c2 = self.cTI
        alpha = c1 * ti_ref  # (n_sources,) or (1,)
        ct_safe = jnp.maximum(ct, 1e-20)
        beta = c2 * ti_ref / jnp.sqrt(ct_safe)  # (n_sources,)

        # Factor term (same as PyWake)
        fac = self.A * ti_ref * diameter / beta  # (n_sources,)

        # Broadcast to (1, n_sources) for matrix operations
        alpha = alpha[None, :]
        beta = beta[None, :]
        fac = fac[None, :]

        # Normalized downwind distance: (n_receivers, n_sources)
        dw_normalized = ctx.dw / diameter

        # Term calculations following TurbOPark formulation
        term1 = jnp.sqrt((alpha + beta * dw_normalized) ** 2 + 1)
        term2 = jnp.sqrt(1 + alpha**2)
        term3 = (term1 + 1) * alpha
        # Use gradient-safe absolute value: sqrt(x^2) instead of abs(x)
        dw_abs = jnp.sqrt(jnp.maximum(dw_normalized**2, 1e-20))
        term4 = (term2 + 1) * (alpha + beta * dw_abs)

        # Wake expansion (dimensional) - direct log like PyWake
        expansion = fac * (term1 - term2 - jnp.log(term3 / term4))

        # Add near-wake epsilon term
        epsilon = self._epsilon(ct)  # (n_sources,)
        sigma = expansion + epsilon[None, :] * diameter  # (n_receivers, n_sources)

        # Normalize by diameter
        sigma_normalized = sigma / diameter
        ct = ct[None, :]  # (1, n_sources)

        return sigma_normalized, ct

    def sigma(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Returns the dimensional wake width sigma.

        This method is used by the GaussianOverlapAvgModel for rotor averaging.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The dimensional wake width sigma (in meters).
        """
        sigma_normalized, _ = self.__wake_params(ws_eff, ti_eff, ctx)
        return sigma_normalized * ctx.turbine.rotor_diameter

    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Computes the wake deficit from each turbine.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The wake deficit matrix (n_receivers, n_sources) in m/s.
        """
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

    def _wake_radius(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Calculates the wake radius for each turbine.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The wake radius (2*sigma) for masking purposes.
        """
        sigma_normalized, _ = self.__wake_params(ws_eff, ti_eff, ctx)
        return 2.0 * sigma_normalized * ctx.turbine.rotor_diameter
