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

    Attributes:
        k: Wake expansion coefficient.
        ceps: Near-wake coefficient for initial wake expansion.
        ctlim: Maximum thrust coefficient for numerical stability.
        ct2a: A callable that converts thrust coefficient to induction factor.
        use_effective_ws: A boolean indicating whether to use the effective
            wind speed as the reference for deficit calculation.

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
            rotor_avg_model: An optional rotor averaging model.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.k = k
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a
        self.use_effective_ws = use_effective_ws

    def compute(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the wake deficit at specified locations.

        Args:
            ws_eff: The effective wind speeds at the source turbines.
            ti_eff: The effective turbulence intensity at the source turbines.
            ctx: The simulation context.

        Returns:
            A tuple containing the wake deficit matrix and the wake radius.
        """
        # Compute wake parameters for all source-receiver pairs
        eps = get_float_eps()
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        diameter = ctx.turbine.rotor_diameter

        # Beta coefficient from Bastankhah formulation
        ct_limited = jnp.minimum(self.ctlim, ct)
        sqrt_term = jnp.sqrt(jnp.maximum(eps, 1.0 - ct_limited))
        beta = 0.5 * (1.0 + sqrt_term) / sqrt_term

        # Initial wake expansion (near-wake)
        epsilon = self.ceps * jnp.sqrt(beta)  # (n_sources,)

        # Wake width parameter (normalized by diameter)
        k_expansion = jnp.asarray(self.wake_expansion_coefficient(ctx.ti, ti_eff))

        if ctx.dw.ndim > 2:  # Rotor-averaged deficit
            # Reshape source-turbine dependent quantities to be broadcastable with
            # the integration points dimension.
            epsilon = epsilon[None, :, None]
            ct = ct[None, :, None]
            ws_reference = ws_eff[None, :, None]
            k_expansion = jnp.atleast_1d(k_expansion)[None, :, None]
        else:
            # Prepare for broadcasting without rotor averaging
            epsilon = epsilon[None, :]
            ct = ct[None, :]
            ws_reference = ws_eff[None, :]

        sigma_normalized = (
            k_expansion * ctx.dw / diameter + epsilon
        )  # (n_receivers, n_sources)

        # Dimensional wake radius (2*sigma per Niayifar)
        wake_radius = 2.0 * sigma_normalized * diameter

        diameter = ctx.turbine.rotor_diameter

        # Effective thrust coefficient accounting for wake expansion
        ct_effective = ct / (8.0 * sigma_normalized**2 + eps)

        # Centerline deficit (as fraction of reference wind speed)
        centerline_deficit = jnp.minimum(1.0, 2.0 * self.ct2a(ct_effective))

        # Gaussian radial decay
        sigma_dimensional = sigma_normalized * diameter
        radial_factor = jnp.exp(-(ctx.cw**2) / (2.0 * sigma_dimensional**2 + eps))

        # Total deficit as fraction
        deficit_fraction = centerline_deficit * radial_factor

        # Convert to absolute deficit (m/s)
        ws_reference = (
            ws_reference
            if self.use_effective_ws
            else jnp.full_like(ws_reference, ctx.ws)
        )
        return deficit_fraction * ws_reference, wake_radius

    def wake_expansion_coefficient(
        self, ti_amb: jnp.ndarray | None, ti_eff: jnp.ndarray | None
    ) -> jnp.ndarray | float:
        """Returns the wake expansion coefficient.

        For the `BastankhahGaussianDeficit` model, this is a constant value.
        """
        _ = (ti_amb, ti_eff)  # unused
        return self.k


class NiayifarGaussianDeficit(BastankhahGaussianDeficit):
    """Implements the Niayifar-Gaussian wake deficit model.

    This model extends the `BastankhahGaussianDeficit` by making the wake
    expansion coefficient dependent on the turbulence intensity, following the
    formulation `k = a[0] * TI + a[1]`.

    Attributes:
        a: A tuple `(a0, a1)` for the linear relationship between `k` and `TI`.
        use_effective_ti: A boolean indicating whether to use the effective
            turbulence intensity for the wake expansion calculation.

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
            rotor_avg_model: An optional rotor averaging model.
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
