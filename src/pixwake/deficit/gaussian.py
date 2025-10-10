from typing import Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..turbulence.base import WakeTurbulence
from ..utils import ct2a_madsen
from .base import WakeDeficit


class BastankhahGaussianDeficit(WakeDeficit):
    """Bastankhah-Gaussian wake deficit model.

    Implementation of the Gaussian wake model from Bastankhah and Porte-Agel (2014).
    The wake deficit is computed using a Gaussian radial profile with expansion
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
        use_radius_mask: bool = False,
    ) -> None:
        """Initialize the Bastankhah-Gaussian wake deficit model.

        Args:
            k: Wake expansion coefficient (default from Bastankhah 2014).
            ceps: Near-wake coefficient for initial wake expansion.
            ctlim: Maximum thrust coefficient for numerical stability.
            ct2a: Function to convert thrust coefficient to induction factor.
            use_effective_ws: If True, use effective wind speed as reference for
                deficit calculation instead of ambient wind speed.
            use_radius_mask: If True, apply wake only within 2*sigma radius.
        """
        super().__init__()
        self.k = k
        self.ceps = ceps
        self.ctlim = ctlim
        self.ct2a = ct2a
        self.use_effective_ws = use_effective_ws
        self.use_radius_mask = use_radius_mask

    def compute(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute wake deficits at receiver locations.

        Args:
            ws_eff: Effective wind speeds at source turbines (n_sources,).
            ti_eff: Effective turbulence intensity at sources. Defaults to ctx.ti.
            ctx: Simulation context with turbine and wind data.

        Returns:
            Effective wind speeds at receivers after wake deficits (n_receivers,).
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
        sigma_normalized = (
            k_expansion * ctx.dw / diameter + epsilon[None, :]
        )  # (n_receivers, n_sources)

        # Dimensional wake radius (2*sigma per Niayifar)
        wake_radius = 2.0 * sigma_normalized * diameter
        # if not self.use_radius_mask:
        #     wake_radius = jnp.full_like(wake_radius, jnp.inf)

        diameter = ctx.turbine.rotor_diameter

        # Effective thrust coefficient accounting for wake expansion
        ct_effective = ct[None, :] / (8.0 * sigma_normalized**2 + eps)

        # Centerline deficit (as fraction of reference wind speed)
        centerline_deficit = jnp.minimum(1.0, 2.0 * self.ct2a(ct_effective))

        # Gaussian radial decay
        sigma_dimensional = sigma_normalized * diameter
        radial_factor = jnp.exp(-(ctx.cw**2) / (2.0 * sigma_dimensional**2 + eps))

        # Total deficit as fraction
        deficit_fraction = centerline_deficit * radial_factor

        # Convert to absolute deficit (m/s)
        ws_reference = (
            ws_eff if self.use_effective_ws else jnp.full_like(ws_eff, ctx.ws)
        )
        return deficit_fraction * ws_reference[None, :], wake_radius

    def wake_expansion_coefficient(
        self, ti_amb: jnp.ndarray | None, ti_eff: jnp.ndarray | None
    ) -> jnp.ndarray | float:
        """Get wake expansion coefficient (constant for Bastankhah model)."""
        _ = (ti_amb, ti_eff)  # unused
        return self.k


class NiayifarGaussianDeficit(BastankhahGaussianDeficit):
    """Niayifar-Gaussian wake deficit model with TI-dependent wake expansion.

    Extends the Bastankhah model with turbulence-intensity-dependent wake
    expansion: k = a[0] * TI + a[1]. Optionally includes wake-added turbulence
    effects on wake expansion.

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
        """Initialize the Niayifar-Gaussian wake deficit model.

        Args:
            a: Tuple (a0, a1) for wake expansion: k = a0 * TI + a1.
            use_effective_ti: If True, compute effective TI including wake-added
                turbulence and use it for wake expansion.
            turbulence_model: Turbulence model for computing wake-added TI.
                Required if use_effective_ti is True.
            **kwargs: Additional arguments passed to BastankhahGaussianDeficit.
        """
        super().__init__(**kwargs)
        self.a = a
        self.use_effective_ti = use_effective_ti

    def wake_expansion_coefficient(
        self,
        ti: jnp.ndarray | None,
        ti_eff: jnp.ndarray | None,
    ) -> jnp.ndarray | float:
        """Calculate TI-dependent wake expansion coefficient: k = a0 * TI + a1."""
        ti = ti_eff if self.use_effective_ti else ti
        assert ti is not None
        a0, a1 = self.a
        return a0 * ti + a1
