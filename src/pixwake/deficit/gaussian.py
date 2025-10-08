from typing import Any, Callable, cast

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficitModel


class BastankhahGaussianDeficit(WakeDeficitModel):
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

    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Compute wake deficits at receiver locations.

        Args:
            ws_eff: Effective wind speeds at source turbines (n_sources,).
            ctx: Simulation context with turbine and wind data.
            xs_r: Receiver x-coordinates. Defaults to ctx.xs.
            ys_r: Receiver y-coordinates. Defaults to ctx.ys.
            ti_eff: Effective turbulence intensity at sources. Defaults to ctx.ti.

        Returns:
            Effective wind speeds at receivers after wake deficits (n_receivers,).
        """
        xs_r = xs_r if xs_r is not None else ctx.xs
        ys_r = ys_r if ys_r is not None else ctx.ys

        downwind_dist, crosswind_dist = self.get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, xs_r, ys_r, ctx.wd
        )

        # Compute wake parameters for all source-receiver pairs
        wake_params = self._compute_wake_parameters(ws_eff, ctx, downwind_dist, ti_eff)

        # Compute deficit matrix (receivers x sources)
        deficit_matrix = self._compute_deficit_matrix(
            wake_params, downwind_dist, crosswind_dist, ws_eff, ctx
        )

        # Apply wake superposition (quadratic sum) and return effective wind speed
        eps = get_float_eps()
        total_deficit = jnp.sqrt(jnp.sum(deficit_matrix**2, axis=1) + eps)
        return jnp.maximum(0.0, ctx.ws - total_deficit)

    def _compute_wake_parameters(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        downwind_dist: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
    ) -> dict[str, jnp.ndarray]:
        """Compute wake expansion parameters.

        Returns:
            Dictionary with 'beta', 'epsilon', 'sigma', 'wake_radius', 'ct'.
        """
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
        ti_for_expansion = ti_eff if ti_eff is not None else ctx.ti
        k_expansion = jnp.asarray(self.wake_expansion_coefficient(ti_for_expansion))
        sigma_normalized = (
            k_expansion * downwind_dist / diameter + epsilon[None, :]
        )  # (n_receivers, n_sources)

        # Dimensional wake radius (2*sigma per Niayifar)
        wake_radius = 2.0 * sigma_normalized * diameter

        return {
            "beta": beta,
            "epsilon": epsilon,
            "sigma_normalized": sigma_normalized,
            "wake_radius": wake_radius,
            "ct": ct,
        }

    def _compute_deficit_matrix(
        self,
        wake_params: dict[str, jnp.ndarray],
        downwind_dist: jnp.ndarray,
        crosswind_dist: jnp.ndarray,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Compute absolute wake deficit matrix.

        Returns:
            Deficit matrix (n_receivers, n_sources) in m/s.
        """
        eps = get_float_eps()
        sigma = wake_params["sigma_normalized"]
        ct = wake_params["ct"]
        diameter = ctx.turbine.rotor_diameter

        # Effective thrust coefficient accounting for wake expansion
        ct_effective = ct[None, :] / (8.0 * sigma**2 + eps)

        # Centerline deficit (as fraction of reference wind speed)
        centerline_deficit = jnp.minimum(1.0, 2.0 * self.ct2a(ct_effective))

        # Gaussian radial decay
        sigma_dimensional = sigma * diameter
        radial_factor = jnp.exp(
            -(crosswind_dist**2) / (2.0 * sigma_dimensional**2 + eps)
        )

        # Total deficit as fraction
        deficit_fraction = centerline_deficit * radial_factor

        # Convert to absolute deficit (m/s)
        ws_reference = (
            ws_eff if self.use_effective_ws else jnp.full_like(ws_eff, ctx.ws)
        )
        deficit_absolute = deficit_fraction * ws_reference[None, :]

        # Apply masks: upstream and optionally wake radius
        is_downstream = downwind_dist > 0
        if self.use_radius_mask:
            is_inside_wake = jnp.abs(crosswind_dist) <= wake_params["wake_radius"]
            mask = is_downstream & is_inside_wake
        else:
            mask = is_downstream

        return jnp.where(mask, deficit_absolute, 0.0)

    def wake_expansion_coefficient(
        self, ti: jnp.ndarray | None = None
    ) -> jnp.ndarray | float:
        """Get wake expansion coefficient (constant for Bastankhah model)."""
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
        turbulence_model: Any | None = None,
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
        self.turbulence_model = turbulence_model

        if use_effective_ti and turbulence_model is None:
            raise ValueError(
                "turbulence_model must be provided when use_effective_ti=True"
            )

    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Compute wake deficits with optional effective TI calculation.

        When use_effective_ti is False, behaves like BastankhahGaussianDeficit.
        When True, computes wake-added turbulence and returns both wind speed
        and turbulence intensity.

        Returns:
            If use_effective_ti is False: Effective wind speeds (n_receivers,).
            If use_effective_ti is True: Tuple of (ws_eff, ti_eff).
        """
        if not self.use_effective_ti:
            return super().compute_deficit(ws_eff, ctx, xs_r, ys_r, ti_eff)

        if ctx.ti is None:
            raise ValueError("ctx.ti must be provided when use_effective_ti=True")

        # Initialize effective TI with ambient value
        if ti_eff is None:
            ti_eff = jnp.full_like(ws_eff, ctx.ti)

        # Compute effective TI including wake-added turbulence
        ti_eff_updated = self._compute_effective_ti(ws_eff, ctx, ti_eff)

        # Compute wind speeds using updated TI for wake expansion
        ws_eff_updated = cast(
            jnp.ndarray,
            super().compute_deficit(ws_eff, ctx, xs_r, ys_r, ti_eff_updated),
        )

        return ws_eff_updated, ti_eff_updated

    def _compute_effective_ti(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        ti_eff: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective turbulence intensity including wake-added turbulence.

        Args:
            ws_eff: Effective wind speeds at turbines (n_turbines,).
            ctx: Simulation context.
            ti_eff: Current effective TI at turbines (n_turbines,).

        Returns:
            Updated effective TI at each turbine (n_turbines,).
        """
        assert self.turbulence_model is not None

        # Get turbine-to-turbine distances
        downwind_dist, crosswind_dist = self.get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, ctx.xs, ctx.ys, ctx.wd
        )

        # Compute wake parameters using current effective TI
        wake_params = self._compute_wake_parameters(ws_eff, ctx, downwind_dist, ti_eff)

        # Compute wake-added turbulence
        return self.turbulence_model(
            ctx=ctx,
            ws_eff=ws_eff,
            dw=downwind_dist,
            cw=crosswind_dist,
            ti_eff=ti_eff,
            wake_radius=wake_params["wake_radius"],
            ct=wake_params["ct"],
        )

    def wake_expansion_coefficient(
        self, ti: jnp.ndarray | None = None
    ) -> jnp.ndarray | float:
        """Calculate TI-dependent wake expansion coefficient: k = a0 * TI + a1."""
        if ti is None:
            raise ValueError("Turbulence intensity required for Niayifar model")
        a0, a1 = self.a
        return a0 * ti + a1
