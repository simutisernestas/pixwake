from dataclasses import replace
from typing import Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficitModel


class BastankhahGaussianDeficit(WakeDeficitModel):
    """A Bastankhah-Gaussian wake model.

    This model is based on the work of Bastankhah and Porte-Agel (2014),
    and the implementation in PyWake.
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
        self.use_radius_mask = use_radius_mask

    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if xs_r is None:
            xs_r = ctx.xs
        if ys_r is None:
            ys_r = ctx.ys

        x_d, y_d = self.get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, xs_r, ys_r, ctx.wd
        )

        # per-source Ct (1d, length n_sources)
        ct_src = ctx.turbine.ct(ws_eff)

        # Mask for upstream turbines (receivers x sources)
        mask = x_d > 0

        eps = get_float_eps()

        # beta computed per source (1d)
        sqrt_1_minus_ct = jnp.sqrt(
            jnp.maximum(eps, 1.0 - jnp.minimum(self.ctlim, ct_src))
        )
        beta = 0.5 * (1.0 + sqrt_1_minus_ct) / sqrt_1_minus_ct

        D_src = ctx.turbine.rotor_diameter

        # epsilon per source (1d); make broadcasting explicit below
        epsilon_ilk = self.ceps * jnp.sqrt(beta)

        ti_for_expansion = ti_eff if ti_eff is not None else ctx.ti

        # sigma_term: x_d has shape (n_receivers, n_sources).
        # We add epsilon_ilk broadcast along the source axis (columns).
        # k can be scalar or per-source array
        k_expansion = self.wake_expansion_coefficient(ti_for_expansion)
        # Ensure k_expansion is broadcastable: if it's an array (ndim > 0), add None for receiver dim
        k_expansion_arr = jnp.asarray(k_expansion)
        sigma_term = (
            k_expansion_arr * x_d / D_src + epsilon_ilk[None, :]
        )  # shape (R, S)
        sigma_sqr = sigma_term**2

        # ct_eff_matrix per (receiver, source): use ct_src broadcast to source axis
        ct_eff_matrix = ct_src[None, :] / (8.0 * sigma_sqr + eps)  # shape (R, S)

        # deficit centre (fraction) per (receiver, source)
        deficit_centre = jnp.minimum(1.0, 2.0 * self.ct2a(ct_eff_matrix))

        sigma_dimensional_sqr = sigma_sqr * (D_src**2)

        exponent = -1.0 / (2.0 * sigma_dimensional_sqr + eps) * (y_d**2)

        # fraction of the source reference wind speed lost at each receiver
        deficit_fraction_matrix = deficit_centre * jnp.exp(exponent)  # shape (R, S)

        # get reference wind per source: either per-source effective ws or ambient ws
        if self.use_effective_ws:
            ws_ref_sources = ws_eff  # shape (S,)
        else:
            ws_ref_sources = jnp.full_like(ws_eff, ctx.ws)  # shape (S,)

        # convert fractions to absolute deficits (m/s) using each *source* reference
        deficit_abs_matrix = deficit_fraction_matrix * ws_ref_sources[None, :]  # (R, S)

        # according to Niayifar, the wake radius is twice sigma
        wake_radius = 2.0 * sigma_term * D_src  # dimensional radius

        if self.use_radius_mask:
            inside_wake = jnp.abs(y_d) <= wake_radius
            mask &= inside_wake

        # apply upstream mask
        deficit_abs_matrix = jnp.where(mask, deficit_abs_matrix, 0.0)
        # combine absolute deficits in quadrature and subtract from ambient
        total_abs_deficit = jnp.sqrt(jnp.sum(deficit_abs_matrix**2, axis=1) + eps)
        return jnp.maximum(0.0, ctx.ws - total_abs_deficit)

    def wake_expansion_coefficient(
        self, ti: jnp.ndarray | None = None
    ) -> jnp.ndarray | float:
        """Returns the wake expansion coefficient."""
        _ = ti  # ignored
        return self.k


class NiayifarGaussianDeficit(BastankhahGaussianDeficit):
    """A Niayifar-Gaussian wake model.

    Amin Niayifar and Fernando Porté-Agel
    Analytical Modeling of Wind Farms: A New Approach for Power Prediction
    Energies 2016, 9, 741; doi:10.3390/en9090741
    """

    def __init__(
        self,
        a: tuple = (0.38, 4e-3),
        use_effective_ti: bool = False,
        turbulence_model: Any | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.a = a
        self.use_effective_ti = use_effective_ti
        self.turbulence_model = turbulence_model

    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple:
        """Computes the wake deficit with optional effective TI calculation."""
        if not self.use_effective_ti:
            return super().compute_deficit(ws_eff, ctx, xs_r, ys_r, ti_eff)

        if self.turbulence_model is None:
            raise ValueError(
                "A turbulence model must be provided when use_effective_ti=True"
            )
        if ctx.ti is None:
            raise ValueError("Ambient TI must be provided when use_effective_ti=True")

        if ti_eff is None:
            ti_eff = jnp.full_like(ws_eff, ctx.ti)

        new_ti_eff = self._compute_effective_ti(ws_eff, ctx, ti_eff)
        new_ws_eff = super().compute_deficit(ws_eff, ctx, xs_r, ys_r, new_ti_eff)

        return new_ws_eff, new_ti_eff

    def _compute_effective_ti(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Computes the effective turbulence intensity at each turbine."""
        assert ctx.ti is not None, "TI must be provided"
        assert self.turbulence_model is not None, "Turbulence model must be provided"

        # Get downwind and crosswind distances between turbines
        x_d, y_d = self.get_downwind_crosswind_distances(
            ctx.xs, ctx.ys, ctx.xs, ctx.ys, ctx.wd
        )

        # Compute CT for each turbine
        ct_src = ctx.turbine.ct(ws_eff)

        # Compute wake radius using current TI (start with ambient)
        # For the Niayifar model, wake radius is 2*sigma
        D_src = ctx.turbine.rotor_diameter
        eps = get_float_eps()

        # beta computed per source (1d)
        sqrt_1_minus_ct = jnp.sqrt(
            jnp.maximum(eps, 1.0 - jnp.minimum(self.ctlim, ct_src))
        )
        beta = 0.5 * (1.0 + sqrt_1_minus_ct) / sqrt_1_minus_ct
        epsilon_ilk = self.ceps * jnp.sqrt(beta)

        # Use effective TI for wake radius calculation
        k_eff = self.wake_expansion_coefficient(ti_eff)
        sigma_term = k_eff[None, :] * x_d / D_src + epsilon_ilk[None, :]
        wake_radius = 2.0 * sigma_term * D_src

        # Compute added turbulence using the turbulence model
        ti_amb_array = jnp.full_like(ws_eff, ctx.ti)
        ti_add = self.turbulence_model.calc_added_turbulence(
            ctx=ctx,
            ws_eff=ws_eff,
            dw=x_d,
            cw=y_d,
            ti_eff=ti_eff,
            wake_radius=wake_radius,
            ct=ct_src,
        )

        # Compute effective TI using superposition
        return self.turbulence_model.superposition_model(ti_amb_array, ti_add)

    def wake_expansion_coefficient(
        self, ti: jnp.ndarray | None = None
    ) -> jnp.ndarray | float:
        """Calculates the wake expansion coefficient based on turbulence intensity."""
        if ti is None:
            raise ValueError("Turbulence intensity must be provided.")
        a0, a1 = self.a
        return a0 * ti + a1
