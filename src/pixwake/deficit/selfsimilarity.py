"""Self-similarity blockage deficit models.

These models calculate the blockage (induction) effect upstream of wind turbines
based on self-similarity theory. Unlike wake deficit models that operate downstream,
blockage models reduce wind speed upstream of the rotor due to the pressure field
induced by the rotor's thrust.

Reference:
    N. Troldborg, A.R. Meyer Forsting, "A simple model of the wind turbine
    induction zone derived from numerical simulations", Wind Energy, 2016
"""

from typing import Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..superposition import LinearSum
from ..utils import ct2a_madsen
from .base import WakeDeficit


class SelfSimilarityBlockageDeficit(WakeDeficit):
    """Self-similarity blockage deficit model.

    This model calculates the blockage (induction) effect upstream of wind turbines
    based on the self-similarity theory. The model is based on the analytical
    solution for the flow field around an actuator disk.

    Unlike wake deficit models that operate downstream, this blockage model
    reduces wind speed upstream of the rotor due to the pressure field induced
    by the rotor's thrust.

    Reference:
        N. Troldborg, A.R. Meyer Forsting, "A simple model of the wind turbine
        induction zone derived from numerical simulations", Wind Energy, 2016
    """

    def __init__(
        self,
        ss_gamma: float = 1.1,
        ss_lambda: float = 0.587,
        ss_eta: float = 1.32,
        ss_alpha: float = 8.0 / 9.0,
        ss_beta: float | None = None,
        limiter: float = 1e-10,
        ct2a: Callable = ct2a_madsen,
        **kwargs: Any,
    ) -> None:
        """Initialize the SelfSimilarityBlockageDeficit model.

        Args:
            ss_gamma: Scaling factor for thrust coefficient. Default is 1.1.
            ss_lambda: Self-similarity parameter lambda. Default is 0.587.
            ss_eta: Self-similarity parameter eta. Default is 1.32.
            ss_alpha: Radial decay exponent. Default is 8/9.
            ss_beta: Radial decay scaling. Default is sqrt(2).
            limiter: Small value to avoid singularity at rotor plane.
            ct2a: Callable to convert thrust coefficient to induction factor.
            **kwargs: Additional arguments passed to parent WakeDeficit.
        """
        # Force use_radius_mask to False since we handle masking ourselves
        kwargs["use_radius_mask"] = False
        # Use LinearSum by default for blockage models (preserves sign for speedup effects)
        kwargs.setdefault("superposition", LinearSum())
        super().__init__(**kwargs)

        self.ss_gamma = ss_gamma
        self.ss_lambda = ss_lambda
        self.ss_eta = ss_eta
        self.ss_alpha = ss_alpha
        self.ss_beta = float(jnp.sqrt(2.0)) if ss_beta is None else float(ss_beta)
        self.limiter = limiter
        self.ct2a = ct2a

    def r12(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        """Compute half radius of self-similar profile.

        Eq. (13) from [1]: r12 = sqrt(lambda * (eta + x^2))

        Args:
            x_norm: Normalized streamwise location (x/R, negative upstream)

        Returns:
            Half radius of the self-similar profile
        """
        return jnp.sqrt(self.ss_lambda * (self.ss_eta + x_norm**2))

    def f_eps(
        self, x_norm: jnp.ndarray, cw: jnp.ndarray, R: jnp.ndarray | float
    ) -> jnp.ndarray:
        """Radial induction shape function.

        Eq. (6) from [1]: f_eps = (1 / cosh(beta * cw / (R * r12)))^alpha

        Args:
            x_norm: Normalized streamwise location (x/R, negative upstream)
            cw: Crosswind distance in meters
            R: Rotor radius in meters

        Returns:
            Radial shape function (0-1)
        """
        feps = get_float_eps()
        r12 = self.r12(x_norm)

        # cosh argument: beta * (cw / (R * r12))
        cosh_arg = self.ss_beta * jnp.abs(cw) / (R * r12 + feps)

        # Clamp cosh_arg to prevent overflow (cosh grows very fast)
        cosh_arg_clamped = jnp.minimum(cosh_arg, 50.0)

        f_eps_val = (1.0 / jnp.cosh(cosh_arg_clamped)) ** self.ss_alpha

        # Only apply upstream of rotor (x < -limiter in normalized coords)
        return jnp.where(x_norm < -self.limiter, f_eps_val, 0.0)

    def ct2af(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        """Axial induction shape function along centerline.

        Eq. (7) from [1]: ct2af = 1 + x / sqrt(1 + x^2)

        Derived from vortex cylinder theory.

        Args:
            x_norm: Normalized streamwise location (x/R)

        Returns:
            Axial shape function (0-2)
        """
        feps = get_float_eps()
        return 1.0 + x_norm / jnp.sqrt(1.0 + x_norm**2 + feps)

    def ct2a0(self, ct: jnp.ndarray) -> jnp.ndarray:
        """BEM axial induction from thrust coefficient.

        Uses gamma-scaled CT as in Eq. (8) from [1].

        Args:
            ct: Thrust coefficient

        Returns:
            Axial induction factor
        """
        return self.ct2a(self.ss_gamma * ct)

    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Compute blockage deficit.

        Eq. (5) from [1]: deficit = WS * a0 * ct2af * f_eps

        Args:
            ws_eff: Effective wind speed at each turbine (n_sources,)
            ti_eff: Effective turbulence intensity (unused)
            ctx: Simulation context

        Returns:
            Deficit matrix (n_receivers, n_sources) in m/s
        """
        _ = ti_eff  # unused

        # Get thrust coefficients and rotor properties
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        diameter = ctx.turbine.rotor_diameter
        R = diameter / 2.0  # rotor radius

        # Normalize downwind distance by rotor radius
        # Use absolute value and negate: x = -|dw|/R (upstream is negative x)
        x_norm = -jnp.abs(ctx.dw) / R  # (n_receivers, n_sources)

        # Compute axial induction components
        ct2a0 = self.ct2a0(ct)  # (n_sources,)
        ct2af = self.ct2af(x_norm)  # (n_receivers, n_sources)
        f_eps = self.f_eps(x_norm, ctx.cw, R)  # (n_receivers, n_sources)

        # Combined axial induction
        ct2ax = ct2a0[None, :] * ct2af  # (n_receivers, n_sources)

        # Deficit = WS * axial_induction * radial_shape
        deficit = ctx.ws * ct2ax * f_eps  # (n_receivers, n_sources)

        return deficit

    def _wake_radius(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Compute effective 'wake' radius for blockage.

        For blockage models, this represents the lateral extent of the
        induction zone based on the self-similar half-width r12.

        Args:
            ws_eff: Effective wind speed (unused)
            ti_eff: Effective turbulence intensity (unused)
            ctx: Simulation context

        Returns:
            Wake radius array (n_receivers, n_sources)
        """
        _ = (ws_eff, ti_eff)  # unused

        R = ctx.turbine.rotor_diameter / 2.0

        # Compute normalized distance for all receiver-source pairs
        x_norm = -jnp.abs(ctx.dw) / R  # (n_receivers, n_sources)

        # Compute r12 for each position
        r12 = self.r12(x_norm)  # (n_receivers, n_sources)

        # Return R * r12 as approximate lateral extent of induction zone
        return R * r12

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, SimulationContext]:
        """Calculate effective wind speed after blockage/induction effects.

        Override base class to handle the self-similarity induction model:
        - Upstream (dw < 0): positive deficit (blockage, reduces wind speed)
        - Downstream (dw > 0): negative deficit (speedup, increases wind speed)

        This matches PyWake's behavior where the deficit is negated for
        downstream locations to represent the flow acceleration that occurs
        downstream of a rotor before the wake fully develops.

        Args:
            ws_eff: Effective wind speed at each turbine
            ti_eff: Effective turbulence intensity at each turbine
            ctx: Simulation context

        Returns:
            Tuple of (updated effective wind speeds, updated context)
        """
        # Compute wake radius (for context)
        ctx.wake_radius = self._wake_radius(ws_eff, ti_eff, ctx)

        # Compute deficit (positive = reduction in wind speed)
        ws_deficit_m = self._deficit(ws_eff, ti_eff, ctx)

        # Apply sign based on location:
        # - Upstream (dw < 0): positive deficit (blockage, reduces wind speed)
        # - Downstream (dw > 0): negative deficit (speedup, increases wind speed)
        signed_deficit = jnp.where(ctx.dw < 0.0, ws_deficit_m, -ws_deficit_m)

        # Apply superposition
        new_eff_ws = self.superposition(ctx.ws, signed_deficit)

        return jnp.maximum(0.0, new_eff_ws), ctx


class SelfSimilarityBlockageDeficit2020(SelfSimilarityBlockageDeficit):
    """Updated self-similarity blockage model (2020 version).

    This is an updated version with:
    1. Linear r12 approximation instead of the original quadratic, which
       ensures the induction half-width continues to diminish approaching
       the rotor, avoiding unphysically large lateral induction tails.
    2. Variable gamma(CT, x) that better matches CFD results across a wider
       range of thrust coefficients.

    Reference:
        N. Troldborg, A.R. Meyer Forsting, Wind Energy, 2016 (with 2020 updates)
    """

    def __init__(
        self,
        ss_alpha: float = 8.0 / 9.0,
        ss_beta: float | None = None,
        r12p: tuple[float, float] = (-0.672, 0.4897),
        ngp: tuple[float, float, float, float] = (-1.381, 2.627, -1.524, 1.336),
        fgp: tuple[float, float, float, float] = (-0.06489, 0.4911, 1.116, -0.1577),
        limiter: float = 1e-10,
        ct2a: Callable = ct2a_madsen,
        **kwargs: Any,
    ) -> None:
        """Initialize the updated SelfSimilarityBlockageDeficit2020 model.

        Args:
            ss_alpha: Radial decay exponent. Default is 8/9.
            ss_beta: Radial decay scaling. Default is sqrt(2).
            r12p: Coefficients for linear r12 approximation (slope, intercept).
            ngp: Coefficients for near-field gamma polynomial.
            fgp: Coefficients for far-field gamma sinusoidal fit.
            limiter: Small value to avoid singularity at rotor plane.
            ct2a: Callable to convert thrust coefficient to induction factor.
            **kwargs: Additional arguments passed to parent.
        """
        # Don't call parent __init__ with the original parameters
        # Just set up the base WakeDeficit
        kwargs["use_radius_mask"] = False
        # Use LinearSum by default for blockage models (preserves sign for speedup effects)
        kwargs.setdefault("superposition", LinearSum())
        WakeDeficit.__init__(self, **kwargs)

        self.ss_alpha = ss_alpha
        self.ss_beta = float(jnp.sqrt(2.0)) if ss_beta is None else float(ss_beta)
        self.r12p = r12p
        self.ngp = ngp
        self.fgp = fgp
        self.limiter = limiter
        self.ct2a = ct2a

    def r12(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        """Compute half radius using linear approximation.

        Linear replacement of Eq. (13): r12 = r12p[0] * x + r12p[1]

        Args:
            x_norm: Normalized streamwise location (x/R, negative upstream)

        Returns:
            Half radius of the self-similar profile
        """
        return self.r12p[0] * x_norm + self.r12p[1]

    def near_gamma(self, ct: jnp.ndarray) -> jnp.ndarray:
        """Compute gamma at x/R = -1 (near field).

        Cubic polynomial fit: gamma = ngp[0]*ct^3 + ngp[1]*ct^2 + ngp[2]*ct + ngp[3]

        Args:
            ct: Thrust coefficient

        Returns:
            Near-field gamma value
        """
        # Use Horner's method for efficiency
        return ((self.ngp[0] * ct + self.ngp[1]) * ct + self.ngp[2]) * ct + self.ngp[3]

    def far_gamma(self, ct: jnp.ndarray) -> jnp.ndarray:
        """Compute gamma at x/R = -6 (far field).

        Sinusoidal fit: gamma = fgp[0] * sin((ct - fgp[1]) / fgp[3]) + fgp[2]

        Args:
            ct: Thrust coefficient

        Returns:
            Far-field gamma value
        """
        return self.fgp[0] * jnp.sin((ct - self.fgp[1]) / self.fgp[3]) + self.fgp[2]

    def inter_gamma_fac(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        """Compute interpolation factor between near and far field gamma.

        Args:
            x_norm: Normalized streamwise location (x/R)

        Returns:
            Interpolation factor (0 at x=-1, 1 at x=-6)
        """
        # ct2af values at reference locations
        ct2af_x = self.ct2af(x_norm)
        ct2af_near = self.ct2af(jnp.asarray(-1.0))
        ct2af_far = self.ct2af(jnp.asarray(-6.0))

        feps = get_float_eps()
        finter = jnp.abs(ct2af_x - ct2af_near) / (
            jnp.abs(ct2af_far - ct2af_near) + feps
        )

        # Clamp: 1 for x < -6, 0 for x > -1
        finter = jnp.where(x_norm < -6.0, 1.0, finter)
        finter = jnp.where(x_norm > -1.0, 0.0, finter)

        return finter

    def gamma(self, x_norm: jnp.ndarray, ct: jnp.ndarray) -> jnp.ndarray:
        """Compute position and CT-dependent gamma.

        Interpolates between near-field and far-field gamma values.

        Args:
            x_norm: Normalized streamwise location (x/R)
            ct: Thrust coefficient (n_sources,)

        Returns:
            Gamma values (n_receivers, n_sources)
        """
        ng = self.near_gamma(ct)  # (n_sources,)
        fg = self.far_gamma(ct)  # (n_sources,)
        finter = self.inter_gamma_fac(x_norm)  # (n_receivers, n_sources)

        # Interpolate: gamma = finter * far + (1 - finter) * near
        gamma = finter * fg[None, :] + (1.0 - finter) * ng[None, :]

        return gamma

    def ct2a0_2020(self, x_norm: jnp.ndarray, ct: jnp.ndarray) -> jnp.ndarray:
        """BEM axial induction with position-dependent gamma.

        Uses variable gamma(x, CT) for the 2020 model.

        Args:
            x_norm: Normalized streamwise location
            ct: Thrust coefficient

        Returns:
            Axial induction factor (n_receivers, n_sources)
        """
        gamma_ct = self.gamma(x_norm, ct) * ct[None, :]  # effective CT
        return self.ct2a(gamma_ct)

    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Compute blockage deficit with 2020 updates.

        Args:
            ws_eff: Effective wind speed at each turbine
            ti_eff: Effective turbulence intensity (unused)
            ctx: Simulation context

        Returns:
            Deficit matrix (n_receivers, n_sources) in m/s
        """
        _ = ti_eff  # unused

        # Get thrust coefficients and rotor properties
        ct = ctx.turbine.ct(ws_eff)  # (n_sources,)
        diameter = ctx.turbine.rotor_diameter
        R = diameter / 2.0

        # Normalize downwind distance by rotor radius
        x_norm = -jnp.abs(ctx.dw) / R  # (n_receivers, n_sources)

        # Use the 2020 version with variable gamma
        ct2a0 = self.ct2a0_2020(x_norm, ct)  # (n_receivers, n_sources)
        ct2af = self.ct2af(x_norm)  # (n_receivers, n_sources)
        f_eps = self.f_eps(x_norm, ctx.cw, R)  # (n_receivers, n_sources)

        # Combined axial induction
        ct2ax = ct2a0 * ct2af  # (n_receivers, n_sources)

        # Deficit
        deficit = ctx.ws * ct2ax * f_eps

        return deficit
