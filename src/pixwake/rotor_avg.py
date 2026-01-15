from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .core import SimulationContext
from .jax_utils import get_float_eps, ssqrt

if TYPE_CHECKING:
    from .deficit.gaussian import BastankhahGaussianDeficit, TurboGaussianDeficit

    GaussianDeficit: TypeAlias = "BastankhahGaussianDeficit | TurboGaussianDeficit"


class RotorAvg(ABC):
    """Abstract base class for all rotor average models."""

    @abstractmethod
    def __call__(
        self,
        func: Callable,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jax.Array:  # pragma: no cover
        """Computes the rotor-averaged value of a function."""
        raise NotImplementedError


_CGI_NODES_AND_WEIGHTS = {
    4: lambda pm: (pm * jnp.array([0.5, 0.5, 1 / 4])).T,
    7: lambda pm: jnp.concatenate(
        [
            jnp.array(
                [
                    [0, 0, 1 / 4],
                    [-jnp.sqrt(2 / 3), 0, 1 / 8],
                    [jnp.sqrt(2 / 3), 0, 1 / 8],
                ]
            ),
            pm * jnp.array([jnp.sqrt(1 / 6), jnp.sqrt(1 / 2), 1 / 8]),
        ]
    ).T,
    9: lambda pm: jnp.concatenate(
        [
            jnp.array(
                [
                    [0, 0, 1 / 6],
                    [-1, 0, 1 / 24],
                    [1, 0, 1 / 24],
                    [0, -1, 1 / 24],
                    [0, 1, 1 / 24],
                ]
            ),
            pm * jnp.array([1 / 2, 1 / 2, 1 / 6]),
        ]
    ).T,
    21: lambda _: jnp.concatenate(
        [
            jnp.array([[0, 0, 1 / 9]]),
            jnp.array(
                [
                    [
                        jnp.sqrt((6 - jnp.sqrt(6)) / 10) * jnp.cos(2 * jnp.pi * k / 10),
                        jnp.sqrt((6 - jnp.sqrt(6)) / 10) * jnp.sin(2 * jnp.pi * k / 10),
                        (16 + jnp.sqrt(6)) / 360,
                    ]
                    for k in range(1, 11)
                ]
            ),
            jnp.array(
                [
                    [
                        jnp.sqrt((6 + jnp.sqrt(6)) / 10) * jnp.cos(2 * jnp.pi * k / 10),
                        jnp.sqrt((6 + jnp.sqrt(6)) / 10) * jnp.sin(2 * jnp.pi * k / 10),
                        (16 - jnp.sqrt(6)) / 360,
                    ]
                    for k in range(1, 11)
                ]
            ),
        ]
    ).T,
}


class CGIRotorAvg(RotorAvg):
    """This model computes rotor-averaged quantities by evaluating
    a function at specific, predetermined points (nodes) across the
    rotor disk and combining the results using associated weights.
    The node locations and weights are optimized for circular rotor
    averaging, and the number of integration points (4, 7, 9, or 21)
    allows for a tradeoff between computational accuracy and performance.
    """

    nodes_x: jax.Array
    nodes_y: jax.Array
    weights: jax.Array
    n_points: int

    def __init__(self, n_points: int = 7) -> None:
        """
        Initialize CGI rotor averaging model
        Parameters
        ----------
        n_points : int
            Number of integration points: 4, 7, 9, or 21
            More points = higher accuracy but slower computation
        """
        self.n_points = n_points

        # Get predefined node positions and weights for this configuration
        (
            self.nodes_x,
            self.nodes_y,
            self.weights,
        ) = self._get_cgi_nodes_and_weights(n_points)
        self._cache: dict[int, Callable] = {}

    @staticmethod
    def _get_cgi_nodes_and_weights(
        n: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get CGI integration points and weights for circular rotor
        These are predetermined optimal locations for integrating over a circle.
        Coordinates are normalized to rotor radius (range: -1 to 1)
        Returns
        -------
        nodes_x : array
            X-coordinates of integration points (normalized by radius)
        nodes_y : array
            Y-coordinates of integration points (normalized by radius)
        weights : array
            Integration weights (sum to 1.0)
        """
        pm = jnp.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
        if n not in _CGI_NODES_AND_WEIGHTS:
            raise ValueError(f"Invalid number of points: {n}")
        x, y, w = _CGI_NODES_AND_WEIGHTS[n](pm)
        return (jnp.array(x), jnp.array(y), jnp.array(w))

    def __call__(
        self,
        func: Callable,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jax.Array:
        """Computes the rotor-averaged value of `func`. The function must have
        call signature of: func(ws_eff, ti_eff, ctx). The context will be modified
        to evaluate at each integration point across the rotor disk.

        This method handles all dimensional reshaping internally, so `func`
        doesn't need to know about rotor averaging.

        Args:
            func: The function to be rotor-averaged. It should work with
                standard 2D arrays (n_receivers, n_sources).
            ws_eff: The effective wind speeds at each turbine.
            ti_eff: The effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The rotor-averaged value of the function `func`.
        """
        R_dst = jnp.array(ctx.turbine.rotor_diameter / 2.0)

        # Expand to integration points: (n_receivers, n_sources, n_points)
        dw = ctx.dw[..., jnp.newaxis]
        cw = ctx.cw[..., jnp.newaxis]

        # TODO: is this correct ? Should add test against pywake !
        node_x_offset = self.nodes_x.reshape(1, 1, -1) * R_dst.reshape(1, -1, 1)
        node_y_offset = self.nodes_y.reshape(1, 1, -1) * R_dst.reshape(1, -1, 1)

        hcw_at_nodes = cw + node_x_offset
        dh_at_nodes = 0.0 + node_y_offset  # TODO: 0 should be ctx.dh ???
        dw_at_nodes = jnp.broadcast_to(dw, hcw_at_nodes.shape)
        cw_at_nodes = ssqrt(hcw_at_nodes**2 + dh_at_nodes**2)

        if id(func) not in self._cache:
            # Evaluate func at each integration point by vmapping over last axis
            def eval_single_point(
                dw_single: jax.Array, cw_single: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                ctx_single = SimulationContext(
                    turbine=ctx.turbine,
                    dw=dw_single,
                    cw=cw_single,
                    ws=ctx.ws,
                    ti=ctx.ti,
                    wake_radius=ctx.wake_radius,
                )
                return func(ws_eff, ti_eff, ctx_single)

            # Map over integration points (last dimension)
            self._cache[id(func)] = jax.vmap(
                eval_single_point, in_axes=(2, 2), out_axes=2
            )

        value_at_nodes = self._cache[id(func)](
            dw_at_nodes,
            cw_at_nodes,
        )
        # Weighted average over integration points
        weights_broadcast = self.weights.reshape(1, 1, -1)
        return jnp.sum(value_at_nodes * weights_broadcast, axis=-1)


_OVERLAP_TABLE_FILENAME = "gaussian_overlap_.02_.02_128_512.nc"


def _load_overlap_table() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Try to load the precomputed overlap table.

    First tries to load from the local data directory (bundled with pixwake),
    then falls back to PyWake's location if installed.

    Returns:
        Tuple of (R_sigma, CW_sigma, overlap_table) arrays if successful, None otherwise.
    """
    import os
    from pathlib import Path

    # Try loading from local data directory first
    local_table_path = Path(__file__).parent / "data" / _OVERLAP_TABLE_FILENAME
    if local_table_path.exists():
        try:
            # TODO: need to bake it into dependencies...
            import xarray as xr

            table = xr.load_dataarray(local_table_path, engine="h5netcdf")
            return (
                table.R_sigma.values.astype(np.float64),
                table.CW_sigma.values.astype(np.float64),
                table.values.astype(np.float64),
            )
        except (ImportError, OSError):
            pass

    # Fall back to PyWake's location
    try:
        import xarray as xr
        from py_wake.rotor_avg_models import gaussian_overlap_model

        table_path = os.path.join(
            os.path.dirname(gaussian_overlap_model.__file__),
            _OVERLAP_TABLE_FILENAME,
        )
        table = xr.load_dataarray(table_path, engine="h5netcdf")
        return (
            table.R_sigma.values.astype(np.float64),
            table.CW_sigma.values.astype(np.float64),
            table.values.astype(np.float64),
        )
    except (ImportError, FileNotFoundError, OSError):
        return None


def _make_overlap_lookup_table(
    n_theta: int = 128,
    n_r: int = 512,
    dr: float = 0.02,
    dcw: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the Gaussian overlap lookup table.

    This follows the PyWake implementation which computes the integral of a
    Gaussian wake profile over a circular rotor disk.

    Args:
        n_theta: Number of azimuthal discretization points.
        n_r: Number of radial discretization points.
        dr: Grid spacing for R_sigma.
        dcw: Grid spacing for CW_sigma.

    Returns:
        Tuple of (R_sigma, CW_sigma, overlap_table) arrays.
    """
    # Downstream rotor radius normalized with characteristic wake width, sigma
    R_sigma = np.arange(0, 20 + dr, dr)
    # Crosswind distance normalized with characteristic wake width, sigma
    CW_sigma = np.arange(0, 10 + dcw, dcw)

    theta = np.linspace(0, 2 * np.pi, n_theta)  # Azimuthal discretization
    r_norm = np.linspace(0, 1, n_r)  # Normalized radial discretization [0, 1]

    dtheta = theta[1] - theta[0]
    dr_norm = r_norm[1] - r_norm[0]

    # Build overlap table: for each R_sigma value, compute integral
    overlap_table = np.zeros((len(R_sigma), len(CW_sigma)))

    # R_sigma = 0 case: use point value at center
    overlap_table[0, :] = np.exp(-(CW_sigma**2) / 2)

    # For R_sigma > 0: vectorized computation
    # Process in batches to avoid memory issues
    R_sigma_nonzero = R_sigma[1:]  # Skip R_sigma = 0

    for i, R_val in enumerate(R_sigma_nonzero):
        # r is radial coordinate on downstream rotor: r = r_norm * R_val
        r = r_norm * R_val  # (n_r,)

        # Broadcasting shapes:
        # CW_sigma: (n_cw,) -> (n_cw, 1, 1)
        # theta: (n_theta,) -> (1, n_theta, 1)
        # r: (n_r,) -> (1, 1, n_r)
        cw_3d = CW_sigma[:, np.newaxis, np.newaxis]
        theta_3d = theta[np.newaxis, :, np.newaxis]
        r_3d = r[np.newaxis, np.newaxis, :]

        # Gaussian integrand: exp(-1/2 * (r^2 + cw^2 - 2*r*cw*cos(theta))) * r
        exponent = -0.5 * (r_3d**2 + cw_3d**2 - 2 * r_3d * cw_3d * np.cos(theta_3d))
        integrand = np.exp(exponent) * r_3d  # (n_cw, n_theta, n_r)

        # Integrate over theta using trapezoid rule
        integral_theta = np.trapezoid(integrand, dx=dtheta, axis=1)  # (n_cw, n_r)

        # Integrate over r using trapezoid rule
        # Need to scale dr_norm by R_val to get actual dr
        dr_actual = dr_norm * R_val
        integral_total = np.trapezoid(integral_theta, dx=dr_actual, axis=1)  # (n_cw,)

        # Normalize by rotor area (pi * R^2)
        overlap_table[i + 1, :] = integral_total / (np.pi * R_val**2)

    return R_sigma, CW_sigma, overlap_table


class GaussianOverlapAvgModel(RotorAvg):
    """Gaussian overlap averaging model for rotor-averaged wake deficits.

    This model computes rotor-averaged wake deficits by evaluating the overlap
    integral of a Gaussian wake profile over a circular rotor disk. It uses a
    pre-computed lookup table for efficiency and is specifically designed for
    Gaussian wake deficit models (BastankhahGaussianDeficit and NiayifarGaussianDeficit).

    The model follows the implementation from PyWake, computing the overlap factor
    based on:
    - R_sigma: downstream rotor radius normalized by wake width sigma
    - CW_sigma: crosswind distance normalized by wake width sigma

    Reference:
        Based on PyWake's GaussianOverlapAvgModel implementation.
        https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake
    """

    deficit_model: "GaussianDeficit | None"

    def __init__(
        self,
        *,
        n_theta: int = 128,
        n_r: int = 512,
        dr: float = 0.02,
        dcw: float = 0.02,
    ) -> None:
        """Initialize the GaussianOverlapAvgModel.

        Args:
            n_theta: Number of azimuthal discretization points for lookup table.
            n_r: Number of radial discretization points for lookup table.
            dr: Grid spacing for R_sigma in lookup table.
            dcw: Grid spacing for CW_sigma in lookup table.

        Note:
            The `deficit_model` attribute is set automatically when this model
            is passed to a Gaussian deficit model's `rotor_avg_model` parameter.
        """
        self.deficit_model = None

        # Try to load precomputed table
        precomputed_table = _load_overlap_table()
        if precomputed_table is not None:
            self._r_sigma, self._cw_sigma, self._overlap_table = precomputed_table
        else:
            # Generate our own table (slower)
            warnings.warn(
                "Precomputed Gaussian overlap table not found. "
                "Generating lookup table on-the-fly, will take awhile.. "
                "Consider installing xarray and h5netcdf for faster loading.",
                RuntimeWarning,
            )
            self._r_sigma, self._cw_sigma, self._overlap_table = (
                _make_overlap_lookup_table(n_theta=n_theta, n_r=n_r, dr=dr, dcw=dcw)
            )

        # Pre-interpolate to finer grid for JAX-compatible lookup
        R_sigma_fine = np.arange(0, 20.001, 0.01)
        CW_sigma_fine = np.arange(0, 10.01, 0.01)
        spline = RectBivariateSpline(self._r_sigma, self._cw_sigma, self._overlap_table)
        self._overlap_fine = jnp.array(spline(R_sigma_fine, CW_sigma_fine))
        self._r_sigma_fine = jnp.array(R_sigma_fine)
        self._cw_sigma_fine = jnp.array(CW_sigma_fine)
        self._dr_fine = 0.01
        self._dcw_fine = 0.01

    def _interpolate_overlap(
        self, r_sigma: jax.Array, cw_sigma: jax.Array
    ) -> jax.Array:
        """Bilinear interpolation of overlap factor from lookup table.

        Args:
            r_sigma: Normalized rotor radius (R / sigma).
            cw_sigma: Normalized crosswind distance (|cw| / sigma).

        Returns:
            Interpolated overlap factors.
        """
        # Clamp to table bounds
        r_sigma_clamped = jnp.clip(r_sigma, 0.0, 20.0)
        cw_sigma_clamped = jnp.clip(cw_sigma, 0.0, 10.0)

        # Convert to table indices (fractional)
        r_idx = r_sigma_clamped / self._dr_fine
        cw_idx = cw_sigma_clamped / self._dcw_fine

        # Get integer indices for corners
        r_idx_lo = jnp.floor(r_idx).astype(jnp.int32)
        cw_idx_lo = jnp.floor(cw_idx).astype(jnp.int32)

        # Clamp to valid range for indexing
        max_r_idx = self._overlap_fine.shape[0] - 2
        max_cw_idx = self._overlap_fine.shape[1] - 2
        r_idx_lo = jnp.clip(r_idx_lo, 0, max_r_idx)
        cw_idx_lo = jnp.clip(cw_idx_lo, 0, max_cw_idx)

        r_idx_hi = r_idx_lo + 1
        cw_idx_hi = cw_idx_lo + 1

        # Fractional parts for interpolation weights
        r_frac = r_idx - r_idx_lo
        cw_frac = cw_idx - cw_idx_lo

        # Get corner values
        v00 = self._overlap_fine[r_idx_lo, cw_idx_lo]
        v01 = self._overlap_fine[r_idx_lo, cw_idx_hi]
        v10 = self._overlap_fine[r_idx_hi, cw_idx_lo]
        v11 = self._overlap_fine[r_idx_hi, cw_idx_hi]

        # Bilinear interpolation
        v0 = v00 * (1 - cw_frac) + v01 * cw_frac
        v1 = v10 * (1 - cw_frac) + v11 * cw_frac
        return v0 * (1 - r_frac) + v1 * r_frac

    def __call__(
        self,
        func: Callable,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jax.Array:
        """Computes the rotor-averaged value using Gaussian overlap integration.

        This method evaluates the deficit at the centerline (cw=0) and then
        applies the overlap factor computed from the lookup table.

        Args:
            func: The deficit function to be rotor-averaged. Must be the
                `_deficit` method of a Gaussian deficit model.
            ws_eff: The effective wind speeds at each turbine.
            ti_eff: The effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            The rotor-averaged deficit values.

        Raises:
            ValueError: If deficit_model has not been set (this is set
                automatically when passed to a Gaussian deficit model).
        """
        if self.deficit_model is None:
            raise ValueError(
                "GaussianOverlapAvgModel.deficit_model is not set. "
                "Pass this rotor_avg_model to a Gaussian deficit model's "
                "rotor_avg_model parameter to set it automatically."
            )

        feps = get_float_eps()

        # Evaluate deficit at centerline (cw=0)
        ctx_centerline = SimulationContext(
            turbine=ctx.turbine,
            dw=ctx.dw,
            cw=jnp.zeros_like(ctx.cw),
            ws=ctx.ws,
            ti=ctx.ti,
            wake_radius=ctx.wake_radius,
        )
        centerline_deficit = func(ws_eff, ti_eff, ctx_centerline)

        # Get sigma from the deficit model
        sigma = self.deficit_model.sigma(ws_eff, ti_eff, ctx)

        # Compute normalized coordinates
        # R_sigma = (D_dst / 2) / sigma = downstream rotor radius / sigma
        R_dst = ctx.turbine.rotor_diameter / 2.0
        r_sigma = R_dst / (sigma + feps)

        # CW_sigma = |cw| / sigma = crosswind distance / sigma
        cw_sigma = jnp.abs(ctx.cw) / (sigma + feps)

        # Get overlap factors from lookup table
        overlap_factor = self._interpolate_overlap(r_sigma, cw_sigma)

        # Apply overlap factor to centerline deficit
        return centerline_deficit * overlap_factor
