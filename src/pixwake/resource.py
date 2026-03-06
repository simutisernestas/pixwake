"""Heterogeneous wind resource models for pixwake.

This module provides classes for representing spatially varying wind resources,
where wind speed and turbulence intensity can vary across the wind farm domain.
Wind direction remains uniform per case (a single value drives the coordinate
rotation for all turbines).

Two representations are supported:
    - GridWindResource: Regular 2D grid of resource data with bilinear interpolation.
    - ScatteredWindResource: Arbitrary scatter points with barycentric interpolation.

Both classes support:
    - Multiple wind cases (time series) with the same spatial point distribution.
    - JAX-differentiable interpolation (GridWindResource fully; ScatteredWindResource
      differentiable w.r.t. resource values but not w.r.t. turbine query positions).
    - Out-of-domain detection with a small floating-point tolerance.

Example::

    import jax.numpy as jnp
    from pixwake import WakeSimulation
    from pixwake.resource import GridWindResource
    from pixwake.deficit import NOJDeficit

    xs = jnp.linspace(0, 2000, 5)
    ys = jnp.linspace(0, 2000, 5)
    ws = jnp.ones((3, 5, 5)) * jnp.array([8.0, 10.0, 12.0])[:, None, None]
    wd = jnp.array([270.0, 270.0, 270.0])
    resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

    sim = WakeSimulation(turbine, NOJDeficit())
    result = sim(wt_xs, wt_ys, wind_resource=resource)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

# Relative tolerance factor used for domain boundary checks.
# A turbine at `x` is considered inside [x0, x1] if
#   x >= x0 - tol * (x1 - x0)  and  x <= x1 + tol * (x1 - x0)
_DOMAIN_TOL_FACTOR: float = 1e-6


class WindResource(ABC):
    """Abstract base class for heterogeneous wind resources.

    A wind resource holds spatially distributed wind conditions (ws, optionally ti)
    together with a uniform wind direction per case.  Subclasses implement
    ``interpolate`` which maps turbine (x, y) positions to per-turbine per-case
    arrays.

    Attributes:
        wd: Wind direction in degrees, shape ``(n_cases,)``.
        ws: Wind speed data with a spatial dimension, shape depends on subclass.
        ti: Optional turbulence intensity data with the same spatial shape as ws.
    """

    @abstractmethod
    def interpolate(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Interpolate resource values at turbine positions.

        Args:
            wt_x: Turbine x-coordinates, shape ``(n_turbines,)``.
            wt_y: Turbine y-coordinates, shape ``(n_turbines,)``.

        Returns:
            A tuple of:
                - ``ws_at_turbines``: shape ``(n_cases, n_turbines)``
                - ``wd_per_case``: shape ``(n_cases,)``
                - ``ti_at_turbines``: shape ``(n_cases, n_turbines)`` or ``None``

        Raises:
            ValueError: If any turbine position lies outside the resource domain
                (beyond the tolerance ``_DOMAIN_TOL_FACTOR``).
        """
        ...


class GridWindResource(WindResource):
    """Heterogeneous wind resource on a regular 2D grid.

    The wind speed (and optional turbulence intensity) are defined on a regular
    rectangular grid ``(xs, ys)``.  Each wind case has its own 2D field, but all
    cases share the same grid coordinates.

    Interpolation uses bilinear (linear in x, linear in y) interpolation
    implemented entirely in JAX, making it fully differentiable with respect to
    both the resource values and the turbine query positions.

    Args:
        xs: 1D grid x-coordinates, shape ``(Nx,)``, must be strictly increasing.
        ys: 1D grid y-coordinates, shape ``(Ny,)``, must be strictly increasing.
        wd: Wind direction per case in degrees, shape ``(n_cases,)``.
        ws: Wind speed grid, shape ``(n_cases, Nx, Ny)``.
        ti: Optional turbulence intensity grid, shape ``(n_cases, Nx, Ny)``.

    Raises:
        ValueError: On construction if array shapes are inconsistent.

    Example::

        xs = jnp.linspace(0, 2000, 11)   # 11 points in x
        ys = jnp.linspace(0, 1000, 6)    # 6 points in y
        wd = jnp.array([270.0, 280.0])   # 2 cases
        ws = jnp.ones((2, 11, 6)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)
    """

    def __init__(
        self,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        wd: jnp.ndarray,
        ws: jnp.ndarray,
        ti: jnp.ndarray | None = None,
    ) -> None:
        self.xs = jnp.asarray(xs)
        self.ys = jnp.asarray(ys)
        self.wd = jnp.asarray(wd)
        self.ws = jnp.asarray(ws)
        self.ti = jnp.asarray(ti) if ti is not None else None

        self._validate()

    def _validate(self) -> None:
        """Validate shapes and monotonicity."""
        if self.xs.ndim != 1:
            raise ValueError(f"xs must be 1D, got shape {self.xs.shape}")
        if self.ys.ndim != 1:
            raise ValueError(f"ys must be 1D, got shape {self.ys.shape}")
        if self.wd.ndim != 1:
            raise ValueError(f"wd must be 1D, got shape {self.wd.shape}")

        n_cases = self.wd.shape[0]
        nx = self.xs.shape[0]
        ny = self.ys.shape[0]

        if self.ws.shape != (n_cases, nx, ny):
            raise ValueError(
                f"ws must have shape (n_cases, Nx, Ny) = ({n_cases}, {nx}, {ny}), "
                f"got {self.ws.shape}"
            )
        if self.ti is not None and self.ti.shape != (n_cases, nx, ny):
            raise ValueError(
                f"ti must have shape (n_cases, Nx, Ny) = ({n_cases}, {nx}, {ny}), "
                f"got {self.ti.shape}"
            )

    def _check_domain(self, wt_x: jnp.ndarray, wt_y: jnp.ndarray) -> None:
        """Raise ValueError if any turbine lies outside the grid domain."""
        x0, x1 = float(self.xs[0]), float(self.xs[-1])
        y0, y1 = float(self.ys[0]), float(self.ys[-1])
        tol_x = _DOMAIN_TOL_FACTOR * (x1 - x0)
        tol_y = _DOMAIN_TOL_FACTOR * (y1 - y0)

        wt_x_np = jnp.asarray(wt_x)
        wt_y_np = jnp.asarray(wt_y)

        out_x = (wt_x_np < x0 - tol_x) | (wt_x_np > x1 + tol_x)
        out_y = (wt_y_np < y0 - tol_y) | (wt_y_np > y1 + tol_y)

        if jnp.any(out_x):
            bad = wt_x_np[out_x]
            raise ValueError(
                f"Turbine x-coordinates {bad.tolist()} are outside the grid domain "
                f"[{x0}, {x1}] (tolerance {tol_x:.6g})."
            )
        if jnp.any(out_y):
            bad = wt_y_np[out_y]
            raise ValueError(
                f"Turbine y-coordinates {bad.tolist()} are outside the grid domain "
                f"[{y0}, {y1}] (tolerance {tol_y:.6g})."
            )

    def _bilinear_interp_field(
        self,
        field: jnp.ndarray,
        px: jnp.ndarray,
        py: jnp.ndarray,
    ) -> jnp.ndarray:
        """Bilinear interpolation of a single 2D field at query points.

        The interpolation is fully differentiable w.r.t. both ``field`` and
        the query positions ``(px, py)``.

        Args:
            field: 2D array of shape ``(Nx, Ny)``.
            px: Query x-positions, shape ``(n_turbines,)``.
            py: Query y-positions, shape ``(n_turbines,)``.

        Returns:
            Interpolated values, shape ``(n_turbines,)``.
        """
        xs = self.xs
        ys = self.ys
        nx = xs.shape[0]
        ny = ys.shape[0]

        # Clamp query positions to valid range (already validated above)
        px_c = jnp.clip(px, xs[0], xs[-1])
        py_c = jnp.clip(py, ys[0], ys[-1])

        # Find lower grid index for each query point
        # searchsorted gives insertion point; subtract 1 to get floor index
        ix = jnp.searchsorted(xs, px_c, side="right") - 1
        iy = jnp.searchsorted(ys, py_c, side="right") - 1

        # Clamp indices to [0, N-2] so ix+1 is always valid
        ix = jnp.clip(ix, 0, nx - 2)
        iy = jnp.clip(iy, 0, ny - 2)

        # Fractional position within the cell
        dx = xs[ix + 1] - xs[ix]
        dy = ys[iy + 1] - ys[iy]
        tx = (px_c - xs[ix]) / dx  # [0, 1]
        ty = (py_c - ys[iy]) / dy  # [0, 1]

        # Gather the four corner values
        f00 = field[ix, iy]
        f10 = field[ix + 1, iy]
        f01 = field[ix, iy + 1]
        f11 = field[ix + 1, iy + 1]

        # Bilinear combination
        return (
            f00 * (1 - tx) * (1 - ty)
            + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty
            + f11 * tx * ty
        )

    def interpolate(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Bilinearly interpolate wind resource at turbine positions.

        Args:
            wt_x: Turbine x-coordinates, shape ``(n_turbines,)``.
            wt_y: Turbine y-coordinates, shape ``(n_turbines,)``.

        Returns:
            - ``ws_at_turbines``: shape ``(n_cases, n_turbines)``
            - ``wd_per_case``: shape ``(n_cases,)``
            - ``ti_at_turbines``: shape ``(n_cases, n_turbines)`` or ``None``

        Raises:
            ValueError: If any turbine is outside the grid domain.
        """
        wt_x = jnp.asarray(wt_x)
        wt_y = jnp.asarray(wt_y)

        # Domain check (eager — outside JAX trace)
        self._check_domain(wt_x, wt_y)

        # Vectorise over cases: for each case, interpolate over all turbines
        def _interp_case(field_2d: jnp.ndarray) -> jnp.ndarray:
            return self._bilinear_interp_field(field_2d, wt_x, wt_y)

        # ws has shape (n_cases, Nx, Ny) → map over axis 0
        ws_at_turbines = jax.vmap(_interp_case)(self.ws)  # (n_cases, n_turbines)

        ti_at_turbines: jnp.ndarray | None = None
        if self.ti is not None:
            ti_at_turbines = jax.vmap(_interp_case)(self.ti)  # (n_cases, n_turbines)

        return ws_at_turbines, self.wd, ti_at_turbines


class ScatteredWindResource(WindResource):
    """Heterogeneous wind resource at arbitrary scatter points.

    Wind speed (and optional turbulence intensity) are defined at an arbitrary
    set of 2D points.  Interpolation uses barycentric coordinates within the
    Delaunay triangulation of the scatter points.

    The Delaunay triangulation is computed once at construction time using
    ``scipy.spatial.Delaunay`` and stored as static JAX arrays.  The
    barycentric weight computation is differentiable w.r.t. the resource
    *values*, but the triangle-lookup step is **not** differentiable w.r.t.
    the turbine query *positions*.  For fully position-differentiable
    interpolation (e.g. layout optimisation), prefer :class:`GridWindResource`.

    Args:
        points: Scatter point coordinates, shape ``(N, 2)``, columns are (x, y).
        wd: Wind direction per case in degrees, shape ``(n_cases,)``.
        ws: Wind speed at scatter points, shape ``(n_cases, N)``.
        ti: Optional turbulence intensity at scatter points, shape ``(n_cases, N)``.

    Raises:
        ValueError: On construction if array shapes are inconsistent.
        ImportError: If ``scipy`` is not installed.

    Example::

        import numpy as np
        import jax.numpy as jnp
        from pixwake.resource import ScatteredWindResource

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 2000, (30, 2))
        wd = jnp.array([270.0, 280.0])
        ws = jnp.ones((2, 30)) * 10.0
        resource = ScatteredWindResource(points=pts, wd=wd, ws=ws)
    """

    def __init__(
        self,
        points: jnp.ndarray,
        wd: jnp.ndarray,
        ws: jnp.ndarray,
        ti: jnp.ndarray | None = None,
    ) -> None:
        try:
            from scipy.spatial import Delaunay  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "scipy is required for ScatteredWindResource. "
                "Install it with: pip install scipy"
            ) from exc

        self.points = jnp.asarray(points)
        self.wd = jnp.asarray(wd)
        self.ws = jnp.asarray(ws)
        self.ti = jnp.asarray(ti) if ti is not None else None

        self._validate()

        # Build Delaunay triangulation (scipy, outside JAX trace)
        import numpy as np

        pts_np = np.asarray(self.points)
        self._tri = Delaunay(pts_np)

        # Pre-convert triangulation data to JAX arrays for fast lookup
        # simplices: (n_triangles, 3)  vertex indices
        self._simplices = jnp.array(self._tri.simplices)  # (n_tri, 3)
        # Transform matrices for barycentric coords: (n_tri, 2, 2)
        # For triangle with vertices A, B, C:
        #   T = [[Bx-Ax, Cx-Ax], [By-Ay, Cy-Ay]]
        #   bary([px, py]) = T^-1 @ ([px, py] - [Ax, Ay])
        pts = pts_np[self._tri.simplices]  # (n_tri, 3, 2)
        T = pts[:, 1:, :] - pts[:, :1, :]  # (n_tri, 2, 2) — rows are (B-A, C-A)
        self._T = jnp.array(T.transpose(0, 2, 1))  # (n_tri, 2, 2) in column-major
        self._T_inv = jnp.linalg.inv(self._T)  # (n_tri, 2, 2)
        self._vert_a = jnp.array(pts[:, 0, :])  # (n_tri, 2) — vertex A per triangle

    def _validate(self) -> None:
        """Validate shapes."""
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2), got {self.points.shape}"
            )
        if self.wd.ndim != 1:
            raise ValueError(f"wd must be 1D, got shape {self.wd.shape}")

        n_cases = self.wd.shape[0]
        n_pts = self.points.shape[0]

        if n_pts < 3:
            raise ValueError(
                f"ScatteredWindResource requires at least 3 points, got {n_pts}"
            )

        if self.ws.shape != (n_cases, n_pts):
            raise ValueError(
                f"ws must have shape (n_cases, N) = ({n_cases}, {n_pts}), "
                f"got {self.ws.shape}"
            )
        if self.ti is not None and self.ti.shape != (n_cases, n_pts):
            raise ValueError(
                f"ti must have shape (n_cases, N) = ({n_cases}, {n_pts}), "
                f"got {self.ti.shape}"
            )

    def _check_domain(self, wt_x: jnp.ndarray, wt_y: jnp.ndarray) -> None:
        """Raise ValueError if any turbine lies outside the convex hull."""
        import numpy as np

        query = np.column_stack([np.asarray(wt_x), np.asarray(wt_y)])
        simplex_ids = self._tri.find_simplex(query)
        outside = simplex_ids == -1

        if outside.any():
            bad_x = np.asarray(wt_x)[outside]
            bad_y = np.asarray(wt_y)[outside]
            bad_pts = list(zip(bad_x.tolist(), bad_y.tolist()))
            raise ValueError(
                f"Turbine positions {bad_pts} are outside the convex hull of the "
                "scatter points. Extrapolation is not allowed."
            )

    def _find_containing_triangle(
        self, px: jnp.ndarray, py: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return barycentric weights and triangle vertex indices for query points.

        This step uses scipy's ``find_simplex`` and is *not* differentiable
        w.r.t. query positions; it runs outside the JAX trace.

        Args:
            px: Query x-coordinates, shape ``(n_turbines,)``.
            py: Query y-coordinates, shape ``(n_turbines,)``.

        Returns:
            - ``simplex_idx``: Triangle index per turbine, shape ``(n_turbines,)``.
            - ``bary``: Barycentric weights ``(w_a, w_b, w_c)``,
              shape ``(n_turbines, 3)``.
        """
        import numpy as np

        query = np.column_stack([np.asarray(px), np.asarray(py)])
        simplex_idx = self._tri.find_simplex(query)  # (n_turbines,)

        # Compute barycentric weights using pre-inverted transform matrices
        # For query point p and triangle i with vertex A:
        #   [b1, b2] = T_inv[i] @ (p - A[i])
        #   w_b = b1, w_c = b2, w_a = 1 - b1 - b2
        T_inv_sel = self._T_inv[simplex_idx]  # (n_turbines, 2, 2)
        vert_a_sel = self._vert_a[simplex_idx]  # (n_turbines, 2)

        query_jax = jnp.array(query)  # (n_turbines, 2)
        diff = query_jax - vert_a_sel  # (n_turbines, 2)

        # Matrix-vector product: (n_turbines, 2, 2) @ (n_turbines, 2, 1)
        bc = jnp.einsum("tij,tj->ti", T_inv_sel, diff)  # (n_turbines, 2)
        w_b = bc[:, 0]
        w_c = bc[:, 1]
        w_a = 1.0 - w_b - w_c

        bary = jnp.stack([w_a, w_b, w_c], axis=1)  # (n_turbines, 3)
        return jnp.array(simplex_idx), bary

    def _interp_at_turbines(
        self,
        field: jnp.ndarray,
        simplex_idx: jnp.ndarray,
        bary: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate a scatter field at pre-computed triangle positions.

        Args:
            field: Values at scatter points, shape ``(N,)`` for a single case.
            simplex_idx: Triangle index per turbine, shape ``(n_turbines,)``.
            bary: Barycentric weights, shape ``(n_turbines, 3)``.

        Returns:
            Interpolated values, shape ``(n_turbines,)``.
        """
        # Vertex indices of containing triangle per turbine: (n_turbines, 3)
        vert_idx = self._simplices[simplex_idx]  # (n_turbines, 3)
        # Values at each vertex: (n_turbines, 3)
        vert_vals = field[vert_idx]
        # Weighted sum
        return jnp.sum(bary * vert_vals, axis=1)

    def interpolate(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Barycentrically interpolate wind resource at turbine positions.

        Args:
            wt_x: Turbine x-coordinates, shape ``(n_turbines,)``.
            wt_y: Turbine y-coordinates, shape ``(n_turbines,)``.

        Returns:
            - ``ws_at_turbines``: shape ``(n_cases, n_turbines)``
            - ``wd_per_case``: shape ``(n_cases,)``
            - ``ti_at_turbines``: shape ``(n_cases, n_turbines)`` or ``None``

        Raises:
            ValueError: If any turbine is outside the convex hull.
        """
        wt_x = jnp.asarray(wt_x)
        wt_y = jnp.asarray(wt_y)

        # Domain check (eager — outside JAX trace)
        self._check_domain(wt_x, wt_y)

        # Triangle lookup (not differentiable w.r.t. query positions)
        simplex_idx, bary = self._find_containing_triangle(wt_x, wt_y)

        # Interpolate each case using barycentric weights
        def _interp_case(field_1d: jnp.ndarray) -> jnp.ndarray:
            return self._interp_at_turbines(field_1d, simplex_idx, bary)

        ws_at_turbines = jax.vmap(_interp_case)(self.ws)  # (n_cases, n_turbines)

        ti_at_turbines: jnp.ndarray | None = None
        if self.ti is not None:
            ti_at_turbines = jax.vmap(_interp_case)(self.ti)  # (n_cases, n_turbines)

        return ws_at_turbines, self.wd, ti_at_turbines
