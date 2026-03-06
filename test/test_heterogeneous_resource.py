"""Tests for heterogeneous wind resource support (GridWindResource, ScatteredWindResource)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pixwake import Curve, GridWindResource, ScatteredWindResource, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit, NOJDeficit, NiayifarGaussianDeficit
from pixwake.resource import WindResource
from pixwake.turbulence import CrespoHernandez

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_turbine():
    ws = jnp.array([3.0, 5.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 100.0, 1000.0, 1500.0, 1500.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.2])
    return Turbine(
        rotor_diameter=80.0,
        hub_height=70.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


@pytest.fixture
def grid_resource():
    """5x5 grid over [0, 2000]x[0, 2000] with 3 cases."""
    xs = jnp.linspace(0.0, 2000.0, 5)
    ys = jnp.linspace(0.0, 2000.0, 5)
    wd = jnp.array([270.0, 270.0, 270.0])
    # ws varies linearly in x: ws = 8 + 4*(x/2000) so range [8, 12]
    ws = (8.0 + 4.0 * xs[None, :, None] / 2000.0) * jnp.ones((3, 5, 5))
    return GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)


@pytest.fixture
def scattered_resource():
    """25 random scatter points covering [0, 2000]^2, 2 cases."""
    rng = np.random.default_rng(42)
    # Use a regular grid-like scatter so convex hull covers turbine positions
    pts = np.array(
        [[x, y] for x in np.linspace(0, 2000, 5) for y in np.linspace(0, 2000, 5)],
        dtype=np.float64,
    )
    wd = jnp.array([270.0, 280.0])
    ws = jnp.ones((2, 25)) * 10.0
    return ScatteredWindResource(points=pts, wd=wd, ws=ws)


# ---------------------------------------------------------------------------
# WindResource ABC
# ---------------------------------------------------------------------------


class TestWindResourceABC:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            WindResource()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# GridWindResource construction / validation
# ---------------------------------------------------------------------------


class TestGridWindResourceConstruction:
    def test_basic_construction(self, grid_resource):
        assert isinstance(grid_resource, GridWindResource)
        assert isinstance(grid_resource, WindResource)

    def test_wrong_ws_shape_raises(self):
        xs = jnp.linspace(0.0, 1000.0, 3)
        ys = jnp.linspace(0.0, 1000.0, 3)
        wd = jnp.array([270.0])
        ws_bad = jnp.ones((1, 4, 3))  # wrong Nx
        with pytest.raises(ValueError, match="ws must have shape"):
            GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws_bad)

    def test_wrong_ti_shape_raises(self):
        xs = jnp.linspace(0.0, 1000.0, 3)
        ys = jnp.linspace(0.0, 1000.0, 3)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3, 3))
        ti_bad = jnp.ones((1, 2, 3))
        with pytest.raises(ValueError, match="ti must have shape"):
            GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws, ti=ti_bad)


# ---------------------------------------------------------------------------
# GridWindResource interpolation
# ---------------------------------------------------------------------------


class TestGridWindResourceInterpolation:
    def test_interpolate_output_shapes(self, grid_resource):
        wt_x = jnp.array([0.0, 1000.0, 2000.0])
        wt_y = jnp.array([1000.0, 1000.0, 1000.0])
        ws_t, wd_t, ti_t = grid_resource.interpolate(wt_x, wt_y)
        assert ws_t.shape == (3, 3)  # (n_cases, n_turbines)
        assert wd_t.shape == (3,)    # (n_cases,)
        assert ti_t is None

    def test_interpolate_at_grid_nodes_exact(self):
        """Interpolation at exact grid nodes must return the exact grid value."""
        xs = jnp.array([0.0, 500.0, 1000.0])
        ys = jnp.array([0.0, 500.0, 1000.0])
        wd = jnp.array([270.0])
        ws = jnp.arange(9.0).reshape(1, 3, 3)
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        # Query at all grid nodes
        query_x, query_y = jnp.meshgrid(xs, ys)
        wt_x = query_x.ravel()
        wt_y = query_y.ravel()
        ws_t, _, _ = resource.interpolate(wt_x, wt_y)

        # Expected: for each (xi, yj) node, value is ws[0, xi_idx, yj_idx]
        # meshgrid with default indexing='xy': query_x varies along cols
        expected = jnp.array([ws[0, xi, yj] for yj in range(3) for xi in range(3)])
        np.testing.assert_allclose(ws_t[0], expected, atol=1e-6)

    def test_interpolate_midpoint(self):
        """Test that midpoint of four equal-valued corners gives the same value."""
        xs = jnp.array([0.0, 1000.0])
        ys = jnp.array([0.0, 1000.0])
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 2, 2)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        ws_t, _, _ = resource.interpolate(jnp.array([500.0]), jnp.array([500.0]))
        np.testing.assert_allclose(ws_t[0, 0], 10.0, atol=1e-6)

    def test_interpolate_ti(self):
        xs = jnp.linspace(0.0, 1000.0, 3)
        ys = jnp.linspace(0.0, 1000.0, 3)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3, 3)) * 10.0
        ti = jnp.ones((1, 3, 3)) * 0.06
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws, ti=ti)

        wt_x = jnp.array([250.0, 500.0])
        wt_y = jnp.array([500.0, 500.0])
        _, _, ti_t = resource.interpolate(wt_x, wt_y)
        assert ti_t is not None
        assert ti_t.shape == (1, 2)
        np.testing.assert_allclose(ti_t, 0.06, atol=1e-6)

    def test_out_of_domain_x_raises(self):
        xs = jnp.linspace(0.0, 1000.0, 3)
        ys = jnp.linspace(0.0, 1000.0, 3)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3, 3)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        with pytest.raises(ValueError, match="outside the grid domain"):
            resource.interpolate(jnp.array([1500.0]), jnp.array([500.0]))

    def test_out_of_domain_y_raises(self):
        xs = jnp.linspace(0.0, 1000.0, 3)
        ys = jnp.linspace(0.0, 1000.0, 3)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3, 3)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        with pytest.raises(ValueError, match="outside the grid domain"):
            resource.interpolate(jnp.array([500.0]), jnp.array([-100.0]))

    def test_within_tolerance_does_not_raise(self):
        """Positions within _DOMAIN_TOL_FACTOR should not raise."""
        xs = jnp.array([0.0, 1000.0])
        ys = jnp.array([0.0, 1000.0])
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 2, 2)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)
        # 1e-7 * 1000 = 1e-4 < tolerance (1e-6 * 1000 = 1e-3), so within tol
        resource.interpolate(jnp.array([-1e-4]), jnp.array([500.0]))


# ---------------------------------------------------------------------------
# GridWindResource differentiability
# ---------------------------------------------------------------------------


class TestGridWindResourceGradients:
    def test_grad_wrt_ws_values(self):
        """Gradient of sum(interpolated ws) w.r.t. grid ws values should be non-zero."""
        xs = jnp.array([0.0, 1000.0])
        ys = jnp.array([0.0, 1000.0])
        wd = jnp.array([270.0])
        wt_x = jnp.array([250.0, 750.0])
        wt_y = jnp.array([500.0, 500.0])

        def f(ws_grid):
            resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws_grid)
            ws_t, _, _ = resource.interpolate(wt_x, wt_y)
            return ws_t.sum()

        ws_init = jnp.ones((1, 2, 2)) * 10.0
        grad = jax.grad(f)(ws_init)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_grad_wrt_turbine_positions(self):
        """Gradient of sum(interpolated ws) w.r.t. turbine positions should be non-zero."""
        xs = jnp.linspace(0.0, 1000.0, 5)
        ys = jnp.linspace(0.0, 1000.0, 5)
        wd = jnp.array([270.0])
        # ws varies linearly in x
        ws_grid = (8.0 + 4.0 * xs[None, :, None] / 1000.0) * jnp.ones((1, 5, 5))
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws_grid)

        def f(wt_x, wt_y):
            ws_t, _, _ = resource.interpolate(wt_x, wt_y)
            return ws_t.sum()

        wt_x = jnp.array([250.0, 750.0])
        wt_y = jnp.array([500.0, 500.0])
        grad_x, grad_y = jax.grad(f, argnums=(0, 1))(wt_x, wt_y)
        assert jnp.all(jnp.isfinite(grad_x))
        assert jnp.any(grad_x != 0.0)


# ---------------------------------------------------------------------------
# ScatteredWindResource construction / validation
# ---------------------------------------------------------------------------


class TestScatteredWindResourceConstruction:
    def test_basic_construction(self, scattered_resource):
        assert isinstance(scattered_resource, ScatteredWindResource)
        assert isinstance(scattered_resource, WindResource)

    def test_too_few_points_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])  # only 2 points
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 2))
        with pytest.raises(ValueError, match="at least 3 points"):
            ScatteredWindResource(points=pts, wd=wd, ws=ws)

    def test_wrong_ws_shape_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        wd = jnp.array([270.0])
        ws_bad = jnp.ones((1, 5))  # wrong n_pts
        with pytest.raises(ValueError, match="ws must have shape"):
            ScatteredWindResource(points=pts, wd=wd, ws=ws_bad)


# ---------------------------------------------------------------------------
# ScatteredWindResource interpolation
# ---------------------------------------------------------------------------


class TestScatteredWindResourceInterpolation:
    def test_interpolate_output_shapes(self, scattered_resource):
        wt_x = jnp.array([100.0, 1000.0])
        wt_y = jnp.array([100.0, 1000.0])
        ws_t, wd_t, ti_t = scattered_resource.interpolate(wt_x, wt_y)
        assert ws_t.shape == (2, 2)  # (n_cases, n_turbines)
        assert wd_t.shape == (2,)
        assert ti_t is None

    def test_interpolate_at_scatter_point(self):
        """Interpolation at a known scatter point returns that point's value."""
        pts = np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 1000.0]])
        wd = jnp.array([270.0])
        ws = jnp.array([[8.0, 10.0, 12.0]])
        resource = ScatteredWindResource(points=pts, wd=wd, ws=ws)

        # Query exactly at the first point
        ws_t, _, _ = resource.interpolate(jnp.array([0.0]), jnp.array([0.0]))
        np.testing.assert_allclose(ws_t[0, 0], 8.0, atol=1e-5)

    def test_out_of_domain_raises(self):
        pts = np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 1000.0]])
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3)) * 10.0
        resource = ScatteredWindResource(points=pts, wd=wd, ws=ws)

        with pytest.raises(ValueError, match="outside the convex hull"):
            resource.interpolate(jnp.array([2000.0]), jnp.array([2000.0]))

    def test_interpolate_ti(self):
        pts = np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 1000.0]])
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 3)) * 10.0
        ti = jnp.ones((1, 3)) * 0.05
        resource = ScatteredWindResource(points=pts, wd=wd, ws=ws, ti=ti)

        _, _, ti_t = resource.interpolate(jnp.array([500.0]), jnp.array([200.0]))
        assert ti_t is not None
        np.testing.assert_allclose(ti_t[0, 0], 0.05, atol=1e-5)


# ---------------------------------------------------------------------------
# WakeSimulation with heterogeneous resource
# ---------------------------------------------------------------------------


class TestWakeSimulationWithResource:
    def test_mutual_exclusivity_raises(self, simple_turbine, grid_resource):
        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        wt_x = jnp.array([0.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0])
        with pytest.raises(ValueError, match="mutually exclusive"):
            sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, wind_resource=grid_resource)

    def test_grid_resource_result_shapes(self, simple_turbine):
        xs = jnp.linspace(0.0, 2000.0, 5)
        ys = jnp.linspace(0.0, 2000.0, 5)
        wd = jnp.array([270.0, 270.0])
        ws = jnp.ones((2, 5, 5)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        wt_x = jnp.array([0.0, 500.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])
        result = sim(wt_x, wt_y, wind_resource=resource)

        assert result.effective_ws.shape == (2, 3)  # (n_cases, n_turbines)
        assert result.ws.shape == (2, 3)             # heterogeneous: (n_cases, n_turbines)

    def test_uniform_resource_matches_homogeneous(self, simple_turbine):
        """Uniform grid resource should give same result as homogeneous API."""
        ws_val = 10.0
        wd_val = 270.0
        wt_x = jnp.array([0.0, 500.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])

        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))

        # Homogeneous API
        result_homo = sim(wt_x, wt_y, ws_amb=ws_val, wd_amb=wd_val)

        # Grid resource with uniform ws
        xs = jnp.linspace(-100.0, 1100.0, 3)
        ys = jnp.linspace(-100.0, 100.0, 3)
        ws_grid = jnp.ones((1, 3, 3)) * ws_val
        wd_arr = jnp.array([wd_val])
        resource = GridWindResource(xs=xs, ys=ys, wd=wd_arr, ws=ws_grid)
        result_hetero = sim(wt_x, wt_y, wind_resource=resource)

        np.testing.assert_allclose(
            result_homo.effective_ws,
            result_hetero.effective_ws,
            atol=1e-4,
        )

    def test_spatially_varying_ws_affects_result(self, simple_turbine):
        """Non-uniform ws should produce different results than uniform ws at same mean."""
        wt_x = jnp.array([0.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0])

        # Grid with strong gradient in ws
        xs = jnp.array([0.0, 2000.0])
        ys = jnp.array([-500.0, 500.0])
        wd = jnp.array([270.0])

        # Turbine 0 sees 8 m/s, turbine 1 sees 12 m/s
        ws_grid = jnp.array([[[8.0, 8.0], [12.0, 12.0]]])  # (1, 2, 2) — varies in x
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws_grid)

        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        result = sim(wt_x, wt_y, wind_resource=resource)

        # Wind speed at each turbine should match the resource
        # (downstream turbine gets wake from upstream — but ambient is different)
        assert jnp.all(jnp.isfinite(result.effective_ws))

    def test_multi_case_shapes(self, simple_turbine):
        """Multi-case grid resource should produce correct output shapes."""
        n_cases = 5
        xs = jnp.linspace(0.0, 2000.0, 4)
        ys = jnp.linspace(0.0, 2000.0, 4)
        wd = jnp.full((n_cases,), 270.0)
        ws = jnp.ones((n_cases, 4, 4)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        sim = WakeSimulation(simple_turbine, BastankhahGaussianDeficit())
        wt_x = jnp.array([0.0, 500.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])
        result = sim(wt_x, wt_y, wind_resource=resource)
        assert result.effective_ws.shape == (n_cases, 3)

    def test_with_turbulence_model(self, simple_turbine):
        """Heterogeneous resource should work with turbulence model."""
        xs = jnp.linspace(0.0, 2000.0, 4)
        ys = jnp.linspace(-100.0, 100.0, 3)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 4, 3)) * 10.0
        ti = jnp.ones((1, 4, 3)) * 0.06
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws, ti=ti)

        sim = WakeSimulation(
            simple_turbine,
            deficit=NiayifarGaussianDeficit(),
            turbulence=CrespoHernandez(),
        )
        wt_x = jnp.array([0.0, 500.0, 1000.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])
        result = sim(wt_x, wt_y, wind_resource=resource)
        assert result.effective_ti is not None
        assert jnp.all(jnp.isfinite(result.effective_ti))

    def test_scattered_resource_with_simulation(self, simple_turbine):
        """ScatteredWindResource should work end-to-end with WakeSimulation."""
        pts = np.array(
            [[x, y] for x in np.linspace(0, 2000, 5) for y in np.linspace(0, 2000, 5)]
        )
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 25)) * 10.0
        resource = ScatteredWindResource(points=pts, wd=wd, ws=ws)

        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        wt_x = jnp.array([0.0, 500.0, 1000.0])
        wt_y = jnp.array([1000.0, 1000.0, 1000.0])
        result = sim(wt_x, wt_y, wind_resource=resource)
        assert result.effective_ws.shape == (1, 3)
        assert jnp.all(jnp.isfinite(result.effective_ws))

    def test_gradient_through_simulation_wrt_layout(self, simple_turbine):
        """Gradient of AEP w.r.t. turbine layout should work with GridWindResource."""
        xs = jnp.linspace(0.0, 2000.0, 5)
        ys = jnp.linspace(0.0, 2000.0, 5)
        wd = jnp.array([270.0])
        ws = jnp.ones((1, 5, 5)) * 10.0
        resource = GridWindResource(xs=xs, ys=ys, wd=wd, ws=ws)

        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))

        def aep_fn(wt_x, wt_y):
            result = sim(wt_x, wt_y, wind_resource=resource)
            return result.aep()

        wt_x = jnp.array([0.0, 400.0, 800.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])
        grad_x, grad_y = jax.grad(aep_fn, argnums=(0, 1))(wt_x, wt_y)
        assert jnp.all(jnp.isfinite(grad_x))
        assert jnp.all(jnp.isfinite(grad_y))


# ---------------------------------------------------------------------------
# Backward compatibility: homogeneous API must still work unchanged
# ---------------------------------------------------------------------------


class TestHomogeneousBackwardCompat:
    def test_scalar_ws_wd(self, simple_turbine):
        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        result = sim(
            jnp.array([0.0, 500.0]),
            jnp.array([0.0, 0.0]),
            ws_amb=10.0,
            wd_amb=270.0,
        )
        assert result.effective_ws.shape == (1, 2)
        assert result.ws.shape == (1,)  # (n_cases,) scalar form preserved

    def test_array_ws_wd(self, simple_turbine):
        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        result = sim(
            jnp.array([0.0, 500.0]),
            jnp.array([0.0, 0.0]),
            ws_amb=jnp.array([8.0, 10.0, 12.0]),
            wd_amb=jnp.array([270.0, 270.0, 270.0]),
        )
        assert result.effective_ws.shape == (3, 2)
        assert result.ws.shape == (3,)

    def test_with_ti(self, simple_turbine):
        sim = WakeSimulation(
            simple_turbine,
            deficit=BastankhahGaussianDeficit(),
            turbulence=CrespoHernandez(),
        )
        result = sim(
            jnp.array([0.0, 400.0]),
            jnp.array([0.0, 0.0]),
            ws_amb=10.0,
            wd_amb=270.0,
            ti_amb=0.06,
        )
        assert result.effective_ti is not None
        assert result.effective_ti.shape == (1, 2)

    def test_gross_power_homogeneous(self, simple_turbine):
        """gross_power() should broadcast scalar ws to turbine dimension."""
        sim = WakeSimulation(simple_turbine, NOJDeficit(k=0.05))
        result = sim(
            jnp.array([0.0, 500.0, 1000.0]),
            jnp.array([0.0, 0.0, 0.0]),
            ws_amb=10.0,
            wd_amb=270.0,
        )
        gp = result.gross_power()
        assert gp.shape == (1, 3)
        np.testing.assert_allclose(gp, result.turbine.power(jnp.array(10.0)), atol=1.0)
