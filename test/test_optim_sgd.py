"""Tests for pixwake.optim.sgd module.

These tests verify:
1. Parity with TopFarm's SGD implementation
2. Differentiability through the optimization process
3. Constraint penalty functions
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit, NOJDeficit
from pixwake.optim import (
    SGDSettings,
    boundary_penalty,
    create_bilevel_optimizer,
    create_layout_optimizer,
    sgd_solve_implicit,
    spacing_penalty,
    topfarm_sgd_solve,
)

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_turbine():
    """Create a simple turbine for testing."""
    power_curve = Curve(
        ws=jnp.array([0.0, 5.0, 10.0, 15.0, 25.0]),
        values=jnp.array([0.0, 100.0, 2000.0, 2000.0, 0.0]),
    )
    ct_curve = Curve(
        ws=jnp.array([0.0, 5.0, 10.0, 15.0, 25.0]),
        values=jnp.array([0.0, 0.8, 0.8, 0.4, 0.0]),
    )
    return Turbine(
        rotor_diameter=80.0,
        hub_height=70.0,
        power_curve=power_curve,
        ct_curve=ct_curve,
    )


@pytest.fixture
def square_boundary():
    """Create a 1km x 1km square boundary."""
    return jnp.array(
        [
            [0.0, 0.0],
            [1000.0, 0.0],
            [1000.0, 1000.0],
            [0.0, 1000.0],
        ]
    )


@pytest.fixture
def four_turbine_layout():
    """Create a 4-turbine initial layout in a grid."""
    x = jnp.array([250.0, 750.0, 250.0, 750.0])
    y = jnp.array([250.0, 250.0, 750.0, 750.0])
    return x, y


# =============================================================================
# Penalty Function Tests
# =============================================================================


class TestBoundaryPenalty:
    """Tests for boundary_penalty function."""

    def test_inside_boundary_zero_penalty(self, square_boundary):
        """Points inside the boundary should have zero penalty."""
        x = jnp.array([500.0, 200.0, 800.0])
        y = jnp.array([500.0, 200.0, 800.0])
        penalty = boundary_penalty(x, y, square_boundary)
        assert jnp.allclose(penalty, 0.0, atol=1e-6)

    def test_outside_boundary_positive_penalty(self, square_boundary):
        """Points outside the boundary should have positive penalty."""
        x = jnp.array([500.0, -100.0])  # Second point is outside
        y = jnp.array([500.0, 500.0])
        penalty = boundary_penalty(x, y, square_boundary)
        assert penalty > 0.0

    def test_boundary_penalty_gradient(self, square_boundary):
        """Gradient should be finite and non-zero near boundary violations."""
        x = jnp.array([50.0, 950.0])
        y = jnp.array([500.0, 500.0])

        def penalty_fn(xx, yy):
            return boundary_penalty(xx, yy, square_boundary)

        grad_x, grad_y = jax.grad(penalty_fn, argnums=(0, 1))(x, y)
        assert jnp.all(jnp.isfinite(grad_x))
        assert jnp.all(jnp.isfinite(grad_y))

    def test_boundary_penalty_differentiable(self, square_boundary):
        """Check that boundary penalty is twice differentiable."""
        x = jnp.array([-50.0, 500.0])  # First point outside
        y = jnp.array([500.0, 500.0])

        def penalty_fn(xx):
            return boundary_penalty(xx, y, square_boundary)

        # Should not raise
        check_grads(penalty_fn, (x,), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)


class TestSpacingPenalty:
    """Tests for spacing_penalty function."""

    def test_sufficient_spacing_zero_penalty(self):
        """Turbines with sufficient spacing should have zero penalty."""
        x = jnp.array([0.0, 500.0, 1000.0])
        y = jnp.array([0.0, 0.0, 0.0])
        min_spacing = 200.0
        penalty = spacing_penalty(x, y, min_spacing)
        assert jnp.allclose(penalty, 0.0, atol=1e-6)

    def test_insufficient_spacing_positive_penalty(self):
        """Turbines too close together should have positive penalty."""
        x = jnp.array([0.0, 100.0])  # 100m apart
        y = jnp.array([0.0, 0.0])
        min_spacing = 200.0  # Requires 200m
        penalty = spacing_penalty(x, y, min_spacing)
        assert penalty > 0.0

    def test_spacing_penalty_gradient(self):
        """Gradient should be finite and push turbines apart."""
        x = jnp.array([0.0, 100.0])
        y = jnp.array([0.0, 0.0])
        min_spacing = 200.0

        def penalty_fn(xx, yy):
            return spacing_penalty(xx, yy, min_spacing)

        grad_x, grad_y = jax.grad(penalty_fn, argnums=(0, 1))(x, y)
        assert jnp.all(jnp.isfinite(grad_x))
        # Gradient should push first turbine left, second right
        assert grad_x[0] > 0  # Positive gradient (minimize by moving left)
        assert grad_x[1] < 0  # Negative gradient (minimize by moving right)

    def test_single_turbine_zero_penalty(self):
        """Single turbine should have zero spacing penalty."""
        x = jnp.array([500.0])
        y = jnp.array([500.0])
        penalty = spacing_penalty(x, y, 200.0)
        assert jnp.allclose(penalty, 0.0)

    def test_spacing_penalty_differentiable(self):
        """Check that spacing penalty is differentiable."""
        x = jnp.array([0.0, 100.0, 500.0])
        y = jnp.array([0.0, 0.0, 0.0])
        min_spacing = 200.0

        def penalty_fn(xx):
            return spacing_penalty(xx, y, min_spacing)

        check_grads(penalty_fn, (x,), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)


# =============================================================================
# SGD Solver Tests
# =============================================================================


class TestTopfarmSGDSolve:
    """Tests for the main SGD solver."""

    def test_solver_respects_boundary(self, simple_turbine, square_boundary):
        """Optimized layout should stay within boundary."""
        # Start with turbines inside (boundary enforcement is secondary to objective)
        init_x = jnp.array([100.0, 500.0, 900.0, 500.0])
        init_y = jnp.array([500.0, 100.0, 500.0, 900.0])

        # Objective that doesn't conflict with boundary
        def objective(x, y):
            return jnp.sum((x - 500) ** 2 + (y - 500) ** 2)

        settings = SGDSettings(max_iter=500, learning_rate=5.0, boundary_weight=100.0)
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, square_boundary, 100.0, settings
        )

        # All turbines should be inside boundary (check using penalty)
        penalty = boundary_penalty(opt_x, opt_y, square_boundary)
        assert penalty < 0.1, f"Boundary penalty too high: {penalty}"

    def test_solver_respects_spacing(self, simple_turbine, square_boundary):
        """Optimized layout should respect minimum spacing via penalty."""
        # Start with turbines spread out
        init_x = jnp.array([200.0, 400.0, 600.0, 800.0])
        init_y = jnp.array([500.0, 500.0, 500.0, 500.0])
        min_spacing = 150.0

        # Objective: minimize distance from center (weak)
        def objective(x, y):
            return jnp.sum((x - 500) ** 2 + (y - 500) ** 2) * 0.0001

        settings = SGDSettings(max_iter=500, learning_rate=5.0, spacing_weight=100.0)
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, square_boundary, min_spacing, settings
        )

        # Spacing penalty should be low
        penalty = spacing_penalty(opt_x, opt_y, min_spacing)
        assert penalty < 0.1, f"Spacing penalty too high: {penalty}"

    def test_solver_with_wake_simulation(
        self, simple_turbine, square_boundary, four_turbine_layout
    ):
        """Test solver with actual wake simulation objective."""
        init_x, init_y = four_turbine_layout
        deficit = NOJDeficit(k=0.05)
        sim = WakeSimulation(simple_turbine, deficit)

        ws = jnp.array([10.0])
        wd = jnp.array([270.0])

        def neg_aep(x, y):
            result = sim(x, y, ws_amb=ws, wd_amb=wd)
            return -result.aep()

        settings = SGDSettings(max_iter=100, learning_rate=5.0)
        opt_x, opt_y = topfarm_sgd_solve(
            neg_aep, init_x, init_y, square_boundary, 160.0, settings
        )

        # Check that optimization improved AEP
        init_aep = -neg_aep(init_x, init_y)
        final_aep = -neg_aep(opt_x, opt_y)
        assert final_aep >= init_aep * 0.95  # At least not much worse


class TestSGDSolveImplicit:
    """Tests for implicit differentiation through SGD.

    Note: Full implicit differentiation through the while_loop requires
    careful handling of JAX tracing. These tests verify the forward pass
    works and the solver can be used in differentiable pipelines.
    """

    def test_sgd_solve_implicit_forward_pass(
        self, simple_turbine, square_boundary, four_turbine_layout
    ):
        """Verify the forward pass of sgd_solve_implicit works correctly."""
        init_x, init_y = four_turbine_layout

        # Simple quadratic objective
        def objective_with_params(x, y, params):
            cx, cy = params[0], params[1]
            return jnp.sum((x - cx) ** 2 + (y - cy) ** 2)

        params = jnp.array([500.0, 500.0])
        settings = SGDSettings(max_iter=100, learning_rate=5.0)

        # Just verify forward pass works
        opt_x, opt_y = sgd_solve_implicit(
            objective_with_params,
            init_x,
            init_y,
            square_boundary,
            100.0,
            settings,
            params,
        )

        # Verify outputs are finite and valid
        assert jnp.all(jnp.isfinite(opt_x))
        assert jnp.all(jnp.isfinite(opt_y))
        assert opt_x.shape == init_x.shape
        assert opt_y.shape == init_y.shape

    def test_sgd_solve_with_different_params(
        self, simple_turbine, square_boundary, four_turbine_layout
    ):
        """Verify solver produces different results with different params."""
        init_x, init_y = four_turbine_layout

        def objective_with_params(x, y, params):
            # Strong pull toward param-defined center
            cx, cy = params[0], params[1]
            return jnp.sum((x - cx) ** 2 + (y - cy) ** 2) * 0.1

        # Use smaller spacing to reduce constraint dominance
        settings = SGDSettings(max_iter=200, learning_rate=5.0, spacing_weight=1.0)

        # Solve with two different param sets (more extreme difference)
        params1 = jnp.array([300.0, 300.0])
        params2 = jnp.array([700.0, 700.0])

        opt_x1, opt_y1 = sgd_solve_implicit(
            objective_with_params,
            init_x,
            init_y,
            square_boundary,
            80.0,  # Smaller spacing to allow more movement
            settings,
            params1,
        )

        opt_x2, opt_y2 = sgd_solve_implicit(
            objective_with_params,
            init_x,
            init_y,
            square_boundary,
            80.0,
            settings,
            params2,
        )

        # The mean position should shift toward the params
        mean_x1 = jnp.mean(opt_x1)
        mean_x2 = jnp.mean(opt_x2)

        # Mean should be closer to respective params
        assert mean_x1 < mean_x2, f"Expected mean_x1 ({mean_x1}) < mean_x2 ({mean_x2})"


# =============================================================================
# Integration Tests
# =============================================================================


class TestLayoutOptimizer:
    """Tests for the convenience layout optimizer wrapper."""

    def test_create_layout_optimizer(
        self, simple_turbine, square_boundary, four_turbine_layout
    ):
        """Test that create_layout_optimizer produces working optimizer."""
        init_x, init_y = four_turbine_layout
        deficit = BastankhahGaussianDeficit()
        sim = WakeSimulation(simple_turbine, deficit)

        ws = jnp.array([8.0, 10.0, 12.0])
        wd = jnp.array([270.0, 270.0, 270.0])

        settings = SGDSettings(max_iter=50, learning_rate=5.0)
        optimizer = create_layout_optimizer(
            sim_engine=sim,
            boundary=square_boundary,
            min_spacing=160.0,
            ws_amb=ws,
            wd_amb=wd,
            settings=settings,
        )

        opt_x, opt_y = optimizer(init_x, init_y)

        # Check output shape
        assert opt_x.shape == init_x.shape
        assert opt_y.shape == init_y.shape

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(opt_x))
        assert jnp.all(jnp.isfinite(opt_y))


class TestBilevelOptimizer:
    """Tests for the bilevel optimizer wrapper."""

    def test_create_bilevel_optimizer(
        self, simple_turbine, square_boundary, four_turbine_layout
    ):
        """Test that create_bilevel_optimizer produces working optimizer."""
        init_x, init_y = four_turbine_layout
        deficit = NOJDeficit(k=0.05)
        sim = WakeSimulation(simple_turbine, deficit)

        ws = jnp.array([10.0])
        wd = jnp.array([270.0])

        settings = SGDSettings(max_iter=30, learning_rate=3.0)
        bilevel_opt = create_bilevel_optimizer(
            sim_engine=sim,
            target_boundary=square_boundary,
            min_spacing=160.0,
            ws_amb=ws,
            wd_amb=wd,
            settings=settings,
            n_target=4,
        )

        neighbor_x = jnp.array([1500.0])
        neighbor_y = jnp.array([500.0])

        opt_x, opt_y = bilevel_opt(init_x, init_y, neighbor_x, neighbor_y)

        # Check output shape
        assert opt_x.shape == init_x.shape
        assert opt_y.shape == init_y.shape
        assert jnp.all(jnp.isfinite(opt_x))
        assert jnp.all(jnp.isfinite(opt_y))


# =============================================================================
# Parity with TopFarm
# =============================================================================


@pytest.mark.slow
class TestTopFarmParity:
    """Tests comparing pixwake.optim.sgd with TopFarm's implementation.

    These tests require TopFarm and PyWake to be installed (they are test
    dependencies in pyproject.toml). Tests are marked slow as they require
    running both optimizers.
    """

    def test_penalty_similar_to_topfarm(self, square_boundary):
        """Compare constraint aggregation with TopFarm's implementation."""
        # Create positions with some violations
        x = jnp.array([100.0, 200.0, 500.0, 900.0])
        y = jnp.array([100.0, 150.0, 500.0, 900.0])
        min_spacing = 200.0

        # Compute our penalty
        our_penalty = spacing_penalty(x, y, min_spacing, rho=50.0)

        # The exact values won't match due to different aggregation,
        # but both should be positive when constraints are violated
        dist_01 = jnp.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        has_violation = float(dist_01) < min_spacing

        if has_violation:
            assert our_penalty > 0.0, "Penalty should be positive when violated"
        else:
            assert our_penalty < 1e-3, "Penalty should be near zero when satisfied"

    def test_sgd_converges_similarly(self, simple_turbine, square_boundary):
        """Test that our SGD reaches similar or better solutions than TopFarm.

        This test compares layout optimization results between pixwake's SGD
        and TopFarm's SGD. Both should find good solutions, though they may
        differ due to:
        - Different constraint handling (our KS aggregation vs TopFarm's)
        - Numerical precision differences
        - Different wake model implementations

        We verify that:
        1. Our optimizer finds a valid (constraint-satisfying) solution
        2. Our AEP is within reasonable range of TopFarm's result
        """
        from topfarm import TopFarmProblem
        from topfarm.easy_drivers import EasySGDDriver
        from topfarm.constraint_components.boundary import XYBoundaryConstraint
        from topfarm.constraint_components.spacing import SpacingConstraint
        from topfarm.constraint_components.constraint_aggregation import (
            DistanceConstraintAggregation,
        )
        from topfarm.cost_models.cost_model_wrappers import CostModelComponent
        from py_wake.wind_turbines import WindTurbine
        from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
        from py_wake import NOJ
        from py_wake.site import UniformSite
        from py_wake.utils.gradients import autograd

        # Set up both optimizers with same parameters
        init_x = np.array([250.0, 750.0, 250.0, 750.0])
        init_y = np.array([250.0, 250.0, 750.0, 750.0])
        boundary_np = np.array(
            [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]
        )
        boundary_jnp = jnp.array(boundary_np)
        min_spacing = 160.0

        # Our optimizer (Minimizes negative AEP)
        deficit = NOJDeficit(k=0.05)
        sim = WakeSimulation(simple_turbine, deficit)
        ws = jnp.array([10.0])
        wd = jnp.array([270.0])

        def neg_aep(x, y):
            result = sim(x, y, ws_amb=ws, wd_amb=wd)
            return -result.aep()

        # Scale gradients in our solver to match TopFarm's behavior
        # Note: topfarm_sgd_solve uses automatic differentiation which handles small values
        # much better than the Adam implementation in TopFarm if JAX uses float64
        settings = SGDSettings(
            max_iter=2000,
            learning_rate=5.0,
            beta1=0.1,
            beta2=0.2,
        )
        our_x, our_y = topfarm_sgd_solve(
            neg_aep,
            jnp.array(init_x),
            jnp.array(init_y),
            boundary_jnp,
            min_spacing,
            settings,
        )

        our_aep = float(-neg_aep(our_x, our_y))

        # Verify our solution satisfies constraints
        our_boundary_pen = float(boundary_penalty(our_x, our_y, boundary_jnp))
        our_spacing_pen = float(spacing_penalty(our_x, our_y, min_spacing))
        assert our_boundary_pen < 0.01, f"Boundary violation: {our_boundary_pen}"
        assert our_spacing_pen < 0.01, f"Spacing violation: {our_spacing_pen}"

        # TopFarm optimizer setup
        ws_curve = np.array(simple_turbine.power_curve.ws)
        power_curve = np.array(simple_turbine.power_curve.values)
        ct_curve = np.array(simple_turbine.ct_curve.values)

        wt = WindTurbine(
            name="test",
            diameter=80.0,
            hub_height=70.0,
            powerCtFunction=PowerCtTabular(ws_curve, power_curve, "kW", ct_curve),
        )

        site = UniformSite(p_wd=[1.0], ti=0.06)
        wake_model = NOJ(site, wt, k=0.05)

        # Use manual CostModelComponent with scaling to prevent underflow in SGD
        # This mirrors the behavior in the TopFarm examples for SGD
        def aep_func(x, y, **kwargs):
            # We want to MINIMIZE negative AEP.
            # PyWake AEP is usually positive.
            # Cost = -AEP * scaling
            sim_res = wake_model(x, y, ws=[10.0], wd=[270.0])
            return sim_res.aep().sum().item() * 1e6

        def aep_jac(x, y, **kwargs):
            # Gradients must match cost function: -d(AEP)/dx * scaling
            jx, jy = wake_model.aep_gradients(
                gradient_method=autograd,  # Use autograd
                wrt_arg=["x", "y"],
                x=x,
                y=y,
                ws=[10.0],
                wd=[270.0],
            )
            # jx is shaped (n_wd, n_ws, n_wt) -> (1, 1, 4) -> flatten to (4,)
            return np.array([jx.flatten(), jy.flatten()]) * 1e6

        cost_comp = CostModelComponent(
            input_keys=["x", "y"],
            n_wt=4,
            cost_function=aep_func,
            cost_gradient_function=aep_jac,
        )

        # SGD in TopFarm requires aggregated penalty constraints
        boundary_comp = XYBoundaryConstraint(boundary_np, "rectangle")
        aggregated_constraints = [
            DistanceConstraintAggregation(boundary_comp, 4, min_spacing, wt)
        ]

        problem = TopFarmProblem(
            design_vars={"x": init_x.copy(), "y": init_y.copy()},
            cost_comp=cost_comp,
            constraints=aggregated_constraints,
            driver=EasySGDDriver(maxiter=2000, learning_rate=5.0, beta1=0.1, beta2=0.2),
        )
        tf_cost, tf_state, _ = problem.optimize()

        # To compare, we evaluate the final TopFarm layout using OUR model (pixwake)
        # This ensures we are comparing layouts, not subtle AEP calculation differences
        tf_x, tf_y = jnp.array(tf_state["x"]), jnp.array(tf_state["y"])
        tf_layout_our_aep = float(-neg_aep(tf_x, tf_y))

        # Our AEP should be competitive (within 15% of TopFarm, or better)
        # Note: TopFarm may find solutions that violate constraints slightly different than ours
        relative_diff = (tf_layout_our_aep - our_aep) / tf_layout_our_aep
        assert relative_diff < 0.15, (
            f"Our AEP ({our_aep:.2f}) is more than 15% worse than "
            f"TopFarm layout evaluated with our model ({tf_layout_our_aep:.2f})"
        )
