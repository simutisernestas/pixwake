"""JAX-native constrained SGD optimizer mirroring TopFarm's EasySGDDriver.

This module provides a fully differentiable SGD solver that enables bilevel
optimization through the Implicit Function Theorem. An adversary can
differentiate through the entire layout optimization process to find
neighbor configurations that maximize regret.

The implementation mirrors TopFarm's SGD with ADAM momentum:
- First moment: m = beta1 * m + (1 - beta1) * grad
- Second moment: v = beta2 * v + (1 - beta2) * grad^2
- Update: x -= lr * m_hat / (sqrt(v_hat) + eps)

Constraints are handled via differentiable penalty functions:
- Boundary constraint: KS aggregation of distance violations
- Spacing constraint: KS aggregation of inter-turbine distance violations

References:
    TopFarm SGD: topfarm/drivers/stochastic_gradient_descent_driver.py
    TopFarm constraints: topfarm/constraint_components/constraint_aggregation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.lax import while_loop

# =============================================================================
# SGD State (Optax-style stateless design)
# =============================================================================


class SGDState(NamedTuple):
    """State container for SGD optimizer.

    Follows Optax pattern for stateless optimization.

    Attributes:
        m: First moment estimate (momentum).
        v: Second moment estimate (adaptive scaling).
        iteration: Current iteration count.
        learning_rate: Current learning rate.
        alpha: Current constraint penalty coefficient.
    """

    m_x: jnp.ndarray
    m_y: jnp.ndarray
    v_x: jnp.ndarray
    v_y: jnp.ndarray
    iteration: int
    learning_rate: float
    alpha: float
    alpha0: float
    lr0: float


@dataclass(frozen=True)
class SGDSettings:
    """Configuration for TopFarm-style SGD optimizer.

    The learning rate decays from `learning_rate` to `learning_rate * gamma_min_factor`
    over `max_iter` iterations using the decay function:
        lr_t = lr_{t-1} * 1/(1 + mid * t)

    The `mid` parameter is computed via bisection search to achieve the target
    final learning rate, matching TopFarm's EasySGDDriver behavior.

    Attributes:
        learning_rate: Initial learning rate (default: 10.0).
        gamma_min_factor: Final learning rate as fraction of initial (default: 0.01).
            Final lr = learning_rate * gamma_min_factor.
        beta1: First moment decay rate (default: 0.1).
        beta2: Second moment decay rate (default: 0.2).
        max_iter: Maximum number of iterations (default: 3000).
        tol: Convergence tolerance on gradient norm (default: 1e-6).
        mid: Learning rate decay factor. If None, computed via bisection to achieve
            gamma_min_factor. Set explicitly to override bisection.
        bisect_upper: Upper bound for bisection search (default: 0.1).
        bisect_lower: Lower bound for bisection search (default: 0.0).
        ks_rho: KS aggregation smoothness parameter (default: 100.0).
        spacing_weight: Weight for spacing penalty (default: 1.0).
        boundary_weight: Weight for boundary penalty (default: 1.0).
    """

    learning_rate: float = 10.0
    gamma_min_factor: float = 0.01
    beta1: float = 0.1
    beta2: float = 0.2
    max_iter: int = 3000
    tol: float = 1e-6
    mid: float | None = None
    bisect_upper: float = 0.1
    bisect_lower: float = 0.0
    ks_rho: float = 100.0
    spacing_weight: float = 1.0
    boundary_weight: float = 1.0


def _compute_mid_bisection(
    learning_rate: float,
    gamma_min: float,
    max_iter: int,
    lower: float = 0.0,
    upper: float = 0.1,
    n_bisect_iter: int = 100,
) -> float:
    """Compute the learning rate decay parameter via bisection search.

    Finds `mid` such that after `max_iter` steps of decay:
        lr_t = lr_{t-1} * 1/(1 + mid * t)
    the final learning rate equals `gamma_min`.

    This matches TopFarm's SGDDriver initialization.

    Args:
        learning_rate: Initial learning rate.
        gamma_min: Target final learning rate.
        max_iter: Number of optimization iterations.
        lower: Lower bound for bisection (default: 0.0).
        upper: Upper bound for bisection (default: 0.1).
        n_bisect_iter: Number of bisection iterations (default: 100).

    Returns:
        The computed `mid` parameter.
    """

    def final_lr(mid: float) -> float:
        """Compute final learning rate for a given mid value."""
        lr = learning_rate
        for t in range(1, max_iter + 1):
            lr = lr * 1.0 / (1.0 + mid * t)
        return lr

    # Bisection search
    for _ in range(n_bisect_iter):
        mid = (lower + upper) / 2.0
        lr_final = final_lr(mid)
        if lr_final < gamma_min:
            upper = mid
        else:
            lower = mid

    return mid


# =============================================================================
# Constraint Penalty Functions (JAX-native, differentiable)
# =============================================================================


def _signed_distance_to_edge(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> jnp.ndarray:
    """Compute signed distance from points to a line segment edge.

    Positive distance means inside (to the left of edge when traversing CCW).
    Negative distance means outside.

    Args:
        px, py: Point coordinates, shape (n_turbines,).
        x1, y1: Edge start point.
        x2, y2: Edge end point.

    Returns:
        Signed distances, shape (n_turbines,).
    """
    # Edge vector
    edge_x = x2 - x1
    edge_y = y2 - y1
    edge_len = jnp.sqrt(edge_x**2 + edge_y**2) + 1e-10

    # Unit normal pointing inward (90 deg CCW rotation of edge direction)
    normal_x = -edge_y / edge_len
    normal_y = edge_x / edge_len

    # Vector from edge start to point
    ap_x = px - x1
    ap_y = py - y1

    # Signed distance = dot(AP, normal)
    return ap_x * normal_x + ap_y * normal_y


def boundary_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    boundary_vertices: jnp.ndarray,
    rho: float = 100.0,
) -> jnp.ndarray:
    """Compute differentiable boundary constraint penalty using KS aggregation.

    For a convex polygon boundary, computes the signed distance from each
    turbine to each edge. Violated constraints (negative distance) are
    aggregated using the Kreisselmeier-Steinhauser (KS) function for
    smooth, differentiable penalties.

    KS penalty: max(0, -d) aggregated smoothly as:
        C = (1/rho) * log(sum(exp(rho * max(0, -d))))

    Args:
        x: Turbine x positions, shape (n_turbines,).
        y: Turbine y positions, shape (n_turbines,).
        boundary_vertices: Polygon vertices (CCW order), shape (n_vertices, 2).
        rho: KS smoothness parameter. Higher = sharper approximation to max.

    Returns:
        Scalar penalty value (0 if all constraints satisfied).
    """
    n_vertices = boundary_vertices.shape[0]

    # Compute signed distance to each edge for all turbines
    def edge_distances(i: int) -> jnp.ndarray:
        x1, y1 = boundary_vertices[i]
        x2, y2 = boundary_vertices[(i + 1) % n_vertices]
        return _signed_distance_to_edge(x, y, x1, y1, x2, y2)

    # Stack distances: shape (n_edges, n_turbines)
    all_distances = jax.vmap(edge_distances)(jnp.arange(n_vertices))

    # For convex polygon, turbine is inside if ALL edge distances are positive
    # The minimum distance determines how "inside" we are
    min_distances = jnp.min(all_distances, axis=0)  # shape (n_turbines,)

    # Penalty for violations (negative distances)
    # Using squared penalty for smooth gradients at boundary
    violations = jnp.maximum(0.0, -min_distances)

    # KS aggregation for smooth differentiability
    # Avoid numerical issues with large rho*violations
    max_viol = jnp.max(violations)
    if_nonzero = (1.0 / rho) * (
        max_viol + jnp.log(jnp.sum(jnp.exp(rho * (violations - max_viol))) + 1e-10)
    )
    return jnp.where(max_viol > 0, if_nonzero, 0.0)


def spacing_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    min_spacing: float,
    rho: float = 100.0,
) -> jnp.ndarray:
    """Compute differentiable spacing constraint penalty using KS aggregation.

    Penalizes turbine pairs that are closer than min_spacing. Uses the
    Kreisselmeier-Steinhauser function for smooth aggregation.

    Constraint: d_ij >= min_spacing for all i != j
    Violation: v_ij = max(0, min_spacing^2 - d_ij^2)

    Args:
        x: Turbine x positions, shape (n_turbines,).
        y: Turbine y positions, shape (n_turbines,).
        min_spacing: Minimum allowed distance between turbines.
        rho: KS smoothness parameter.

    Returns:
        Scalar penalty value (0 if all constraints satisfied).
    """
    n = x.shape[0]
    if n < 2:
        return jnp.array(0.0)

    # Compute pairwise squared distances
    dx = x[:, None] - x[None, :]  # shape (n, n)
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2

    # Extract upper triangle (unique pairs)
    i_upper, j_upper = jnp.triu_indices(n, k=1)
    pair_dist_sq = dist_sq[i_upper, j_upper]

    # Constraint: d^2 >= min_spacing^2
    # Violation: v = max(0, min_spacing^2 - d^2)
    min_spacing_sq = min_spacing**2
    violations = jnp.maximum(0.0, min_spacing_sq - pair_dist_sq)

    # Normalize by min_spacing^2 for scale invariance
    violations_normalized = violations / (min_spacing_sq + 1e-10)

    # KS aggregation
    max_viol = jnp.max(violations_normalized)
    if_nonzero = (1.0 / rho) * (
        max_viol
        + jnp.log(jnp.sum(jnp.exp(rho * (violations_normalized - max_viol))) + 1e-10)
    )
    return jnp.where(max_viol > 0, if_nonzero, 0.0)


# =============================================================================
# SGD Solver Core
# =============================================================================


def _init_sgd_state(
    x: jnp.ndarray,
    y: jnp.ndarray,
    grad_x: jnp.ndarray,
    grad_y: jnp.ndarray,
    settings: SGDSettings,
) -> SGDState:
    """Initialize SGD state following TopFarm's initialization.

    The initial alpha0 is computed from the initial gradient magnitude
    to balance objective and constraint contributions.

    Args:
        x, y: Initial turbine positions.
        grad_x, grad_y: Initial gradients of the objective.
        settings: SGD configuration.

    Returns:
        Initialized SGDState.
    """
    # Initialize moments to zeros
    m_x = jnp.zeros_like(x)
    m_y = jnp.zeros_like(y)
    v_x = jnp.zeros_like(x)
    v_y = jnp.zeros_like(y)

    # Compute initial alpha0 from gradient magnitude
    # alpha0 = mean(|grad|) / learning_rate (TopFarm: line 135-138)
    grad_mag = jnp.concatenate([jnp.abs(grad_x), jnp.abs(grad_y)])
    alpha0 = jnp.mean(grad_mag) / settings.learning_rate

    return SGDState(
        m_x=m_x,
        m_y=m_y,
        v_x=v_x,
        v_y=v_y,
        iteration=0,
        learning_rate=settings.learning_rate,
        alpha=alpha0,
        alpha0=alpha0,
        lr0=settings.learning_rate,
    )


def _sgd_step(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: SGDState,
    grad_obj_x: jnp.ndarray,
    grad_obj_y: jnp.ndarray,
    grad_con_x: jnp.ndarray,
    grad_con_y: jnp.ndarray,
    settings: SGDSettings,
) -> tuple[jnp.ndarray, jnp.ndarray, SGDState]:
    """Perform one SGD step with ADAM momentum.

    Implements TopFarm's SGD update rule:
    1. Combine objective and constraint gradients
    2. Update momentum estimates (ADAM)
    3. Apply bias correction
    4. Update positions
    5. Decay learning rate and update alpha

    Args:
        x, y: Current turbine positions.
        state: Current optimizer state.
        grad_obj_x, grad_obj_y: Gradients of the objective function.
        grad_con_x, grad_con_y: Gradients of the constraint penalty.
        settings: SGD configuration.

    Returns:
        Updated (x, y, state).
    """
    beta1, beta2 = settings.beta1, settings.beta2
    it = state.iteration + 1

    # Combined gradient: grad = grad_obj + alpha * grad_con
    jacobian_x = grad_obj_x + state.alpha * grad_con_x
    jacobian_y = grad_obj_y + state.alpha * grad_con_y

    # ADAM first moment update: m = beta1 * m + (1 - beta1) * grad
    m_x = beta1 * state.m_x + (1 - beta1) * jacobian_x
    m_y = beta1 * state.m_y + (1 - beta1) * jacobian_y

    # ADAM second moment update: v = beta2 * v + (1 - beta2) * grad^2
    v_x = beta2 * state.v_x + (1 - beta2) * jacobian_x**2
    v_y = beta2 * state.v_y + (1 - beta2) * jacobian_y**2

    # Bias correction
    m_hat_x = m_x / (1 - beta1**it)
    m_hat_y = m_y / (1 - beta1**it)
    v_hat_x = v_x / (1 - beta2**it)
    v_hat_y = v_y / (1 - beta2**it)

    # Position update: x -= lr * m_hat / (sqrt(v_hat) + eps)
    eps = 1e-12
    x_new = x - state.learning_rate * m_hat_x / (jnp.sqrt(v_hat_x) + eps)
    y_new = y - state.learning_rate * m_hat_y / (jnp.sqrt(v_hat_y) + eps)

    # Learning rate decay: lr *= 1 / (1 + mid * iter)
    mid = settings.mid if settings.mid is not None else 1.0 / settings.max_iter
    new_lr = state.learning_rate * 1.0 / (1 + mid * it)

    # Alpha update: alpha = alpha0 * lr0 / lr
    new_alpha = state.alpha0 * state.lr0 / new_lr

    new_state = SGDState(
        m_x=m_x,
        m_y=m_y,
        v_x=v_x,
        v_y=v_y,
        iteration=it,
        learning_rate=new_lr,
        alpha=new_alpha,
        alpha0=state.alpha0,
        lr0=state.lr0,
    )

    return x_new, y_new, new_state


# =============================================================================
# Main Solver with jax.lax.while_loop
# =============================================================================


def topfarm_sgd_solve(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve constrained layout optimization using TopFarm-style SGD.

    This solver uses jax.lax.while_loop for JIT compatibility and enables
    implicit differentiation through the convergence point.

    Args:
        objective_fn: Function (x, y) -> scalar to minimize (e.g., negative AEP).
        init_x: Initial turbine x positions, shape (n_turbines,).
        init_y: Initial turbine y positions, shape (n_turbines,).
        boundary: Polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration. Uses defaults if None.

    Returns:
        Tuple of (optimized_x, optimized_y).

    Example:
        >>> def neg_aep(x, y):
        ...     result = sim(x, y, ws_amb=ws, wd_amb=wd)
        ...     return -result.aep()
        >>> opt_x, opt_y = topfarm_sgd_solve(neg_aep, x0, y0, boundary, min_spacing)
    """
    if settings is None:
        settings = SGDSettings()

    # Compute mid via bisection if not explicitly provided
    if settings.mid is None:
        gamma_min = settings.learning_rate * settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=gamma_min,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        # Create new settings with computed mid
        settings = SGDSettings(
            learning_rate=settings.learning_rate,
            gamma_min_factor=settings.gamma_min_factor,
            beta1=settings.beta1,
            beta2=settings.beta2,
            max_iter=settings.max_iter,
            tol=settings.tol,
            mid=computed_mid,
            bisect_upper=settings.bisect_upper,
            bisect_lower=settings.bisect_lower,
            ks_rho=settings.ks_rho,
            spacing_weight=settings.spacing_weight,
            boundary_weight=settings.boundary_weight,
        )

    rho = settings.ks_rho

    def total_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Augmented objective including constraint penalties."""
        obj = objective_fn(x, y)
        pen_boundary = settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
        pen_spacing = settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
        return obj + pen_boundary + pen_spacing

    def constraint_penalty(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Combined constraint penalty for gradient computation."""
        return settings.boundary_weight * boundary_penalty(
            x, y, boundary, rho
        ) + settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)

    # Compute initial gradients for state initialization
    grad_obj_fn = jax.grad(objective_fn, argnums=(0, 1))
    grad_con_fn = jax.grad(constraint_penalty, argnums=(0, 1))

    init_grad_obj_x, init_grad_obj_y = grad_obj_fn(init_x, init_y)
    init_state = _init_sgd_state(
        init_x, init_y, init_grad_obj_x, init_grad_obj_y, settings
    )

    # State for while_loop: (x, y, sgd_state, prev_x, prev_y)
    def cond_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        x, y, state, prev_x, prev_y = carry
        # Continue if not converged and under max iterations
        change = jnp.max(jnp.abs(x - prev_x)) + jnp.max(jnp.abs(y - prev_y))
        not_converged = change > settings.tol
        under_max_iter = state.iteration < settings.max_iter
        return jnp.logical_and(not_converged, under_max_iter)

    def body_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray]:
        x, y, state, _, _ = carry

        # Compute gradients
        grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
        grad_con_x, grad_con_y = grad_con_fn(x, y)

        # SGD step
        x_new, y_new, new_state = _sgd_step(
            x, y, state, grad_obj_x, grad_obj_y, grad_con_x, grad_con_y, settings
        )

        return (x_new, y_new, new_state, x, y)

    # Run optimization loop
    init_carry = (init_x, init_y, init_state, init_x - 1.0, init_y - 1.0)
    final_x, final_y, _, _, _ = while_loop(cond_fn, body_fn, init_carry)

    return final_x, final_y


# =============================================================================
# Implicit Differentiation Wrapper
# =============================================================================


@partial(custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def sgd_solve_implicit(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """SGD solver with implicit differentiation support.

    This wrapper enables differentiating through the optimization process
    with respect to external parameters (e.g., neighbor turbine positions)
    using the Implicit Function Theorem. This is essential for bilevel
    optimization where an outer adversary wants to find parameters that
    maximize regret after inner layout optimization.

    The key insight is that at convergence, the optimality conditions hold:
        grad_x L(x*, y*, params) = 0
        grad_y L(x*, y*, params) = 0

    Differentiating these conditions with respect to params gives us the
    gradient of the optimal solution (x*, y*) with respect to params without
    unrolling the 3000 SGD iterations.

    Args:
        objective_fn: Function (x, y, params) -> scalar to minimize.
        init_x: Initial turbine x positions.
        init_y: Initial turbine y positions.
        boundary: Polygon vertices (CCW order).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration.
        params: External parameters to differentiate with respect to
            (e.g., neighbor positions).

    Returns:
        Tuple of (optimized_x, optimized_y).

    Example:
        >>> def objective_with_neighbors(x, y, neighbor_pos):
        ...     neighbor_x, neighbor_y = neighbor_pos[:n_neighbors], neighbor_pos[n_neighbors:]
        ...     x_all = jnp.concatenate([x, neighbor_x])
        ...     y_all = jnp.concatenate([y, neighbor_y])
        ...     result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        ...     target_aep = result.aep()[:n_target]
        ...     return -jnp.sum(target_aep)
        >>>
        >>> # Optimize target layout given neighbor positions
        >>> opt_x, opt_y = sgd_solve_implicit(
        ...     objective_with_neighbors, init_x, init_y, boundary, min_spacing, settings, neighbor_pos
        ... )
        >>>
        >>> # Differentiate final AEP with respect to neighbor positions
        >>> def final_aep(neighbor_pos):
        ...     opt_x, opt_y = sgd_solve_implicit(
        ...         objective_with_neighbors, init_x, init_y, boundary, min_spacing, settings, neighbor_pos
        ...     )
        ...     # ... compute final AEP ...
        >>> grad_neighbor = jax.grad(final_aep)(neighbor_pos)
    """
    if settings is None:
        settings = SGDSettings()

    # Wrap objective to include params
    def obj_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return objective_fn(x, y, params)

    return topfarm_sgd_solve(obj_fn, init_x, init_y, boundary, min_spacing, settings)


def _sgd_solve_implicit_fwd(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    params: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple]:
    """Forward pass for implicit differentiation."""
    if settings is None:
        settings = SGDSettings()

    # Wrap objective to include params
    def obj_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return objective_fn(x, y, params)

    # Call the underlying solver (not the custom_vjp wrapper)
    opt_x, opt_y = topfarm_sgd_solve(
        obj_fn, init_x, init_y, boundary, min_spacing, settings
    )
    # Store residuals for backward pass (only JAX-compatible types)
    # settings is passed through nondiff_argnums
    return (opt_x, opt_y), (opt_x, opt_y, params)


def _sgd_solve_implicit_bwd(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    res: tuple,
    g: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward pass using implicit function theorem.

    At optimality, we have:
        grad_x L(x*, y*, params) = 0
        grad_y L(x*, y*, params) = 0

    Differentiating with respect to params:
        H @ d(x*,y*)/d(params) + grad_params grad_(x,y) L = 0

    Where H is the Hessian of L with respect to (x, y).

    We solve for the gradient using the conjugate gradient method
    or a fixed-point iteration.
    """
    if settings is None:
        settings = SGDSettings()
    opt_x, opt_y, params = res
    g_x, g_y = g
    rho = settings.ks_rho

    # Total objective including penalties
    def total_obj(x: jnp.ndarray, y: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        obj = objective_fn(x, y, p)
        pen_b = settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
        pen_s = settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
        return obj + pen_b + pen_s

    # Compute VJP of optimality conditions with respect to params
    # Using the implicit function theorem:
    # d(x*,y*)/d(params) = -H^{-1} @ (d^2 L / d(x,y) d(params))

    # First, compute the mixed second derivative (gradient of gradient w.r.t. params)
    def grad_xy_wrt_params(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        grad_fn = jax.grad(lambda x, y: total_obj(x, y, p), argnums=(0, 1))
        return grad_fn(opt_x, opt_y)

    _, vjp_params_fn = vjp(grad_xy_wrt_params, params)

    # Solve the linear system H @ v = g using fixed-point iteration
    # v = g - (H - I) @ v, which converges for well-conditioned H

    def hvp(vx: jnp.ndarray, vy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Hessian-vector product."""

        def grad_at_opt(
            x: jnp.ndarray, y: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jax.grad(lambda xx, yy: total_obj(xx, yy, params), argnums=(0, 1))(
                x, y
            )

        # Use forward-over-reverse for Hessian-vector product
        primals = (opt_x, opt_y)
        tangents = (vx, vy)
        _, hvp_result = jax.jvp(grad_at_opt, primals, tangents)
        return hvp_result

    # Solve H @ v = g using conjugate gradient (simplified: fixed-point iteration)
    def solve_linear_system(
        g_x: jnp.ndarray, g_y: jnp.ndarray, max_iter: int = 50, tol: float = 1e-6
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Solve H @ v = g approximately using fixed-point iteration."""

        def cond_fn(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
        ) -> jnp.ndarray:
            _, _, r_x, r_y, it = carry
            residual = jnp.max(jnp.abs(r_x)) + jnp.max(jnp.abs(r_y))
            return jnp.logical_and(residual > tol, it < max_iter)

        def body_fn(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
            v_x, v_y, _, _, it = carry
            hv_x, hv_y = hvp(v_x, v_y)
            # Damped update: v_new = v + 0.1 * (g - H @ v)
            r_x = g_x - hv_x
            r_y = g_y - hv_y
            v_x_new = v_x + 0.1 * r_x
            v_y_new = v_y + 0.1 * r_y
            return (v_x_new, v_y_new, r_x, r_y, it + 1)

        # Initial guess: v = g (approximation for H close to identity)
        init_r_x, init_r_y = hvp(g_x, g_y)
        init_r_x = g_x - init_r_x
        init_r_y = g_y - init_r_y
        init_carry = (g_x, g_y, init_r_x, init_r_y, 0)
        v_x, v_y, _, _, _ = while_loop(cond_fn, body_fn, init_carry)
        return v_x, v_y

    # Solve for the adjoint vector
    adj_x, adj_y = solve_linear_system(g_x, g_y)

    # Compute gradient with respect to params
    (grad_params,) = vjp_params_fn((adj_x, adj_y))

    # Gradients with respect to init_x, init_y are zero (fixed point doesn't depend on initial guess)
    return (jnp.zeros_like(opt_x), jnp.zeros_like(opt_y), -grad_params)


sgd_solve_implicit.defvjp(_sgd_solve_implicit_fwd, _sgd_solve_implicit_bwd)


# =============================================================================
# Convenience Wrapper for Common Use Case
# =============================================================================


def create_layout_optimizer(
    sim_engine,
    boundary: jnp.ndarray,
    min_spacing: float,
    ws_amb: jnp.ndarray | float,
    wd_amb: jnp.ndarray | float,
    ti_amb: jnp.ndarray | float | None = None,
    weights: jnp.ndarray | None = None,
    settings: SGDSettings | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Create a layout optimization function using TopFarm-style SGD.

    This is a convenience wrapper that creates a solver configured for
    AEP maximization with boundary and spacing constraints.

    Args:
        sim_engine: WakeSimulation instance.
        boundary: Polygon vertices (CCW order), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        ws_amb: Ambient wind speed(s).
        wd_amb: Wind direction(s).
        ti_amb: Ambient turbulence intensity (optional).
        weights: Probability weights for each wind condition (optional).
        settings: SGD configuration.

    Returns:
        Function (init_x, init_y) -> (opt_x, opt_y).

    Example:
        >>> optimizer = create_layout_optimizer(sim, boundary, 200.0, ws, wd)
        >>> opt_x, opt_y = optimizer(init_x, init_y)
    """
    if settings is None:
        settings = SGDSettings()

    def neg_aep(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        result = sim_engine(x, y, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
        if weights is not None:
            # Weighted AEP
            power = result.power()  # (n_cases, n_turbines)
            weighted_power = jnp.sum(power * weights[:, None])
            return -weighted_power * 8760 / 1e6  # Convert to GWh
        return -result.aep()

    def optimize(
        init_x: jnp.ndarray, init_y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return topfarm_sgd_solve(
            neg_aep, init_x, init_y, boundary, min_spacing, settings
        )

    return optimize


def create_bilevel_optimizer(
    sim_engine,
    target_boundary: jnp.ndarray,
    min_spacing: float,
    ws_amb: jnp.ndarray | float,
    wd_amb: jnp.ndarray | float,
    ti_amb: jnp.ndarray | float | None = None,
    weights: jnp.ndarray | None = None,
    settings: SGDSettings | None = None,
    n_target: int | None = None,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Create a bilevel layout optimizer for adversarial discovery.

    This function creates a solver that optimizes target farm layout while
    accounting for neighbor turbine wake effects. The solver supports implicit
    differentiation, allowing an outer optimizer to compute gradients of the
    optimized target AEP with respect to neighbor positions.

    Args:
        sim_engine: WakeSimulation instance.
        target_boundary: Target farm polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        ws_amb: Ambient wind speed(s).
        wd_amb: Wind direction(s).
        ti_amb: Ambient turbulence intensity (optional).
        weights: Probability weights for each wind condition (optional).
        settings: SGD configuration.
        n_target: Number of target turbines (inferred if None).

    Returns:
        Function (init_x, init_y, neighbor_x, neighbor_y) -> (opt_x, opt_y).

    Example:
        >>> bilevel_opt = create_bilevel_optimizer(sim, boundary, 200.0, ws, wd)
        >>>
        >>> def adversarial_objective(neighbor_x, neighbor_y):
        ...     opt_x, opt_y = bilevel_opt(init_x, init_y, neighbor_x, neighbor_y)
        ...     # Compute final target AEP with neighbors
        ...     x_all = jnp.concatenate([opt_x, neighbor_x])
        ...     y_all = jnp.concatenate([opt_y, neighbor_y])
        ...     result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        ...     return -result.aep()[:n_target].sum()  # Minimize target AEP
        >>>
        >>> # Gradient of optimized AEP w.r.t. neighbor positions
        >>> grad_x, grad_y = jax.grad(adversarial_objective, argnums=(0, 1))(
        ...     neighbor_x, neighbor_y
        ... )
    """
    if settings is None:
        settings = SGDSettings()

    def objective_with_neighbors(
        x: jnp.ndarray, y: jnp.ndarray, neighbor_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Objective function with neighbor parameters."""
        n_neighbors = neighbor_params.shape[0] // 2
        neighbor_x = neighbor_params[:n_neighbors]
        neighbor_y = neighbor_params[n_neighbors:]

        # Combine target and neighbor positions
        x_all = jnp.concatenate([x, neighbor_x])
        y_all = jnp.concatenate([y, neighbor_y])

        # Simulate
        result = sim_engine(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)

        # Target AEP only (first n_target turbines)
        _n_target = n_target if n_target is not None else x.shape[0]
        power = result.power()[:, :_n_target]

        if weights is not None:
            weighted_power = jnp.sum(power * weights[:, None])
            return -weighted_power * 8760 / 1e6
        return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def optimize_with_neighbors(
        init_x: jnp.ndarray,
        init_y: jnp.ndarray,
        neighbor_x: jnp.ndarray,
        neighbor_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Optimize target layout given neighbor positions."""
        neighbor_params = jnp.concatenate([neighbor_x, neighbor_y])
        return sgd_solve_implicit(
            objective_with_neighbors,
            init_x,
            init_y,
            target_boundary,
            min_spacing,
            settings,
            neighbor_params,
        )

    return optimize_with_neighbors
