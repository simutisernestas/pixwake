"""Pixwake optimization module.

This module provides optimization solvers for wind farm layout optimization
that are fully compatible with JAX transformations including automatic
differentiation.

Main components:
    - topfarm_sgd_solve: TopFarm-style SGD optimizer with ADAM momentum
    - sgd_solve_implicit: SGD with implicit differentiation for bilevel optimization
    - create_layout_optimizer: Convenience wrapper for AEP maximization
    - create_bilevel_optimizer: Wrapper for adversarial neighbor discovery

The solvers mirror TopFarm's EasySGDDriver implementation, enabling:
1. Constrained layout optimization with boundary and spacing constraints
2. Implicit differentiation through the optimization process
3. Bilevel optimization where an adversary differentiates through the
   target farm's layout optimization

Example:
    >>> from pixwake.optim import create_layout_optimizer, SGDSettings
    >>> from pixwake import WakeSimulation
    >>>
    >>> # Create optimizer
    >>> settings = SGDSettings(learning_rate=10.0, max_iter=1000)
    >>> optimizer = create_layout_optimizer(
    ...     sim_engine=sim,
    ...     boundary=boundary_vertices,
    ...     min_spacing=200.0,
    ...     ws_amb=ws,
    ...     wd_amb=wd,
    ...     settings=settings,
    ... )
    >>>
    >>> # Optimize layout
    >>> opt_x, opt_y = optimizer(init_x, init_y)

For bilevel optimization (adversarial discovery):
    >>> from pixwake.optim import create_bilevel_optimizer
    >>>
    >>> bilevel_opt = create_bilevel_optimizer(sim, boundary, min_spacing, ws, wd)
    >>>
    >>> def adversarial_objective(neighbor_x, neighbor_y):
    ...     # Inner: optimize target layout given neighbors
    ...     opt_x, opt_y = bilevel_opt(init_x, init_y, neighbor_x, neighbor_y)
    ...     # Compute final AEP with neighbors
    ...     x_all = jnp.concatenate([opt_x, neighbor_x])
    ...     y_all = jnp.concatenate([opt_y, neighbor_y])
    ...     result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    ...     return -result.aep()  # Minimize target AEP
    >>>
    >>> # Outer: gradient of final AEP w.r.t. neighbor positions
    >>> grad = jax.grad(adversarial_objective, argnums=(0, 1))
    >>> grad_x, grad_y = grad(neighbor_x, neighbor_y)

References:
    TopFarm: https://github.com/DTUWindEnergy/TopFarm2
    Implicit Function Theorem for optimization:
        Blondel et al., "Efficient and Modular Implicit Differentiation", NeurIPS 2022
"""

from pixwake.optim.sgd import (
    SGDSettings,
    SGDState,
    boundary_penalty,
    create_bilevel_optimizer,
    create_layout_optimizer,
    sgd_solve_implicit,
    spacing_penalty,
    topfarm_sgd_solve,
)

__all__ = [
    "SGDSettings",
    "SGDState",
    "topfarm_sgd_solve",
    "sgd_solve_implicit",
    "create_layout_optimizer",
    "create_bilevel_optimizer",
    "boundary_penalty",
    "spacing_penalty",
]
