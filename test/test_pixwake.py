import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from pixwake import (
    batched_simulate_case,
    fixed_point,
    simulate_case,
    wake_step,
)


def sqrt_iter(a, x):
    """Newton iteration for ``sqrt(a)``."""
    return 0.5 * (x + a / x)


def test_fixed_point_sqrt():
    a = 2.0
    x0 = 1.0
    res = fixed_point(sqrt_iter, a, x0, tol=1e-8)
    assert jnp.allclose(res, jnp.sqrt(a), rtol=1e-6)


def test_fixed_point_gradient():
    a = 2.0
    x0 = 1.0
    grad_fn = jax.grad(lambda aa: fixed_point(sqrt_iter, aa, x0, tol=1e-8))
    grad = grad_fn(a)
    expected = 1.0 / (2 * jnp.sqrt(a))
    assert jnp.allclose(grad, expected, rtol=1e-6)


def base_params():
    xs = jnp.array([0.0, 500.0])
    ys = jnp.array([0.0, 0.0])
    ws = jnp.array([10.0, 10.0])
    wd = 270.0
    D = 100.0
    k = 0.05
    ct_curve = jnp.stack([jnp.array([0.0, 20.0]), jnp.array([0.8, 0.8])], axis=1)
    return xs, ys, ws, wd, D, k, ct_curve


def rect_grid_params(nx=3, ny=2):
    xs, ys = jnp.meshgrid(
        jnp.linspace(0.0, 500.0 * (nx - 1), nx), jnp.linspace(0.0, 500.0 * (ny - 1), ny)
    )
    xs = xs.ravel()
    ys = ys.ravel()
    ws = jnp.full(xs.shape, 10.0)
    wd = 270.0
    D = 100.0
    k = 0.05
    ct_curve = jnp.stack([jnp.array([0.0, 20.0]), jnp.array([0.8, 0.8])], axis=1)
    return xs, ys, ws, wd, D, k, ct_curve


def test_wake_step_two_turbines():
    xs, ys, ws, wd, D, k, ct_curve = base_params()
    ct_xp, ct_fp = ct_curve[:, 0], ct_curve[:, 1]
    a = (xs, ys, ws, wd, D, k, ct_xp, ct_fp)
    result = wake_step(a, ws)
    expected = jnp.array([10.0, 7.5154347])
    assert jnp.allclose(result, expected, rtol=1e-6)


def test_simulate_case_two_turbines():
    xs, ys, ws, wd, D, k, ct_curve = base_params()
    result = simulate_case(xs, ys, ws, wd, D, k, ct_curve)
    expected = jnp.array([10.0, 7.5154343])
    assert jnp.allclose(result, expected, rtol=1e-6)


def test_batched_simulate_case_matches_individual():
    xs, ys, ws, wd, D, k, ct_curve = base_params()
    ws = jnp.array([10.0, 12.0])
    wd = jnp.array([270.0, 270.0])
    batch_res = batched_simulate_case(xs, ys, ws, wd, D, k, ct_curve)
    indiv = jnp.stack(
        [simulate_case(xs, ys, ws[i], wd[i], D, k, ct_curve) for i in range(2)],
        axis=0,
    )
    assert jnp.allclose(batch_res, indiv, rtol=1e-6)


def test_simulate_case_gradients_and_jit():
    xs, ys, ws, wd, D, k, ct_curve = rect_grid_params()

    def f(xx, yy):
        return simulate_case(xx, yy, ws, wd, D, k, ct_curve)

    check_grads(f, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)

    jitted = jax.jit(f)
    assert jnp.allclose(jitted(xs, ys), f(xs, ys), rtol=1e-6)


def test_batched_simulate_case_jit():
    xs, ys, ws, wd, D, k, ct_curve = rect_grid_params()
    ws_b = jnp.stack([ws, ws + 2.0])
    wd_b = jnp.stack([wd, wd])
    expected = batched_simulate_case(xs, ys, ws_b, wd_b, D, k, ct_curve)
    jitted = jax.jit(batched_simulate_case)
    assert jnp.allclose(jitted(xs, ys, ws_b, wd_b, D, k, ct_curve), expected, rtol=1e-6)
