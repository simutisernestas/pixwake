import jax
import jax.numpy as jnp
from jax import config as jcfg
from jax.test_util import check_grads

from pixwake.core import WakeSimulation, fixed_point
from pixwake.models.noj import NOJModel
from pixwake.types import Curve, Turbine

jcfg.update("jax_enable_x64", True)  # need float64 to match pywake


def sqrt_iter(x, a):
    """Newton iteration for ``sqrt(a)``."""
    return 0.5 * (x + a / x)


def test_fixed_point_sqrt():
    a = 2.0
    x0 = 1.0
    res = fixed_point(sqrt_iter, x0, a, tol=1e-8)
    assert jnp.allclose(res, jnp.sqrt(a), rtol=1e-6)


def test_fixed_point_gradient():
    a = 2.0
    x0 = 1.0
    grad_fn = jax.grad(lambda aa: fixed_point(sqrt_iter, x0, aa, tol=1e-8))
    grad = grad_fn(a)
    expected = 1.0 / (2 * jnp.sqrt(a))
    assert jnp.allclose(grad, expected, rtol=1e-6)


def base_params():
    xs = jnp.array([0.0, 500.0])
    ys = jnp.array([0.0, 0.0])
    ws = jnp.array([10.0, 10.0])
    wd = 270.0
    k = 0.05
    ct_curve_array = jnp.stack([jnp.array([0.0, 20.0]), jnp.array([0.8, 0.8])], axis=1)
    power_curve_array = jnp.stack(
        [jnp.array([0.0, 20.0]), jnp.array([0.0, 3000.0])], axis=1
    )
    turbine = Turbine(
        rotor_diameter=100.0,
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
        ),
        ct_curve=Curve(wind_speed=ct_curve_array[:, 0], values=ct_curve_array[:, 1]),
    )
    return xs, ys, ws, wd, k, turbine


def rect_grid_params(nx=3, ny=2):
    xs, ys = jnp.meshgrid(
        jnp.linspace(0.0, 500.0 * (nx - 1), nx), jnp.linspace(0.0, 500.0 * (ny - 1), ny)
    )
    xs = xs.ravel()
    ys = ys.ravel()
    ws = 10.0
    wd = 270.0
    k = 0.05
    ct_curve_array = jnp.stack([jnp.array([0.0, 20.0]), jnp.array([0.8, 0.8])], axis=1)
    power_curve_array = jnp.stack(
        [jnp.array([0.0, 20.0]), jnp.array([0.0, 3000.0])], axis=1
    )
    turbine = Turbine(
        rotor_diameter=100.0,
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
        ),
        ct_curve=Curve(wind_speed=ct_curve_array[:, 0], values=ct_curve_array[:, 1]),
    )
    return xs, ys, ws, wd, k, turbine


from pixwake.core import SimulationState

def test_wake_step_two_turbines():
    xs, ys, ws, wd, k, turbine = base_params()
    model = NOJModel(k=k)
    state = SimulationState.create(xs, ys, ws, wd, turbine)
    result = model.compute_deficit(ws, state)
    expected = jnp.array([10.0, 7.5154347])
    assert jnp.allclose(result, expected, rtol=1e-6)


def test_simulate_case_two_turbines():
    xs, ys, ws, wd, k, turbine = base_params()
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd), turbine)
    expected = jnp.array([10.0, 7.5154343])
    assert jnp.allclose(result, expected, rtol=1e-6)


def test_simulate_case_gradients_and_jit():
    xs, ys, ws, wd, k, turbine = rect_grid_params()
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)

    def f(xx, yy):
        return sim(xx, yy, jnp.full_like(xx, ws), jnp.full_like(xx, wd), turbine)

    check_grads(f, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)

    jitted = jax.jit(f)
    assert jnp.allclose(jitted(xs, ys), f(xs, ys), rtol=1e-6)


def test_batched_simulate_case_jit():
    xs, ys, ws, wd, k, turbine = rect_grid_params()
    ws_b = jnp.stack([jnp.full_like(xs, ws), jnp.full_like(xs, ws + 2.0)])
    wd_b = jnp.stack([jnp.full_like(xs, wd), jnp.full_like(xs, wd)])
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    expected = sim(xs, ys, ws_b, wd_b, turbine)
    jitted = jax.jit(sim)
    assert jnp.allclose(jitted(xs, ys, ws_b, wd_b, turbine), expected, rtol=1e-6)


def test_single_turbine():
    xs, ys, ws, wd, k, turbine = base_params()
    xs, ys = jnp.atleast_1d(xs[0]), jnp.atleast_1d(ys[0])
    ws = jnp.atleast_1d(ws[0])
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    result = sim(xs, ys, ws, jnp.atleast_1d(wd), turbine)
    assert jnp.allclose(result, ws, rtol=1e-6)


def test_zero_wind_speed():
    xs, ys, _, wd, k, turbine = base_params()
    ws = jnp.array([0.0, 0.0])
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd), turbine)
    assert jnp.allclose(result, jnp.zeros_like(result), rtol=1e-6)


def test_wind_speed_outside_ct_curve():
    xs, ys, _, wd, k, turbine = base_params()
    ws = jnp.array([100.0, 100.0])  # Way outside the curve
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd), turbine)
    # The model should still produce a result, likely with the max Ct value
    assert jnp.isfinite(result).all()


def test_identical_turbine_locations():
    xs, ys, ws, wd, k, turbine = base_params()
    xs = jnp.array([0.0, 1e-6])
    ys = jnp.array([0.0, 0.0])
    ws = jnp.array([10.0, 10.0])
    model = NOJModel(k=k, damp=1.0)
    sim = WakeSimulation(model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd), turbine)
    # Deficit should be very high for the second turbine
    assert result[0, 0] > result[0, 1]
