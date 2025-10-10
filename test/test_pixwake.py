import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from pixwake import Curve, Turbine
from pixwake.core import SimulationContext, WakeSimulation, fixed_point
from pixwake.deficit import (
    BastankhahGaussianDeficit,
    NiayifarGaussianDeficit,
    NOJDeficit,
)
from pixwake.turbulence import CrespoHernandez

jax.config.update("jax_enable_x64", True)


def _sqrt_iter(x, a):
    """Newton iteration for ``sqrt(a)``."""
    return 0.5 * (x + a / x)


def test_fixed_point_sqrt():
    a = 2.0
    x0 = 1.0
    res = fixed_point(_sqrt_iter, x0, a, tol=1e-8)
    assert jnp.allclose(res, jnp.sqrt(a), rtol=1e-6)


def test_fixed_point_gradient():
    a = 2.0
    x0 = 1.0
    grad_fn = jax.grad(lambda aa: fixed_point(_sqrt_iter, x0, aa, tol=1e-8))
    grad = grad_fn(a)
    expected = 1.0 / (2 * jnp.sqrt(a))
    assert jnp.allclose(grad, expected, rtol=1e-6)


def test_fixed_point_gradient_wrt_x_guess():
    a = 2.0
    x0 = 1.0
    # The gradient of fixed_point wrt x_guess is explicitly zero.
    grad_fn = jax.grad(lambda x: fixed_point(_sqrt_iter, x, a, tol=1e-8))
    grad = grad_fn(x0)
    assert jnp.allclose(grad, 0.0)


def base_params():
    xs = jnp.array([0.0, 500.0])
    ys = jnp.array([0.0, 0.0])
    # TODO: should test also the case with a float wind speed
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


def test_simulate_case_two_turbines():
    xs, ys, ws, wd, k, turbine = base_params()
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd))
    expected = jnp.array([10.0, 7.5154343])
    assert jnp.allclose(result.effective_ws, expected, rtol=1e-6)


def test_simulate_case_gradients_and_jit():
    xs, ys, ws, wd, k, turbine = rect_grid_params()
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)

    def f(xx, yy):
        return sim(xx, yy, jnp.full_like(xx, ws), jnp.full_like(xx, wd)).effective_ws

    check_grads(f, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)

    jitted = jax.jit(f)
    assert jnp.allclose(jitted(xs, ys), f(xs, ys), rtol=1e-6)


def test_single_turbine():
    xs, ys, ws, wd, k, turbine = base_params()
    xs, ys = jnp.atleast_1d(xs[0]), jnp.atleast_1d(ys[0])
    ws = jnp.atleast_1d(ws[0])
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.atleast_1d(wd))
    assert jnp.allclose(result.effective_ws, ws, rtol=1e-6)


def test_zero_wind_speed():
    xs, ys, _, wd, k, turbine = base_params()
    ws = jnp.array([0.0, 0.0])
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd))
    assert jnp.allclose(
        result.effective_ws,
        jnp.zeros_like(result.effective_ws),
        rtol=1e-6,
    )


def test_wind_speed_outside_ct_curve():
    xs, ys, _, wd, k, turbine = base_params()
    ws = jnp.array([100.0, 100.0])  # Way outside the curve
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd))
    # The model should still produce a result, likely with the max Ct value
    assert jnp.isfinite(result.effective_ws).all()


def test_identical_turbine_locations():
    xs, ys, ws, wd, k, turbine = base_params()
    xs = jnp.array([0.0, 1.0])  # 1 meter away
    ys = jnp.array([0.0, 0.0])
    ws = jnp.array([10.0, 10.0])
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd)).effective_ws
    # Deficit should be very high for the second turbine
    assert result[0, 0] > result[0, 1]


def test_numpy_inputs():
    xs, ys, ws, wd, k, turbine = base_params()
    deficit_model = NOJDeficit(k=k)

    # Use numpy arrays instead of jax arrays
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    ws_np = np.array(ws)
    wd_np = np.full_like(ws_np, wd)

    # Use lists for the curves
    power_curve_list = [turbine.power_curve.wind_speed, turbine.power_curve.values]
    ct_curve_list = [turbine.ct_curve.wind_speed, turbine.ct_curve.values]

    turbine_np = Turbine(
        rotor_diameter=turbine.rotor_diameter,
        hub_height=turbine.hub_height,
        power_curve=Curve(wind_speed=power_curve_list[0], values=power_curve_list[1]),
        ct_curve=Curve(wind_speed=ct_curve_list[0], values=ct_curve_list[1]),
    )
    sim = WakeSimulation(turbine_np, deficit_model, mapping_strategy="_manual")
    result = sim(xs_np, ys_np, ws_np, wd_np)
    expected = jnp.array([10.0, 7.5154343])
    assert jnp.allclose(result.effective_ws, expected, rtol=1e-6)


def test_invalid_mapping_strategy():
    xs, ys, ws, wd, k, turbine = base_params()
    deficit_model = NOJDeficit(k=k)
    with pytest.raises(ValueError):
        sim = WakeSimulation(
            turbine, deficit_model, mapping_strategy="invalid_strategy"
        )
        sim(xs, ys, ws, jnp.full_like(ws, wd))


def test_simulation_result_power_method():
    xs, ys, ws, wd, k, turbine = base_params()
    deficit_model = NOJDeficit(k=k)
    sim = WakeSimulation(turbine, deficit_model)
    result = sim(xs, ys, ws, jnp.full_like(ws, wd))
    power = result.power()
    expected_power = turbine.power(result.effective_ws)
    assert jnp.allclose(power, expected_power, rtol=1e-6)


def test_wake_simulation_manual_mapping_strategy():
    """Test that WakeSimulation runs with the manual mapping strategy."""
    deficit_model = NOJDeficit(k=0.1)
    power_curve = Curve(
        wind_speed=jnp.asarray([0.0, 10.0]), values=jnp.asarray([0.0, 1000.0])
    )
    ct_curve = Curve(
        wind_speed=jnp.asarray([0.0, 10.0]), values=jnp.asarray([0.8, 0.8])
    )
    turbine = Turbine(
        rotor_diameter=120.0,
        hub_height=100.0,
        power_curve=power_curve,
        ct_curve=ct_curve,
    )
    sim = WakeSimulation(turbine, deficit_model, mapping_strategy="_manual")

    sim(
        jnp.asarray([0.0, 1000.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([10.0, 12.0]),
        jnp.asarray([270.0, 270.0]),
    )


@pytest.mark.parametrize(
    "deficit_model,requires_ti",
    [
        (NOJDeficit(k=0.05), False),
        (BastankhahGaussianDeficit(), False),
        (BastankhahGaussianDeficit(), True),
        (NiayifarGaussianDeficit(), True),
    ],
)
def test_mapping_strategies_give_same_results(deficit_model, requires_ti):
    """Test that different mapping strategies give the same results for various models."""
    xs, ys, ws, wd, _, turbine = rect_grid_params(nx=5, ny=4)

    mapping_methods = getattr(
        WakeSimulation(deficit_model, turbine), "_WakeSimulation__sim_call_table"
    ).keys()

    results = []
    for map_strategy in mapping_methods:
        turbulence_model = CrespoHernandez()
        turb_arg = (turbulence_model,) if requires_ti else {}

        sim = WakeSimulation(
            turbine, deficit_model, *turb_arg, mapping_strategy=map_strategy
        )

        sim_args = {
            "wt_xs": xs,
            "wt_ys": ys,
            "ws_amb": jnp.full_like(xs, ws),
            "wd": jnp.full_like(xs, wd),
        }
        if requires_ti:
            sim_args["ti"] = 0.1

        res = sim(**sim_args)
        results.append(res.effective_ws)

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert jnp.allclose(results[i], results[j], rtol=1e-6)
