import jax.numpy as jnp
import numpy as np
from pixwake import calculate_aep, calculate_power


from pixwake.types import Curve


def test_calculate_power():
    ws_eff = jnp.array([[10.0, 12.0], [15.0, 20.0]])
    power_curve_array = jnp.array([[0.0, 0.0], [10.0, 1000.0], [25.0, 3000.0]])
    power_curve = Curve(
        wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
    )
    pc_xp, pc_fp = power_curve.wind_speed, power_curve.values

    expected_power = jnp.array(
        [
            [jnp.interp(10.0, pc_xp, pc_fp), jnp.interp(12.0, pc_xp, pc_fp)],
            [jnp.interp(15.0, pc_xp, pc_fp), jnp.interp(20.0, pc_xp, pc_fp)],
        ]
    )

    actual_power = calculate_power(ws_eff, power_curve)
    assert jnp.allclose(actual_power, expected_power)


def test_calculate_aep_timeseries():
    ws_eff = jnp.array([[10.0, 12.0], [15.0, 20.0]])
    power_curve_array = jnp.array([[0.0, 0.0], [10.0, 1000.0], [25.0, 3000.0]])
    power_curve = Curve(
        wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
    )
    pc_xp, pc_fp = power_curve.wind_speed, power_curve.values
    powers = jnp.array(
        [
            [jnp.interp(10.0, pc_xp, pc_fp), jnp.interp(12.0, pc_xp, pc_fp)],
            [jnp.interp(15.0, pc_xp, pc_fp), jnp.interp(20.0, pc_xp, pc_fp)],
        ]
    )
    expected_aep = (powers * 1e3 * 24 * 365 * 1e-9).sum() / ws_eff.shape[0]
    actual_aep = calculate_aep(ws_eff, power_curve)
    assert jnp.allclose(actual_aep, expected_aep)


def test_calculate_aep_with_probabilities():
    ws_eff = jnp.array([[10.0, 12.0], [15.0, 20.0]])
    power_curve_array = jnp.array([[0.0, 0.0], [10.0, 1000.0], [25.0, 3000.0]])
    power_curve = Curve(
        wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
    )
    pc_xp, pc_fp = power_curve.wind_speed, power_curve.values
    powers = jnp.array(
        [
            [jnp.interp(10.0, pc_xp, pc_fp), jnp.interp(12.0, pc_xp, pc_fp)],
            [jnp.interp(15.0, pc_xp, pc_fp), jnp.interp(20.0, pc_xp, pc_fp)],
        ]
    )
    probs = jnp.array([0.4, 0.6])
    probs = probs.reshape(1, 2).T
    expected_aep = (powers * probs * 1e3 * 24 * 365 * 1e-9).sum()
    actual_aep = calculate_aep(ws_eff, power_curve, probabilities=probs)
    assert jnp.allclose(actual_aep, expected_aep)
