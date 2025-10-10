import jax.numpy as jnp
import pytest

from pixwake import Curve, Turbine
from pixwake.core import SimulationContext, SimulationResult


@pytest.fixture
def simulation_result():
    power_curve = Curve(
        wind_speed=jnp.array([4.0, 10.0, 25.0]),
        values=jnp.array([0.0, 2000.0, 2000.0]),
    )
    ct_curve = Curve(
        wind_speed=jnp.array([4.0, 10.0, 25.0]),
        values=jnp.array([0.8, 0.8, 0.4]),
    )
    turbine = Turbine(
        rotor_diameter=100.0,
        hub_height=80.0,
        power_curve=power_curve,
        ct_curve=ct_curve,
    )
    effective_wind_speed = jnp.array([[10.0, 12.0], [8.0, 6.0]])

    return SimulationResult(
        turbine,
        wt_x=jnp.array([[0.0, 500.0], [0.0, 500.0]]),
        wt_y=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        wd=jnp.array([[270.0], [270.0]]),
        ws=jnp.array([[10.0], [10.0]]),
        effective_ws=effective_wind_speed,
    )


def test_power(simulation_result):
    power = simulation_result.power()
    expected_power = jnp.array([[2000.0, 2000.0], [1333.33333333, 666.66666667]])
    assert jnp.allclose(power, expected_power)


def test_aep(simulation_result):
    aep = simulation_result.aep()
    expected_aep = 26.28
    assert jnp.allclose(aep, expected_aep)


def test_aep_with_probabilities(simulation_result):
    probabilities = jnp.array([[0.6], [0.4]])
    aep = simulation_result.aep(probabilities=probabilities)
    expected_aep = 28.032
    assert jnp.allclose(aep, expected_aep, rtol=1e-5)
