import jax.numpy as jnp
import numpy as np
import pytest

from pixwake.core import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit


@pytest.fixture
def turbine():
    """A pytest fixture to provide a Turbine object for tests."""
    return Turbine(
        rotor_diameter=126.0,
        hub_height=80.0,
        power_curve=Curve(
            wind_speed=jnp.array([4.0, 25.0]), values=jnp.array([0.0, 5000.0])
        ),
        ct_curve=Curve(wind_speed=jnp.array([4.0, 25.0]), values=jnp.array([0.8, 0.1])),
    )


def test_flow_map(turbine):
    """Test that the flow map is correctly generated and can be plotted."""
    sim = WakeSimulation(BastankhahGaussianDeficit())
    xs = jnp.array([0, 500])
    ys = jnp.array([0, 0])
    ws = jnp.array([10.0])
    wd = jnp.array([270.0])
    grid_x, grid_y = np.mgrid[0:1000:100j, -250:250:100j]

    result = sim(xs, ys, ws, wd, turbine, x=grid_x.flatten(), y=grid_y.flatten())

    assert result.flow_map_ws is not None
    assert result.flow_map_ws.shape == (1, 10000)

    # Verify that the plotting function can be called without error
    try:
        result.plot_flow_map(grid_x, grid_y)
    except Exception as e:
        pytest.fail(f"plot_flow_map failed with {e}")
