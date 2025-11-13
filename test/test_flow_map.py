import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pixwake.core import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit, NiayifarGaussianDeficit
from pixwake.plot import plot_flow_map


@pytest.fixture(
    params=[
        (BastankhahGaussianDeficit(use_effective_ws=True, use_radius_mask=True),),
        (NiayifarGaussianDeficit(use_effective_ws=True, use_radius_mask=True),),
    ]
)
def sim():
    sim = WakeSimulation(
        Turbine(
            rotor_diameter=126.0,
            hub_height=80.0,
            power_curve=Curve(
                ws=jnp.array([4.0, 25.0]), values=jnp.array([0.0, 5000.0])
            ),
            ct_curve=Curve(ws=jnp.array([4.0, 25.0]), values=jnp.array([0.8, 0.1])),
        ),
        BastankhahGaussianDeficit(use_effective_ws=True, use_radius_mask=True),
    )
    return sim


def test_flow_map(sim):
    """Test that the flow map is correctly generated and can be plotted."""
    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])
    grid_density = 100
    grid_x, grid_y = np.mgrid[
        -100 : 2000 : grid_density * 1j, -250 : 750 : grid_density * 1j
    ]

    flow_map, _ = sim.flow_map(
        xs,
        ys,
        fm_x=grid_x.ravel(),
        fm_y=grid_y.ravel(),
    )

    assert flow_map is not None
    assert flow_map.shape == (1, grid_density**2)

    plot_flow_map(grid_x, grid_y, flow_map, xs, ys, show=False)


def test_flow_map_ws_wd_args(sim):
    """Test that the flow map is correctly generated when ws and wd are provided."""
    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])
    grid_density = 100
    grid_x, grid_y = np.mgrid[
        -100 : 2000 : grid_density * 1j, -250 : 750 : grid_density * 1j
    ]

    flow_map, (fm_x, fm_y) = sim.flow_map(
        xs,
        ys,
        fm_x=grid_x.ravel(),
        fm_y=grid_y.ravel(),
        ws=[10.0, 12.0],
        wd=[270.0, 280.0],
    )

    assert flow_map is not None
    assert flow_map.shape == (2, grid_density**2)
    assert fm_x.shape == (grid_density**2,)
    assert fm_y.shape == (grid_density**2,)

    plot_flow_map(fm_x, fm_y, flow_map[0], xs, ys, show=False)
    plot_flow_map(fm_x, fm_y, flow_map[1], xs, ys, show=False)


def test_flow_map_no_grid(sim):
    """Test that the flow map is correctly generated when no grid is provided."""

    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])

    flow_map, (fm_x, fm_y) = sim.flow_map(
        xs,
        ys,
    )

    assert flow_map is not None
    assert flow_map.shape == (1, 200**2)

    plot_flow_map(fm_x, fm_y, flow_map[0], xs, ys, show=False)


def test_flow_map_no_grid_with_ti(sim):
    """Test that the flow map is correctly generated when no grid is provided."""

    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])
    ti = 0.1
    flow_map, (fm_x, fm_y) = sim.flow_map(xs, ys, ti=ti)
    assert flow_map is not None
    assert flow_map.shape == (1, 200**2)
    plot_flow_map(fm_x, fm_y, flow_map[0], xs, ys, show=False)


def test_flow_map_plotting_with_ax_passing(sim):
    """Test that the flow map can be plotted when an axes is provided."""
    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])
    grid_density = 100
    grid_x, grid_y = np.mgrid[
        -100 : 2000 : grid_density * 1j, -250 : 750 : grid_density * 1j
    ]

    flow_map, _ = sim.flow_map(
        xs,
        ys,
        fm_x=grid_x.ravel(),
        fm_y=grid_y.ravel(),
    )

    assert flow_map is not None
    assert flow_map.shape == (1, grid_density**2)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_flow_map(grid_x, grid_y, flow_map, xs, ys, show=False, ax=ax)
    plt.close(fig)


def test_flow_map_with_multiple_turbine_types():
    """Test that the flow map correctly handles multiple turbine types."""
    # Create two different turbine types
    turbine1 = Turbine(
        rotor_diameter=80.0,
        hub_height=70.0,
        power_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0, 2000])),
        ct_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0.8, 0.8])),
    )

    turbine2 = Turbine(
        rotor_diameter=120.0,
        hub_height=100.0,
        power_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0, 5000])),
        ct_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0.8, 0.8])),
    )

    turbine_library = [turbine1, turbine2]
    deficit_model = BastankhahGaussianDeficit(
        use_effective_ws=True, use_radius_mask=True
    )
    sim = WakeSimulation(turbine_library, deficit_model)

    # Layout with alternating turbine types
    xs = jnp.array([0.0, 500.0, 1000.0, 0.0, 500.0, 1000.0])
    ys = jnp.array([0.0, 0.0, 0.0, 500.0, 500.0, 500.0])
    wt_types = [
        turbine1.type_id,
        turbine2.type_id,
        turbine1.type_id,
        turbine2.type_id,
        turbine1.type_id,
        turbine2.type_id,
    ]

    # Generate flow map with multiple turbine types
    flow_map, (fm_x, fm_y) = sim.flow_map(xs, ys, ws=10.0, wd=270.0, wt_types=wt_types)

    # Basic checks
    assert flow_map is not None
    assert flow_map.shape == (1, 200**2)
    assert fm_x.shape == (200**2,)
    assert fm_y.shape == (200**2,)

    # Check that flow map values are reasonable
    assert jnp.all(flow_map >= 0)
    assert jnp.all(flow_map <= 10.0)

    # Test with custom grid
    grid_density = 50
    grid_x, grid_y = np.mgrid[
        -100 : 1200 : grid_density * 1j, -100 : 600 : grid_density * 1j
    ]

    flow_map2, (fm_x2, fm_y2) = sim.flow_map(
        xs,
        ys,
        fm_x=grid_x.ravel(),
        fm_y=grid_y.ravel(),
        ws=10.0,
        wd=270.0,
        wt_types=wt_types,
    )

    assert flow_map2 is not None
    assert flow_map2.shape == (1, grid_density**2)

    # Visualize to ensure it works
    plot_flow_map(grid_x, grid_y, flow_map2[0], xs, ys, show=False)
