import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pixwake.core import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit


def test_flow_map():
    """Test that the flow map is correctly generated and can be plotted."""
    sim = WakeSimulation(
        BastankhahGaussianDeficit(use_effective_ws=True, use_radius_mask=True),
        turbine=Turbine(
            rotor_diameter=126.0,
            hub_height=80.0,
            power_curve=Curve(
                wind_speed=jnp.array([4.0, 25.0]), values=jnp.array([0.0, 5000.0])
            ),
            ct_curve=Curve(
                wind_speed=jnp.array([4.0, 25.0]), values=jnp.array([0.8, 0.1])
            ),
        ),
    )
    xs = jnp.array([0, 500, 0, 1000])
    ys = jnp.array([0, 0, 500, 0])
    grid_density = 100
    grid_x, grid_y = np.mgrid[
        -100 : 2000 : grid_density * 1j, -250 : 750 : grid_density * 1j
    ]

    flow_map = sim.flow_map(
        xs,
        ys,
        fm_x=grid_x.ravel(),
        fm_y=grid_y.ravel(),
    )

    # TODO: test ws,wd arguments !
    # TODO: test without grid arguments !

    assert flow_map is not None
    assert flow_map.shape == (1, grid_density**2)

    # TODO: move inside the package!
    plt.figure(figsize=(10, 8))
    plt.contourf(
        grid_x, grid_y, flow_map.reshape(grid_x.shape), cmap="viridis", levels=100
    )
    plt.colorbar(label="Wind Speed (m/s)")
    plt.scatter(xs, ys, color="red", marker="^", s=50, label="Turbines")
    plt.title("Wind Farm Flow Map")
    plt.xlabel("x-coordinates")
    plt.ylabel("y-coordinates")
    plt.legend()
    plt.show()
