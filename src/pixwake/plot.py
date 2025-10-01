import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_flow_map(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    flow_map_data: jnp.ndarray,
    wt_x: jnp.ndarray | None = None,
    wt_y: jnp.ndarray | None = None,
) -> None:
    """Plots a wind farm flow map.

    Args:
        grid_x: The x-coordinates of the grid.
        grid_y: The y-coordinates of the grid.
        flow_map_data: The wind speed data for the flow map.
        wt_x: The x-coordinates of the turbines (optional).
        wt_y: The y-coordinates of the turbines (optional).
    """
    if grid_x.ndim != 2 or grid_y.ndim != 2:
        side_length = int(jnp.sqrt(grid_x.shape[0]))
        assert side_length * side_length == grid_x.shape[0], (
            "assumption grid_x being a perfect square does not hold"
        )
        grid_x = grid_x.reshape((side_length, side_length))
        grid_y = grid_y.reshape((side_length, side_length))

    plt.figure(figsize=(10, 8))
    plt.contourf(
        grid_x, grid_y, flow_map_data.reshape(grid_x.shape), cmap="viridis", levels=100
    )
    plt.colorbar(label="Wind Speed (m/s)")

    if wt_x is not None and wt_y is not None:
        plt.scatter(wt_x, wt_y, color="red", marker="^", s=50, label="Turbines")
        plt.legend()

    plt.title("Wind Farm Flow Map")
    plt.xlabel("x-coordinates")
    plt.ylabel("y-coordinates")
    plt.show()
