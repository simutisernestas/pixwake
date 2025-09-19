import matplotlib.pyplot as plt
import jax.numpy as jnp


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
