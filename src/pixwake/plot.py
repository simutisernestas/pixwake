import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure


def plot_flow_map(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    flow_map_data: jnp.ndarray,
    wt_x: jnp.ndarray | None = None,
    wt_y: jnp.ndarray | None = None,
    show: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plots a wind farm flow map.
    Args:
        grid_x: The x-coordinates of the grid.
        grid_y: The y-coordinates of the grid.
        flow_map_data: The wind speed data for the flow map.
        wt_x: The x-coordinates of the turbines (optional).
        wt_y: The y-coordinates of the turbines (optional).
        show: If True, calls `plt.show()`. Only has an effect if `ax` is None.
        ax: An optional matplotlib axes to plot on. If not provided, a new
            figure and axes are created.
    Returns:
        The axes on which the flow map was plotted.
    """
    if grid_x.ndim != 2 or grid_y.ndim != 2:
        side_length = int(jnp.sqrt(grid_x.shape[0]))
        assert side_length * side_length == grid_x.shape[0], (
            "assumption grid_x being a perfect square does not hold"
        )
        grid_x = grid_x.reshape((side_length, side_length))
        grid_y = grid_y.reshape((side_length, side_length))

    fig: Figure | SubFigure | None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        show_plot = show
    else:
        fig = ax.get_figure()
        show_plot = False

    contour = ax.contourf(
        grid_x,
        grid_y,
        flow_map_data.reshape(grid_x.shape),
        cmap="viridis",
        levels=100,
    )
    if fig is not None:
        fig.colorbar(contour, ax=ax, label="Wind Speed (m/s)")

    if wt_x is not None and wt_y is not None:
        ax.scatter(wt_x, wt_y, color="red", marker="^", s=50, label="Turbines")
        ax.legend()

    ax.set_title("Wind Farm Flow Map")
    ax.set_xlabel("x-coordinates")
    ax.set_ylabel("y-coordinates")

    if show_plot:
        plt.show()

    return ax
