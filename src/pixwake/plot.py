import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from pixwake import Curve, Turbine


def plot_flow_map(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    flow_map_data: jnp.ndarray,
    wt_x: jnp.ndarray | None = None,
    wt_y: jnp.ndarray | None = None,
    show: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plots a 2D flow map of a wind farm.

    This function generates a contour plot of the wind speed over a specified
    grid, and can optionally overlay the locations of the wind turbines.

    Args:
        grid_x: A JAX numpy array of the x-coordinates of the grid.
        grid_y: A JAX numpy array of the y-coordinates of the grid.
        flow_map_data: A JAX numpy array of the wind speed at each grid point.
        wt_x: An optional JAX numpy array of the x-coordinates of the turbines.
        wt_y: An optional JAX numpy array of the y-coordinates of the turbines.
        show: If `True`, the plot is displayed. This is only active when `ax`
            is not provided.
        ax: An optional Matplotlib `Axes` object to plot on. If not provided, a
            new figure and axes are created.

    Returns:
        The Matplotlib `Axes` object on which the flow map was plotted.
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
    ax.set_aspect("equal", adjustable="box")

    if show_plot:
        plt.show()

    return ax


def plot_power_and_thrust_curve(turbine: Turbine, show=True) -> None:
    """
    Plots the power and thrust coefficient (Ct) curves for a given Turbine object.

    This function creates a figure with two subplots:
    1. The top plot shows the power curve (Power vs. Wind Speed).
    2. The bottom plot shows the thrust coefficient curve (Ct vs. Wind Speed).

    Args:
        turbine: A pixwake Turbine object containing the power and ct curves.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # --- Power Curve Plot ---
    ax1.plot(
        turbine.power_curve.wind_speed,
        turbine.power_curve.values,
        "o-",
        color="royalblue",
        label="Power",
    )
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Power Curve")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # --- Thrust Coefficient Curve Plot ---
    ax2.plot(
        turbine.ct_curve.wind_speed,
        turbine.ct_curve.values,
        "o-",
        color="seagreen",
        label="Thrust Coefficient",
    )
    ax2.set_ylabel("Thrust Coefficient (Ct)")
    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.set_title("Thrust Curve")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    # --- Overall Figure Formatting ---
    fig.suptitle(
        f"Turbine Performance Curves\n(Rotor Diameter: {turbine.rotor_diameter}m, Hub Height: {turbine.hub_height}m)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    if show:
        plt.show()
    else:
        plt.savefig("power_curve")
