import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from pixwake import Curve, Turbine
from pixwake.plot import plot_power_and_thrust_curve


@pytest.fixture
def sample_turbine() -> Turbine:
    """Create a sample turbine for testing."""
    wind_speeds = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    power_values = jnp.array([0.0, 100.0, 500.0, 1000.0, 1200.0, 0.0])
    ct_values = jnp.array([0.0, 0.8, 0.75, 0.5, 0.3, 0.0])

    turbine = Turbine(
        rotor_diameter=120.0,
        hub_height=100.0,
        power_curve=Curve(ws=wind_speeds, values=power_values),
        ct_curve=Curve(ws=wind_speeds, values=ct_values),
    )
    return turbine


def test_plot_power_and_thrust_curve_creates_figure(sample_turbine):
    """Test that the function creates a figure with two subplots."""
    # Close any existing figures
    plt.close("all")

    # Call the function without showing
    plot_power_and_thrust_curve(sample_turbine, show=False)

    # Get the current figure
    fig = plt.gcf()

    # Check that a figure was created
    assert fig is not None

    # Check that there are 2 subplots
    axes = fig.get_axes()
    assert len(axes) == 2

    plt.close("all")


def test_plot_power_and_thrust_curve_plot_contents(sample_turbine):
    """Test that the plots contain the expected data."""
    plt.close("all")

    plot_power_and_thrust_curve(sample_turbine, show=False)

    fig = plt.gcf()
    axes = fig.get_axes()

    # Check power curve plot (first subplot)
    power_ax = axes[0]
    power_lines = power_ax.get_lines()
    assert len(power_lines) == 1

    # Verify power curve data
    power_line = power_lines[0]
    x_data = power_line.get_xdata()
    y_data = power_line.get_ydata()

    assert jnp.allclose(x_data, sample_turbine.power_curve.ws)
    assert jnp.allclose(y_data, sample_turbine.power_curve.values)

    # Check thrust coefficient plot (second subplot)
    ct_ax = axes[1]
    ct_lines = ct_ax.get_lines()
    assert len(ct_lines) == 1

    # Verify ct curve data
    ct_line = ct_lines[0]
    x_data = ct_line.get_xdata()
    y_data = ct_line.get_ydata()

    assert jnp.allclose(x_data, sample_turbine.ct_curve.ws)
    assert jnp.allclose(y_data, sample_turbine.ct_curve.values)

    plt.close("all")


def test_plot_power_and_thrust_curve_labels_and_titles(sample_turbine):
    """Test that the plots have correct labels and titles."""
    plt.close("all")

    plot_power_and_thrust_curve(sample_turbine, show=False)

    fig = plt.gcf()
    axes = fig.get_axes()

    # Check power plot labels
    power_ax = axes[0]
    assert power_ax.get_ylabel() == "Power (kW)"
    assert power_ax.get_title() == "Power Curve"

    # Check ct plot labels
    ct_ax = axes[1]
    assert ct_ax.get_ylabel() == "Thrust Coefficient (Ct)"
    assert ct_ax.get_xlabel() == "Wind Speed (m/s)"
    assert ct_ax.get_title() == "Thrust Curve"

    # Check overall figure title contains turbine info
    suptitle = fig._suptitle
    assert suptitle is not None
    assert "120" in suptitle.get_text()  # Rotor diameter
    assert "100" in suptitle.get_text()  # Hub height

    plt.close("all")


def test_plot_power_and_thrust_curve_show_parameter(sample_turbine):
    """Test that the show parameter works correctly."""
    plt.close("all")

    # Test with show=False (should not raise any errors)
    plot_power_and_thrust_curve(sample_turbine, show=False)

    # Verify figure exists but is not shown
    assert len(plt.get_fignums()) == 1

    plt.close("all")
