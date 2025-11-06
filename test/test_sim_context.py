import importlib

import jax.numpy as jnp
import pytest

from pixwake.core import Curve, SimulationContext, Turbine


@pytest.fixture
def turbine():
    """Create a standard turbine for testing."""
    power_curve = Curve(
        wind_speed=jnp.array([4.0, 10.0, 25.0]),
        values=jnp.array([0.0, 2000.0, 2000.0]),
    )
    ct_curve = Curve(
        wind_speed=jnp.array([4.0, 10.0, 25.0]),
        values=jnp.array([0.8, 0.8, 0.4]),
    )
    return Turbine(
        rotor_diameter=100.0,
        hub_height=80.0,
        power_curve=power_curve,
        ct_curve=ct_curve,
    )


@pytest.fixture
def simulation_context(turbine):
    """Create a simulation context for testing."""
    dw = jnp.array([[0.0, 500.0, 1000.0], [-500.0, 0.0, 500.0], [-1000.0, -500.0, 0.0]])
    cw = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ws = jnp.array(10.0)
    ti = jnp.array(0.06)
    return SimulationContext(turbine=turbine, dw=dw, cw=cw, ws=ws, ti=ti)


def get_all_deficit_models():
    """Discover deficit model classes from pixwake.deficit using its __all__."""
    mod = importlib.import_module("pixwake.deficit")
    names = getattr(mod, "__all__", None)
    assert names is not None

    models = []
    for name in names:
        obj = getattr(mod, name, None)
        assert obj is not None
        instance = obj()
        models.append(instance)

    return models


@pytest.mark.parametrize("deficit_model", get_all_deficit_models())
def test_wake_radius_set_after_deficit_calculation(simulation_context, deficit_model):
    """Test that wake_radius is set with correct shape after deficit calculation."""
    # Setup
    n_turbines = simulation_context.dw.shape[0]
    ws_eff = jnp.full(n_turbines, simulation_context.ws)
    ti_eff = (
        jnp.full(n_turbines, simulation_context.ti)
        if simulation_context.ti is not None
        else None
    )
    ws_result, ctx = deficit_model(ws_eff, ti_eff, simulation_context)

    assert isinstance(ctx, SimulationContext), (
        f"{deficit_model.__class__.__name__} did not return SimulationContext"
    )
    assert ctx.wake_radius is not None, (
        f"{deficit_model.__class__.__name__} did not set wake_radius in SimulationContext"
    )
    # Verify wake_radius has correct shape (n_turbines, n_turbines)
    expected_shape = (n_turbines, n_turbines)
    assert ctx.wake_radius.shape == expected_shape, (
        f"wake_radius shape {ctx.wake_radius.shape} does not match expected {expected_shape} "
        f"for {deficit_model.__class__.__name__}"
    )
    # Verify ws_result has correct shape
    assert ws_result.shape == (n_turbines,), (
        f"ws_result shape {ws_result.shape} does not match expected ({n_turbines},) "
        f"for {deficit_model.__class__.__name__}"
    )


def test_simulation_context_pytree_flatten_unflatten(simulation_context):
    """Test that SimulationContext can be flattened and unflattened correctly."""
    children, aux_data = simulation_context.tree_flatten()

    # Verify children contains dynamic data (JAX arrays)
    assert len(children) == 5
    assert jnp.array_equal(children[0], simulation_context.dw)
    assert jnp.array_equal(children[1], simulation_context.cw)
    assert jnp.array_equal(children[2], simulation_context.ws)
    assert jnp.array_equal(children[3], simulation_context.ti)
    assert children[4] is None

    # Verify aux_data contains static data (Turbine)
    assert len(aux_data) == 1
    assert aux_data[0] is simulation_context.turbine

    # Verify unflatten reconstructs the original context
    reconstructed = SimulationContext.tree_unflatten(aux_data, children)
    assert reconstructed.turbine is simulation_context.turbine
    assert jnp.array_equal(reconstructed.dw, simulation_context.dw)
    assert jnp.array_equal(reconstructed.cw, simulation_context.cw)
    assert jnp.array_equal(reconstructed.ws, simulation_context.ws)
    assert jnp.array_equal(reconstructed.ti, simulation_context.ti)
    assert reconstructed.wake_radius is None


def test_simulation_context_with_none_ti(turbine):
    """Test that SimulationContext handles None turbulence intensity correctly."""
    dw = jnp.array([[0.0, 500.0], [-500.0, 0.0]])
    cw = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    ws = jnp.array(10.0)

    ctx = SimulationContext(turbine=turbine, dw=dw, cw=cw, ws=ws, ti=None)
    assert ctx.ti is None

    # Test pytree operations with None ti
    children, aux_data = ctx.tree_flatten()
    assert children[3] is None

    reconstructed = SimulationContext.tree_unflatten(aux_data, children)
    assert reconstructed.ti is None
