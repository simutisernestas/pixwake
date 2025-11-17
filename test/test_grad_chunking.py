import time
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TOLS
from jax.test_util import check_grads

from pixwake import WakeSimulation
from pixwake.deficit import NiayifarGaussianDeficit
from pixwake.definitions import vestas_v80
from pixwake.turbulence import CrespoHernandez


def generate_turbine_layout(sqrt_n_turbines=3, spacing_D=3, rotor_diameter=120.0):
    """Generates a simple turbine layout."""
    x, y = np.meshgrid(
        np.arange(sqrt_n_turbines) * spacing_D * rotor_diameter,
        np.arange(sqrt_n_turbines) * spacing_D * rotor_diameter,
    )
    x = x.flatten()
    y = y.flatten()
    return jnp.array(x), jnp.array(y)


def generate_time_series_wind_data(n_hours=50):
    """Generates sample time-series wind data."""
    key = jax.random.PRNGKey(0)
    ws_key, wd_key = jax.random.split(key)
    ws = 8.0 + 2.0 * jax.random.normal(ws_key, (n_hours,))
    wd = 360 * jax.random.uniform(wd_key, (n_hours,))
    return ws, wd


@pytest.fixture(scope="module")
def simulation_setup() -> tuple[WakeSimulation, ...]:
    """Provides a configured WakeSimulation and test data."""

    # Simulation setup
    deficit_model = NiayifarGaussianDeficit(
        use_effective_ws=True, use_effective_ti=True
    )
    turbulence_model = CrespoHernandez()
    sim = WakeSimulation(
        vestas_v80,
        deficit_model,
        turbulence=turbulence_model,
        mapping_strategy="vmap",
    )

    # Data setup
    wt_xs, wt_ys = generate_turbine_layout(rotor_diameter=vestas_v80.rotor_diameter)
    ws_amb, wd_amb = generate_time_series_wind_data()
    ti = 0.1
    return sim, wt_xs, wt_ys, ws_amb, wd_amb, ti


@pytest.mark.parametrize("chunk_size", [1, 25, 37, 50, 1000, -1])
def test_chunked_gradients_match(simulation_setup, chunk_size):
    """Tests that the chunked gradient calculation produces
    the same result as the non-chunked (standard) gradient calculation.
    """
    sim, wt_xs, wt_ys, ws_amb, wd_amb, ti = simulation_setup

    # Chunked gradient calculation
    ctx = pytest.raises(AssertionError) if chunk_size <= 0 else nullcontext()
    with ctx:
        aep_chunked, grad_chunked = sim.aep_gradients_chunked(
            wt_xs, wt_ys, ws_amb, wd_amb, ti_amb=ti, chunk_size=chunk_size
        )
        grad_x_chunked, grad_y_chunked = grad_chunked
    if "aep_chunked" not in locals():
        return

    # Standard gradient calculation
    @jax.jit
    def aep_fn(x, y):
        result = sim(x, y, ws_amb, wd_amb, ti)
        return result.aep()

    aep_standard, grad_standard = jax.value_and_grad(aep_fn, argnums=(0, 1))(
        wt_xs, wt_ys
    )
    grad_x_standard, grad_y_standard = grad_standard

    # Compare results
    assert jnp.allclose(aep_standard, aep_chunked, **TOLS)
    assert jnp.allclose(grad_x_standard, grad_x_chunked, **TOLS)
    assert jnp.allclose(grad_y_standard, grad_y_chunked, **TOLS)


def test_gradient_chunked_is_faster_after_warmup_call(simulation_setup):
    sim, wt_xs, wt_ys, ws_amb, wd_amb, ti = simulation_setup

    # Warm-up call for JIT compilation
    start_chunked = time.time()
    aep, (dx, dy) = sim.aep_gradients_chunked(
        wt_xs, wt_ys, ws_amb, wd_amb, ti_amb=ti, chunk_size=51
    )
    aep.block_until_ready()
    dx.block_until_ready()
    dy.block_until_ready()
    end_chunked = time.time()
    warmup_time = end_chunked - start_chunked

    # Measure chunked gradient time
    start_chunked = time.time()
    aep, (dx, dy) = sim.aep_gradients_chunked(
        wt_xs, wt_ys, ws_amb, wd_amb, ti_amb=ti, chunk_size=32
    )
    aep.block_until_ready()
    dx.block_until_ready()
    dy.block_until_ready()
    end_chunked = time.time()
    chunked_time = end_chunked - start_chunked

    assert chunked_time < (warmup_time * 100), (
        "Chunked gradient calculation is not faster than warmup."
    )


def test_chunked_gradients_with_probabilities(simulation_setup):
    """Test that chunked gradients work correctly with probability weights."""
    sim, wt_xs, wt_ys, ws_amb, wd_amb, ti = simulation_setup

    # Create non-uniform probabilities
    probabilities = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (len(ws_amb),)))
    probabilities = probabilities / probabilities.sum()

    # Chunked calculation
    aep_chunked, (grad_x_chunked, grad_y_chunked) = sim.aep_gradients_chunked(
        wt_xs,
        wt_ys,
        ws_amb,
        wd_amb,
        ti_amb=ti,
        chunk_size=50,
        probabilities=probabilities,
    )

    # Standard calculation
    @jax.jit
    def aep_fn(x, y):
        result = sim(x, y, ws_amb, wd_amb, ti)
        return result.aep(probabilities=probabilities)

    aep_standard, (grad_x_standard, grad_y_standard) = jax.value_and_grad(
        aep_fn, argnums=(0, 1)
    )(wt_xs, wt_ys)

    assert jnp.allclose(aep_standard, aep_chunked, **TOLS)
    assert jnp.allclose(grad_x_standard, grad_x_chunked, **TOLS)
    assert jnp.allclose(grad_y_standard, grad_y_chunked, **TOLS)


def test_chunked_gradients_single_timestamp(simulation_setup):
    """Test chunked gradients with only a single timestamp."""
    sim, wt_xs, wt_ys, _, _, ti = simulation_setup

    # Use only first timestamp
    ws_single = 10.0 + jnp.zeros((1,))
    wd_single = 270.0 + jnp.zeros((1,))

    aep_chunked, (grad_x, grad_y) = sim.aep_gradients_chunked(
        wt_xs, wt_ys, ws_single, wd_single, ti_amb=ti, chunk_size=10
    )

    # Standard calculation
    @jax.jit
    def aep_fn(x, y):
        result = sim(x, y, ws_single, wd_single, ti)
        return result.aep()

    check_grads(
        aep_fn, (wt_xs, wt_ys), order=1, modes=["rev"], atol=1e-3, rtol=1e-2, eps=1e-3
    )

    aep_standard, (grad_x_standard, grad_y_standard) = jax.value_and_grad(
        aep_fn, argnums=(0, 1)
    )(wt_xs, wt_ys)

    assert jnp.allclose(aep_standard, aep_chunked, **TOLS)
    assert jnp.allclose(grad_x_standard, grad_x, **TOLS)
    assert jnp.allclose(grad_y_standard, grad_y, **TOLS)


def test_chunked_gradients_do_not_cache_wind_resource(simulation_setup):
    """Test that chunked gradients respond to changes in wind resource"""
    sim, wt_xs, wt_ys, ws_amb, wd_amb, ti = simulation_setup

    aep_chunked0, (grad_x0, grad_y0) = sim.aep_gradients_chunked(
        wt_xs, wt_ys, ws_amb, wd_amb, ti_amb=ti, chunk_size=10
    )

    ws_single = ws_amb + 5
    wd_single = (wd_amb - 45) % 360
    aep_chunked1, (grad_x1, grad_y1) = sim.aep_gradients_chunked(
        wt_xs, wt_ys, ws_single, wd_single, ti_amb=ti, chunk_size=10
    )

    # basically should observe at least 1% change
    assert not jnp.allclose(aep_chunked0, aep_chunked1, rtol=1e-2)
    assert not jnp.allclose(grad_x0, grad_x1, **TOLS)
    assert not jnp.allclose(grad_y0, grad_y1, **TOLS)
