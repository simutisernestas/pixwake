import pytest
from jax import config as jcfg


@pytest.fixture
def float32_config():
    """A pytest fixture to temporarily set JAX to use 32-bit floats."""
    original_flag = jcfg.x64_enabled
    jcfg.update("jax_enable_x64", False)
    yield
    jcfg.update("jax_enable_x64", original_flag)