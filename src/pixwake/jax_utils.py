import jax.numpy as jnp
from jax import config as jcfg


def _is_64bit_enabled() -> bool:
    """Returns True if JAX is configured to use 64-bit floats, False otherwise."""
    return getattr(jcfg, "jax_enable_x64", False)


def get_float_eps() -> jnp.floating:
    """Returns the machine epsilon for the current JAX float precision."""
    return (
        jnp.finfo(jnp.float64).eps
        if _is_64bit_enabled()
        else jnp.finfo(jnp.float32).eps
    )


def default_float_type() -> jnp.dtype:
    """Returns the default JAX float type."""
    return jnp.float64 if _is_64bit_enabled() else jnp.float32
