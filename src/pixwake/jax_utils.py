import jax
import jax.numpy as jnp
from jax import config as jcfg


def _is_64bit_enabled() -> bool:
    """Checks if JAX is configured to use 64-bit floating-point numbers.

    This function inspects the JAX configuration to determine if the
    `jax_enable_x64` flag has been set.

    Returns:
        `True` if JAX is using 64-bit floats, `False` otherwise.
    """
    return getattr(jcfg, "jax_enable_x64", False)


def get_float_eps() -> jnp.floating:
    """Returns the machine epsilon for the current JAX float precision.

    This function provides the smallest number such that `1.0 + eps != 1.0`
    for the current JAX floating-point configuration (either 32-bit or 64-bit).
    This is useful for handling numerical stability in calculations.

    Returns:
        The machine epsilon for the current JAX float type.
    """
    return (
        jnp.finfo(jnp.float64).eps
        if _is_64bit_enabled()
        else jnp.finfo(jnp.float32).eps
    )


def default_float_type() -> jnp.dtype:
    """Returns the default JAX floating-point data type.

    This function returns `jnp.float64` if JAX is configured for 64-bit
    precision, and `jnp.float32` otherwise.

    Returns:
        The default JAX float data type.
    """
    return jnp.float64 if _is_64bit_enabled() else jnp.float32


def ssqrt(x: jax.Array):
    """Gradient-stable square root function."""
    return jnp.sqrt(x + get_float_eps())
