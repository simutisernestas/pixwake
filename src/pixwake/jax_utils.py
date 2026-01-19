"""JAX utility functions for numerical precision and stability.

This module provides helper functions and constants for working with JAX arrays:
    - default_float_type: Get current JAX float precision (float32/float64)
    - get_float_eps: Get machine epsilon for current precision
    - ssqrt: Gradient-stable square root function

Numerical constants for stability:
    - GRAD_SAFE_MIN: Minimum value for gradient-safe operations (1e-20)
    - NUMERICAL_FLOOR: Floor value for power law calculations (1e-10)
"""

import jax
import jax.numpy as jnp
from jax import config as jcfg

# ============================================================================
# Machine Epsilon Constants
# ============================================================================
# Pre-computed epsilon values to avoid repeated jnp.finfo() calls
_EPS_FLOAT32: float = float(jnp.finfo(jnp.float32).eps)
_EPS_FLOAT64: float = float(jnp.finfo(jnp.float64).eps)

# ============================================================================
# Numerical Stability Constants
# ============================================================================
# Minimum value for gradient-safe operations to avoid numerical issues
# (e.g., in sqrt, division). Use when computing gradients near zero.
GRAD_SAFE_MIN: float = 1e-20

# Floor value for numerical stability in power law and similar calculations.
# Slightly larger than GRAD_SAFE_MIN for non-gradient contexts.
NUMERICAL_FLOOR: float = 1e-10


def _is_64bit_enabled() -> bool:
    """Checks if JAX is configured to use 64-bit floating-point numbers.

    This function inspects the JAX configuration to determine if the
    `jax_enable_x64` flag has been set.

    Returns:
        `True` if JAX is using 64-bit floats, `False` otherwise.
    """
    return getattr(jcfg, "jax_enable_x64", False)


def get_float_eps() -> float:
    """Returns the machine epsilon for the current JAX float precision.

    This function provides the smallest number such that `1.0 + eps != 1.0`
    for the current JAX floating-point configuration (either 32-bit or 64-bit).
    This is useful for handling numerical stability in calculations.

    Returns:
        The machine epsilon for the current JAX float type.
    """
    return _EPS_FLOAT64 if _is_64bit_enabled() else _EPS_FLOAT32


def default_float_type() -> jnp.dtype:
    """Returns the default JAX floating-point data type.

    This function returns `jnp.float64` if JAX is configured for 64-bit
    precision, and `jnp.float32` otherwise.

    Returns:
        The default JAX float data type.
    """
    return jnp.float64 if _is_64bit_enabled() else jnp.float32


def ssqrt(x: jax.Array) -> jax.Array:
    """Gradient-stable square root function.

    Uses jnp.maximum instead of addition for better numerical stability
    with negative inputs (which can occur due to floating-point errors).
    """
    return jnp.sqrt(jnp.maximum(x, get_float_eps()))
