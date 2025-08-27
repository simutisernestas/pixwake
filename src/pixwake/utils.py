import jax
import jax.numpy as jnp
from jax import config as jcfg


def get_eps() -> float:
    """Returns the machine epsilon for the current JAX float precision."""
    return (
        jax.numpy.finfo(jnp.float64).eps
        if jcfg.jax_enable_x64
        else jax.numpy.finfo(jnp.float32).eps
    )
