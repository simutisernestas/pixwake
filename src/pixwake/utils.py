import jax.numpy as jnp
from jax import config as jcfg
from numpy import floating


def get_eps() -> floating:
    """Returns the machine epsilon for the current JAX float precision."""
    return (
        jnp.finfo(jnp.float64).eps
        if getattr(jcfg, "jax_enable_x64", False)
        else jnp.finfo(jnp.float32).eps
    )
