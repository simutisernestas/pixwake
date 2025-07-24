import jax
import jax.numpy as jnp
from jax import config as jcfg


def get_eps():
    return (
        jax.numpy.finfo(jnp.float64).eps
        if jcfg.jax_enable_x64
        else jax.numpy.finfo(jnp.float32).eps
    )


def geometry(xs, ys, wd):
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
    cos_a = jnp.cos(wd_rad)
    sin_a = jnp.sin(wd_rad)
    x_d = -(dx * cos_a + dy * sin_a)
    y_d = dx * sin_a - dy * cos_a
    return x_d, y_d
