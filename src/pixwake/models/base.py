import jax.numpy as jnp

from ..utils import geometry


class WakeModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.geometry = geometry

    def __call__(self, ws_eff, a):
        return self.compute_deficit(ws_eff, a)

    def compute_deficit(self, ws_eff, a):
        raise NotImplementedError
