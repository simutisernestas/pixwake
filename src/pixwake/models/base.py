from abc import ABC

import jax.numpy as jnp


class WakeModel(ABC):
    def __init__(self, **kwargs):
        # TODO: should be explicit about model arguments !!!
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, ws_eff, a):
        return self.compute_deficit(ws_eff, a)

    def compute_deficit(self, ws_eff, a):  # pragma: no cover
        _ = (ws_eff, a)
        raise NotImplementedError

    def get_downwind_crosswind_distances(self, xs, ys, wd):
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
        cos_a = jnp.cos(wd_rad)
        sin_a = jnp.sin(wd_rad)
        x_d = -(dx * cos_a + dy * sin_a)
        y_d = dx * sin_a - dy * cos_a
        return x_d, y_d
