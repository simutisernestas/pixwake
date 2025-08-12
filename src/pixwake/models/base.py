from abc import ABC

import jax.numpy as jnp


class WakeModel(ABC):
    """An abstract base class for wake models."""

    def __init__(self):
        """Initializes the WakeModel."""
        pass

    def __call__(self, ws_eff, state):
        """A wrapper around the compute_deficit method.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            state: The state of the simulation.

        Returns:
            The updated effective wind speeds.
        """
        return self.compute_deficit(ws_eff, state)

    def compute_deficit(self, ws_eff, state):  # pragma: no cover
        """Computes the wake deficit.

        This method must be implemented by subclasses.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            state: The state of the simulation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        _ = (ws_eff, state)
        raise NotImplementedError

    def get_downwind_crosswind_distances(self, xs, ys, wd):
        """Calculates the downwind and crosswind distances between turbines.

        Args:
            xs: An array of x-coordinates for each turbine.
            ys: An array of y-coordinates for each turbine.
            wd: The wind direction.

        Returns:
            A tuple containing the downwind and crosswind distances.
        """
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
        cos_a = jnp.cos(wd_rad)
        sin_a = jnp.sin(wd_rad)
        x_d = -(dx * cos_a + dy * sin_a)
        y_d = dx * sin_a - dy * cos_a
        return x_d, y_d
