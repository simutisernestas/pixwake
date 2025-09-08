from abc import ABC

import jax.numpy as jnp

from ..core import SimulationContext


class WakeDeficitModel(ABC):
    """An abstract base class for wake models."""

    def __init__(self) -> None:
        """Initializes the WakeDeficitModel."""
        pass

    def __call__(self, ws_eff: jnp.ndarray, ctx: SimulationContext) -> jnp.ndarray:
        """A wrapper around the compute_deficit method.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.

        Returns:
            The updated effective wind speeds.
        """
        return self.compute_deficit(ws_eff, ctx)

    def compute_deficit(
        self, ws_eff: jnp.ndarray, ctx: SimulationContext
    ) -> jnp.ndarray:  # pragma: no cover
        """Computes the wake deficit.

        This method must be implemented by subclasses.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        _ = (ws_eff, ctx)
        raise NotImplementedError

    def _get_downwind_crosswind_distances(
        self,
        xs_s: jnp.ndarray,
        ys_s: jnp.ndarray,
        xs_r: jnp.ndarray,
        ys_r: jnp.ndarray,
        wd: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates the downwind and crosswind distances between points.

        Args:
            xs_s: An array of x-coordinates for each source point.
            ys_s: An array of y-coordinates for each source point.
            xs_r: An array of x-coordinates for each receiver point.
            ys_r: An array of y-coordinates for each receiver point.
            wd: The wind direction.

        Returns:
            A tuple containing the downwind and crosswind distances.
        """
        dx = xs_r[:, None] - xs_s[None, :]
        dy = ys_r[:, None] - ys_s[None, :]
        wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
        cos_a = jnp.cos(wd_rad)
        sin_a = jnp.sin(wd_rad)
        x_d = -(dx * cos_a + dy * sin_a)
        y_d = dx * sin_a - dy * cos_a
        return x_d, y_d

    def get_downwind_crosswind_distances(
        self, xs: jnp.ndarray, ys: jnp.ndarray, wd: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates the downwind and crosswind distances between turbines.

        Args:
            xs: An array of x-coordinates for each turbine.
            ys: An array of y-coordinates for each turbine.
            wd: The wind direction.

        Returns:
            A tuple containing the downwind and crosswind distances.
        """
        return self._get_downwind_crosswind_distances(xs, ys, xs, ys, wd)

    def flow_map(
        self, ws_eff: jnp.ndarray, ctx: SimulationContext
    ) -> jnp.ndarray:  # pragma: no cover
        """Computes the wind speed on a grid.

        This method must be implemented by subclasses.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        _ = (ws_eff, ctx)
        raise NotImplementedError
