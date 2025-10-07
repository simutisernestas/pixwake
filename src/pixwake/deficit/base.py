from abc import ABC, abstractmethod

import jax.numpy as jnp

from ..core import SimulationContext


class WakeDeficitModel(ABC):
    """An abstract base class for wake models."""

    def __init__(self) -> None:
        """Initializes the WakeDeficitModel."""
        pass

    def __call__(
        self,
        x: jnp.ndarray | tuple,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple:
        """A wrapper around the compute_deficit method.

        Args:
            x: An array of effective wind speeds at each turbine, or a tuple of
                (ws_eff, ti_eff).
            ctx: The context of the simulation.
            xs_r: An array of x-coordinates for each receiver point (optional).
            ys_r: An array of y-coordinates for each receiver point (optional).

        Returns:
            The updated effective wind speeds.
        """
        if isinstance(x, tuple):
            ws_eff, ti_eff = x
        else:
            ws_eff, ti_eff = x, None

        return self.compute_deficit(ws_eff, ctx, xs_r, ys_r, ti_eff=ti_eff)

    @abstractmethod
    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ctx: SimulationContext,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        ti_eff: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple:  # pragma: no cover
        """Computes the wake deficit.

        This method must be implemented by subclasses.

        Args:
            ws_eff: An array of effective wind speeds at each turbine.
            ctx: The context of the simulation.
            xs_r: An array of x-coordinates for each receiver point (optional).
            ys_r: An array of y-coordinates for each receiver point (optional).
            ti_eff: An array of effective turbulence intensity at each turbine (optional).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def get_downwind_crosswind_distances(
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