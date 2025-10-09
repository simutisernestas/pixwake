from abc import ABC, abstractmethod

import jax.numpy as jnp

from pixwake.jax_utils import get_float_eps

from ..core import SimulationContext


class WakeDeficit(ABC):
    """An abstract base class for wake models."""

    def __init__(self) -> None:
        """Initializes the WakeDeficitModel."""
        pass

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
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
        ws_deficit = self.compute_deficit(ws_eff, ti_eff, ctx)
        ws_deficit = jnp.sqrt(jnp.sum(ws_deficit**2, axis=1) + get_float_eps())
        return jnp.maximum(0.0, ctx.ws - ws_deficit)

    @abstractmethod
    def compute_deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
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
