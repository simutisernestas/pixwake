from abc import ABC, abstractmethod

import jax.numpy as jnp

from pixwake.jax_utils import get_float_eps

from ..core import SimulationContext


class WakeDeficit(ABC):
    """An abstract base class for wake models."""

    def __init__(self, use_radius_mask: bool = True) -> None:
        """Initializes the WakeDeficitModel."""
        self.use_radius_mask = use_radius_mask

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        # all2all deficit matrix (n_receivers, n_sources)
        ws_deficit_m, wake_radius = self.compute(ws_eff, ti_eff, ctx)

        in_wake_mask = ctx.dw > 0.0
        if self.use_radius_mask:  # TODO: pywake doesn't do this. Why ?
            in_wake_mask &= jnp.abs(ctx.cw) < wake_radius
        ws_deficit_m = jnp.where(in_wake_mask, ws_deficit_m**2, 0.0)

        # superpose deficits in quadrature
        ws_deficit = jnp.sqrt(jnp.sum(ws_deficit_m, axis=1) + get_float_eps())
        return jnp.maximum(0.0, ctx.ws - ws_deficit), wake_radius

    @abstractmethod
    def compute(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:  # pragma: no cover
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
