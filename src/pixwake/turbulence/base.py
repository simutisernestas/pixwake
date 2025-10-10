from abc import abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp

from pixwake.core import SimulationContext


class Superposition:
    """Base class for combining ambient and wake-added quantities."""

    @abstractmethod
    def __call__(self, ambient: jnp.ndarray, added: jnp.ndarray) -> jnp.ndarray:
        """Combine ambient and added quantities into effective values.

        Args:
            ambient: Ambient quantity at each location (n_locations,).
            added: Added quantity from wake effects (n_receivers, n_sources).

        Returns:
            Effective quantity at each receiver (n_receivers,).
        """
        raise NotImplementedError


class SqrMaxSum(Superposition):
    """Square-root-of-sum-of-squares superposition using maximum contribution.

    Takes the maximum added contribution from all sources and combines it
    with the ambient value: sqrt(ambient^2 + max(added)^2).
    """

    def __call__(self, ambient: jnp.ndarray, added: jnp.ndarray) -> jnp.ndarray:
        """Combine ambient and added quantities.

        Args:
            ambient: Ambient quantity (n_receivers,).
            added: Added quantity from wakes (n_receivers, n_sources).

        Returns:
            Effective quantity (n_receivers,).
        """
        max_added = jnp.max(added, axis=1)
        return jnp.sqrt(ambient**2 + max_added**2)


@dataclass
class WakeTurbulence:
    """Base class for wake-added turbulence models."""

    superposition: Superposition = field(default_factory=SqrMaxSum)

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
        wake_radius: jnp.ndarray,
    ) -> jnp.ndarray:
        ti_added_m = self.added_turbulence(ws_eff, ti_eff, ctx)
        inside_wake = (ctx.dw > 0.0) & (jnp.abs(ctx.cw) < wake_radius)
        ti_added_m = jnp.where(inside_wake, ti_added_m, 0.0)

        # Combine ambient and added turbulence
        ti_ambient = jnp.full_like(ti_eff, ctx.ti)
        return self.superposition(ti_ambient, ti_added_m)

    @abstractmethod
    def added_turbulence(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Calculate wake-added turbulence intensity.

        Args:
            ctx: Simulation context with turbine and wind data.
            ws_eff: Effective wind speed at each source turbine (n_sources,).
            dw: Downwind distances (n_receivers, n_sources).
            cw: Crosswind distances (n_receivers, n_sources).
            ti_eff: Effective TI at each source turbine (n_sources,).
            wake_radius: Wake radius for each receiver-source pair (n_receivers, n_sources).
            ct: Thrust coefficient at each source turbine (n_sources,).

        Returns:
            Added turbulence intensity matrix (n_receivers, n_sources).
        """
        raise NotImplementedError
