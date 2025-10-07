from abc import abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp

from pixwake.core import SimulationContext


class Superposition:
    """Base class for superposition models."""

    @abstractmethod
    def __call__(self, ambient: jnp.ndarray, added: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the effective quantity by combining the ambient and added.

        This method should be implemented by subclasses.

        Parameters
        ----------
        ambient : jax.numpy.ndarray
            Ambient quantity.
        added : jax.numpy.ndarray
            Added quantity from wake effects.

        Returns
        -------
        jax.numpy.ndarray
            Effective quantity.
        """
        raise NotImplementedError


class SqrMaxSum(Superposition):
    """Square root of the sum of squares superposition model."""

    def __call__(self, ambient: jnp.ndarray, added: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the effective quantity as the square root of the
        sum of the squares of the ambient and added.

        Parameters
        ----------
        ambient : jax.numpy.ndarray
            Ambient quantity.
        added : jax.numpy.ndarray
            Added quantity from wake effects. Shape (n_receivers, n_sources).

        Returns
        -------
        jax.numpy.ndarray
            Effective quantity for each receiver.
        """
        # Take the max contribution from all sources (axis=1) for each receiver
        return jnp.sqrt(ambient**2 + jnp.max(added, axis=1) ** 2)


@dataclass
class TurbulenceModel:
    """Base class for turbulence models."""

    superposition_model: Superposition = field(default_factory=SqrMaxSum)

    @abstractmethod
    def calc_added_turbulence(
        self,
        ctx: SimulationContext,
        ws_eff: jnp.ndarray,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        ti_eff: jnp.ndarray,
        wake_radius: jnp.ndarray,
        ct: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculates the added turbulence intensity (TI).

        This method should be implemented by subclasses to define the specific
        turbulence model logic.

        Parameters
        ----------
        ctx : SimulationContext
            The simulation context, containing turbine and wind condition data.
        ws_eff : jnp.ndarray
            The effective wind speed at each turbine.
        dw : jnp.ndarray
            The downwind distance between all pairs of turbines.
        cw : jnp.ndarray
            The crosswind distance between all pairs of turbines.
        ti_eff : jnp.ndarray
            The effective turbulence intensity at each source turbine.
        wake_radius : jnp.ndarray
            The wake radius for each turbine pair.
        ct : jnp.ndarray
            The thrust coefficient for each source turbine.

        Returns
        -------
        jnp.ndarray
            An array representing the added turbulence intensity at each
            turbine from each other turbine.
        """
        raise NotImplementedError