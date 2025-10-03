from abc import abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp

from pixwake.core import SimulationContext


class Superposition:
    """Base class for superposition models."""

    @abstractmethod
    def __call__(self, ambient, added):
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

    def __call__(self, ambient, added):
        """
        Calculates the effective quantity as the square root of the
        sum of the squares of the ambient and added.

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
        return jnp.sqrt(ambient**2 + jnp.max(added, 0) ** 2)


@dataclass
class TurbulenceModel:
    """Base class for turbulence models."""

    superposition_model: Superposition = field(default_factory=SqrMaxSum)

    def __call__(
        self,
        ctx: SimulationContext,
        ws_eff: jnp.ndarray,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        ti_eff: jnp.ndarray,
        ti_amb: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculates the total turbulence by combining ambient and added turbulence.

        Parameters
        ----------
        ctx : SimulationContext
            The simulation context.
        ws_eff : jnp.ndarray
            The effective wind speed at each turbine.
        dw : jnp.ndarray
            The downwind distance between all pairs of turbines.
        cw : jnp.ndarray
            The crosswind distance between all pairs of turbines.
        ti_eff : jnp.ndarray
            The effective turbulence intensity at each source turbine.
        ti_amb : jnp.ndarray
            The ambient turbulence intensity.

        Returns
        -------
        jnp.ndarray
            The total turbulence intensity at each turbine.
        """
        ti_add = self.calc_added_turbulence(ctx, ws_eff, dw, cw, ti_eff)
        return self.superposition_model(ti_amb, ti_add)

    @abstractmethod
    def calc_added_turbulence(
        self,
        ctx: SimulationContext,
        ws_eff: jnp.ndarray,
        dw: jnp.ndarray,
        cw: jnp.ndarray,
        ti_eff: jnp.ndarray,
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

        Returns
        -------
        jnp.ndarray
            An array representing the added turbulence intensity at each
            turbine from each other turbine.
        """
        raise NotImplementedError
