from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Superposition:
    """Base class for turbulence superposition models."""

    @abstractmethod
    def __call__(self, ambient, added):
        """
        Calculates the effective turbulence intensity by combining the ambient
        and added turbulence.

        This method should be implemented by subclasses.

        Parameters
        ----------
        ti_amb : jax.numpy.ndarray
            Ambient turbulence intensity.
        ti_add : jax.numpy.ndarray
            Added turbulence intensity from wake effects.

        Returns
        -------
        jax.numpy.ndarray
            Effective turbulence intensity.
        """
        raise NotImplementedError


@dataclass
class SqrMaxSum(Superposition):
    """Square root of the sum of squares superposition model."""

    def __call__(self, ambient, added):
        """
        Calculates the effective turbulence intensity as the square root of the
        sum of the squares of the ambient and added turbulence.

        Parameters
        ----------
        ti_amb : jax.numpy.ndarray
            Ambient turbulence intensity.
        ti_add : jax.numpy.ndarray
            Added turbulence intensity from wake effects.

        Returns
        -------
        jax.numpy.ndarray
            Effective turbulence intensity, calculated as:
            sqrt(ti_amb^2 + ti_add^2)
        """
        return jnp.sqrt(ti_amb**2 + ti_add**2)
