"""Superposition models for combining wake effects from multiple turbines.

This module provides different strategies for combining wake deficits:
    - SquaredSum: sqrt(sum(deficit²)) - default for wake deficits
    - SqrMaxSum: sqrt(ambient² + max(added)²) - default for turbulence
    - LinearSum: Simple summation - used for blockage models
"""

from abc import abstractmethod

import jax.numpy as jnp

from pixwake.jax_utils import ssqrt


class Superposition:
    """Base class for superposition models.

    These models define how to combine ambient and wake-added quantities, such
    as turbulence intensity or velocity deficits.
    """

    @abstractmethod
    def __call__(
        self, ambient: jnp.ndarray, added: jnp.ndarray, subtract: bool = True
    ) -> jnp.ndarray:  # pragma: no cover
        """Combines ambient and added quantities.

        Args:
            ambient: A JAX numpy array of the ambient quantity.
            added: A JAX numpy array of the added quantity from wake effects.

        Returns:
            A JAX numpy array of the effective (combined) quantity.
        """
        raise NotImplementedError


class SqrMaxSum(Superposition):
    """Implements the square-root-of-sum-of-squares superposition.

    This model takes the maximum added contribution from all sources and
    combines it with the ambient value using the formula:
    `sqrt(ambient^2 + max(added)^2)`.
    """

    def __call__(
        self, ambient: jnp.ndarray, added: jnp.ndarray, _: bool = True
    ) -> jnp.ndarray:
        """Combines ambient and added quantities.

        Args:
            ambient: A JAX numpy array of the ambient quantity.
            added: A JAX numpy array of the added quantity from wakes.

        Returns:
            A JAX numpy array of the effective quantity.
        """
        max_added = jnp.max(added, axis=1)
        return jnp.sqrt(ambient**2 + max_added**2)


class SquaredSum(Superposition):
    """Implements superposition by combining values in quadrature.

    This model computes the square root of the sum of squared contributions
    using the formula: `sqrt(sum(added^2))`. This is the default superposition
    for wake deficit models.
    """

    def __call__(
        self, ambient: jnp.ndarray, added: jnp.ndarray, subtract: bool = True
    ) -> jnp.ndarray:
        """Combines values in quadrature (square root of sum of squares).

        Args:
            ambient: A JAX numpy array of the ambient quantity (unused for
                deficit superposition, but included for API consistency).
            added: A JAX numpy array of the deficit values from each upstream
                turbine with shape (n_receivers, n_sources).

        Returns:
            A JAX numpy array of the total deficit with shape (n_receivers,).
        """
        sign = -1 if subtract else 1
        return ambient + sign * ssqrt(jnp.sum(added**2, axis=1))


class LinearSum(Superposition):
    """Implements superposition by linear summation.

    This model computes the simple sum of all contributions using the formula:
    `sum(added)`. This is a straightforward superposition model that can be
    useful for certain applications.
    """

    def __call__(
        self, ambient: jnp.ndarray, added: jnp.ndarray, subtract: bool = True
    ) -> jnp.ndarray:
        """Combines values by linear summation.

        Args:
            ambient: A JAX numpy array of the ambient quantity (unused for
                this superposition, but included for API consistency).
            added: A JAX numpy array of the values from each upstream
                turbine with shape (n_receivers, n_sources).

        Returns:
            A JAX numpy array of the total with shape (n_receivers,).
        """
        sign = -1 if subtract else 1
        return ambient + sign * jnp.sum(added, axis=1)
