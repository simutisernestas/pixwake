from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp


class RotorAvg(ABC):
    """Abstract base class for all rotor average models."""

    @abstractmethod
    def __call__(
        self, func: Callable, **kwargs: jax.Array
    ) -> jax.Array | tuple[jax.Array, ...]:  # pragma: no cover
        """Computes the rotor-averaged value of a function."""
        raise NotImplementedError


class CGIRotorAvg(RotorAvg):
    """
    Standalone implementation of CGI (Composite Gauss Integration) Rotor Averaging Model
    This implementation removes inheritance complexity and clearly explains each component.
    CGI uses predefined integration points and weights to approximate rotor-averaged values.
    Composite Gauss Integration Rotor Averaging Model
    This model computes rotor-averaged quantities by evaluating a function at
    specific points across the rotor disk and combining them with weights.
    The CGI method uses predetermined node locations and weights optimized for
    circular rotor averaging. Different numbers of points provide different
    accuracy-performance tradeoffs.
    """

    nodes_x: jax.Array
    nodes_y: jax.Array
    weights: jax.Array
    n_points: int

    def __init__(self, n_points: int = 7) -> None:
        """
        Initialize CGI rotor averaging model
        Parameters
        ----------
        n_points : int
            Number of integration points: 4, 7, 9, or 21
            More points = higher accuracy but slower computation
        """
        self.n_points = n_points

        # Get predefined node positions and weights for this configuration
        self.nodes_x, self.nodes_y, self.weights = self._get_cgi_nodes_and_weights(
            n_points
        )

    def _get_cgi_nodes_and_weights(
        self, n: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get CGI integration points and weights for circular rotor
        These are predetermined optimal locations for integrating over a circle.
        Coordinates are normalized to rotor radius (range: -1 to 1)
        Returns
        -------
        nodes_x : array
            X-coordinates of integration points (normalized by radius)
        nodes_y : array
            Y-coordinates of integration points (normalized by radius)
        weights : array
            Integration weights (sum to 1.0)
        """
        pm = jnp.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
        if n == 4:
            x, y, w = (pm * jnp.array([0.5, 0.5, 1 / 4])).T
        elif n == 7:
            x, y, w = jnp.concatenate(
                [
                    jnp.array(
                        [
                            [0, 0, 1 / 4],
                            [-jnp.sqrt(2 / 3), 0, 1 / 8],
                            [jnp.sqrt(2 / 3), 0, 1 / 8],
                        ]
                    ),
                    pm * jnp.array([jnp.sqrt(1 / 6), jnp.sqrt(1 / 2), 1 / 8]),
                ]
            ).T
        elif n == 9:
            x, y, w = jnp.concatenate(
                [
                    jnp.array(
                        [
                            [0, 0, 1 / 6],
                            [-1, 0, 1 / 24],
                            [1, 0, 1 / 24],
                            [0, -1, 1 / 24],
                            [0, 1, 1 / 24],
                        ]
                    ),
                    pm * jnp.array([1 / 2, 1 / 2, 1 / 6]),
                ]
            ).T
        elif n == 21:
            x, y, w = jnp.concatenate(
                [
                    jnp.array([[0, 0, 1 / 9]]),
                    jnp.array(
                        [
                            [
                                jnp.sqrt((6 - jnp.sqrt(6)) / 10)
                                * jnp.cos(2 * jnp.pi * k / 10),
                                jnp.sqrt((6 - jnp.sqrt(6)) / 10)
                                * jnp.sin(2 * jnp.pi * k / 10),
                                (16 + jnp.sqrt(6)) / 360,
                            ]
                            for k in range(1, 11)
                        ]
                    ),
                    jnp.array(
                        [
                            [
                                jnp.sqrt((6 + jnp.sqrt(6)) / 10)
                                * jnp.cos(2 * jnp.pi * k / 10),
                                jnp.sqrt((6 + jnp.sqrt(6)) / 10)
                                * jnp.sin(2 * jnp.pi * k / 10),
                                (16 - jnp.sqrt(6)) / 360,
                            ]
                            for k in range(1, 11)
                        ]
                    ),
                ]
            ).T
        else:
            raise ValueError(f"Invalid number of points: {n}")
        return (jnp.array(x), jnp.array(y), jnp.array(w))

    def __call__(
        self, func: Callable, **kwargs: jax.Array
    ) -> jax.Array | tuple[jax.Array, ...]:
        """
        Compute rotor-averaged value by evaluating func at integration points
        Parameters
        ----------
        func : callable
            Function to evaluate at each rotor point
            Should accept **kwargs and return array with shape (..., n_points)
        **kwargs : dict
            Additional arguments passed to func, must include:
            D_dst_ijl : array
                Destination (rotor) diameter [m]
                Shape: (i turbines, j points, l wind directions)
            cw_ijlk : array
                Horizontal crosswind distance from source to destination [m]
                Shape: (i, j, l, k wind speeds)
            dw_ijlk : array
                Downwind distance [m]
                Shape: (i, j, l, k)
            dh_ijlk : array
                Vertical distance [m]
                Shape: (i, j, l, k)
        Returns
        -------
        result_ijlk : array or tuple of arrays
            Rotor-averaged result
            Shape: (i, j, l, k)
        """
        D_dst_ijl = kwargs.pop("D_dst_ijl")
        cw_ijlk = kwargs.pop("hcw_ijlk")
        dw_ijlk = kwargs.pop("dw_ijlk")
        dh_ijlk = kwargs.pop("dh_ijlk")

        # Get rotor radius (half diameter)
        R_ijlk1 = D_dst_ijl[..., jnp.newaxis, jnp.newaxis] / 2.0

        # Calculate positions of integration points on rotor
        node_x_offset = self.nodes_x.reshape(1, 1, 1, 1, -1) * R_ijlk1
        node_y_offset = self.nodes_y.reshape(1, 1, 1, 1, -1) * R_ijlk1

        # Adjust crosswind and vertical distances to integration points
        cw_ijlk_expanded = cw_ijlk[..., jnp.newaxis]
        dh_ijlk_expanded = dh_ijlk[..., jnp.newaxis]

        # Add offsets to get distance to each integration point
        cw_at_nodes = cw_ijlk_expanded + node_x_offset  # horizontal offset
        dh_at_nodes = dh_ijlk_expanded + node_y_offset  # vertical offset

        # Downwind distance doesn't change across rotor
        dw_at_nodes = jnp.broadcast_to(dw_ijlk[..., jnp.newaxis], cw_at_nodes.shape)

        # Prepare kwargs for function evaluation at all points
        eval_kwargs = kwargs.copy()
        eval_kwargs.update(
            {"dw_ijlk": dw_at_nodes, "hcw_ijlk": cw_at_nodes, "dh_ijlk": dh_at_nodes}
        )
        # Evaluate function at all integration points
        values_at_nodes = func(**eval_kwargs)

        # Weight and sum to get rotor average
        weights_broadcast = self.weights.reshape(1, 1, 1, 1, -1)

        # Weighted sum over last dimension (integration points)
        return jax.tree.map(
            lambda x: jnp.sum(x * weights_broadcast, axis=-1), values_at_nodes
        )
