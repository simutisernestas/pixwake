from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from .core import SimulationContext


class RotorAvg(ABC):
    """Abstract base class for all rotor average models."""

    @abstractmethod
    def __call__(
        self,
        func: Callable,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jax.Array | tuple[jax.Array, ...]:  # pragma: no cover
        """Computes the rotor-averaged value of a function."""
        raise NotImplementedError


_CGI_NODES_AND_WEIGHTS = {
    4: lambda pm: (pm * jnp.array([0.5, 0.5, 1 / 4])).T,
    7: lambda pm: jnp.concatenate(
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
    ).T,
    9: lambda pm: jnp.concatenate(
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
    ).T,
    21: lambda pm: jnp.concatenate(
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
    ).T,
}


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
        (
            self.nodes_x,
            self.nodes_y,
            self.weights,
        ) = self._get_cgi_nodes_and_weights(n_points)

    @staticmethod
    def _get_cgi_nodes_and_weights(
        n: int,
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
        if n not in _CGI_NODES_AND_WEIGHTS:
            raise ValueError(f"Invalid number of points: {n}")
        x, y, w = _CGI_NODES_AND_WEIGHTS[n](pm)
        return (jnp.array(x), jnp.array(y), jnp.array(w))

    def __call__(
        self,
        func: Callable,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jax.Array | tuple[jax.Array, ...]:
        """Computes the rotor-averaged value of `func`.

        This method evaluates the function `func` at a set of predefined
        integration points on the rotor disk and computes a weighted average of
        the results. The integration points and weights are determined by the
        CGI method.

        Args:
            func: The function to be rotor-averaged. It is expected to have a
                signature `func(ws_eff, ti_eff, ctx)`, where `ws_eff` and
                `ti_eff` are the effective wind speed and turbulence intensity,
                and `ctx` is the simulation context.
            ws_eff: The effective wind speeds at each turbine.
            ti_eff: The effective turbulence intensities at each turbine.
            ctx: The simulation context, containing information about the
                wind farm layout, wind conditions, etc.

        Returns:
            The rotor-averaged value of the function `func`.
        """
        # Get the diameter of the destination turbines
        n_receivers, n_sources = ctx.dw.shape
        D_dst = jnp.full((n_receivers, n_sources), ctx.turbine.rotor_diameter)
        R_dst = D_dst / 2.0

        # Create a new axis for the integration points
        dw = ctx.dw[..., jnp.newaxis]
        cw = ctx.cw[..., jnp.newaxis]
        # Get the offsets for the integration points
        node_x_offset = self.nodes_x.reshape(1, 1, -1) * R_dst[..., jnp.newaxis]
        node_y_offset = self.nodes_y.reshape(1, 1, -1) * R_dst[..., jnp.newaxis]

        # Calculate the new crosswind and downwind distances for each
        # integration point
        hcw_at_nodes = cw + node_x_offset
        dh_at_nodes = 0.0 + node_y_offset

        # The downwind distance is the same for all integration points
        dw_at_nodes = jnp.broadcast_to(dw, hcw_at_nodes.shape)

        # Create a new simulation context for the integration points
        ctx_nodes = SimulationContext(
            turbine=ctx.turbine,
            dw=dw_at_nodes,
            cw=jnp.sqrt(hcw_at_nodes**2 + dh_at_nodes**2),
            ws=ctx.ws,
            ti=ctx.ti,
        )
        # Evaluate the function at the integration points
        values_at_nodes = func(ws_eff, ti_eff, ctx_nodes)

        # Compute the weighted average of the values at the integration points
        weights_broadcast = self.weights.reshape(1, 1, -1)

        # Weighted sum over last dimension (integration points)
        return jax.tree.map(
            lambda x: jnp.sum(x * weights_broadcast, axis=-1), values_at_nodes
        )
