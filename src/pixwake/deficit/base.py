from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from pixwake.jax_utils import get_float_eps
from pixwake.rotor_avg import RotorAvg

from ..core import SimulationContext


class WakeDeficit(ABC):
    """Abstract base class for all wake deficit models.

    This class provides the basic structure for wake deficit models, which are
    responsible for calculating the velocity deficit caused by upstream wind
    turbines. The main logic for superposition of deficits is implemented in the
    `__call__` method, which in turn calls the `compute` method that must be
    implemented by subclasses.
    """

    def __init__(
        self,
        use_radius_mask: bool = True,
        rotor_avg_model: RotorAvg | None = None,
    ) -> None:
        """Initializes the `WakeDeficit` model.

        Args:
            use_radius_mask: A boolean indicating whether to use a radius-based
                mask to exclude points that are clearly outside the wake.
            rotor_avg_model: An optional rotor averaging model.
        """
        self.use_radius_mask = use_radius_mask
        self.rotor_avg_model = rotor_avg_model

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates the effective wind speed after considering wake effects.

        This method orchestrates the wake deficit calculation by calling the
        `compute` method to get the deficit from each turbine and then
        superposing them in quadrature to find the total deficit at each point.

        If a `rotor_avg_model` is provided, it will be used to compute the
        rotor-averaged deficit.

        Args:
            ws_eff: A JAX numpy array of the effective wind speeds at each
                turbine.
            ti_eff: An optional JAX numpy array of the effective turbulence
                intensities at each turbine.
            ctx: The simulation context.

        Returns:
            A tuple containing the updated effective wind speeds and the wake
            radius.
        """
        if self.rotor_avg_model:
            # The compute function is defined on a single point in space,
            # but the rotor average model needs to evaluate it at multiple
            # points on the rotor disk. We create a partial function that
            # captures the current state (ws_eff, ti_eff, ctx) and can be
            # called with just the spatial coordinates.
            def compute_at_point(
                dw: jnp.ndarray,
                cw: jnp.ndarray,
                ws_eff: jnp.ndarray,
                ti_eff: jnp.ndarray | None,
                ctx: SimulationContext,
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                # We need to create a new context for each point on the rotor
                # disk. We do this by replacing the downwind and crosswind
                # distances in the original context with the new values.
                # The rest of the context remains the same.
                new_ctx = SimulationContext(
                    turbine=ctx.turbine,
                    dw=dw,
                    cw=cw,
                    ws=ctx.ws,
                    ti=ctx.ti,
                )
                return self.compute(ws_eff, ti_eff, new_ctx)

            # The rotor average model expects the function to be called with
            # specific argument names, so we wrap the call in a lambda.
            # The `unbatched_compute` function is now a closure that can be
            # passed to the rotor average model.
            unbatched_compute = partial(
                compute_at_point, ws_eff=ws_eff, ti_eff=ti_eff, ctx=ctx
            )

            # Reshape inputs to be compatible with the rotor average model,
            # which expects dimensions for wind direction and wind speed cases.
            # In the context of this function, we are only dealing with a single
            # case, so we add dummy dimensions.
            n_receivers, n_sources = ctx.dw.shape
            D_dst = jnp.full((n_receivers, 1, 1), ctx.turbine.rotor_diameter)
            dw = ctx.dw.reshape(n_receivers, n_sources, 1, 1)
            cw = ctx.cw.reshape(n_receivers, n_sources, 1, 1)
            dh = jnp.zeros_like(cw)

            # Call the rotor average model to get the rotor-averaged deficit.
            # The model will call our `unbatched_compute` function at multiple
            # points on the rotor disk and average the results.
            ws_deficit_m, wake_radius = self.rotor_avg_model(
                lambda **kwargs: unbatched_compute(
                    dw=kwargs["dw_ijlk"], cw=kwargs["hcw_ijlk"]
                ),
                D_dst_ijl=D_dst,
                dw_ijlk=dw,
                hcw_ijlk=cw,
                dh_ijlk=dh,
            )
            # Squeeze the dummy dimensions for wind direction and wind speed.
            ws_deficit_m = ws_deficit_m.squeeze(axis=-1).squeeze(axis=-1)
            wake_radius = wake_radius.squeeze(axis=-1).squeeze(axis=-1)
        else:
            # If no rotor average model is provided, just call the compute
            # method directly.
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
        """Computes the wake deficit from each turbine.

        This abstract method must be implemented by all subclasses. It is
        responsible for calculating the velocity deficit caused by each turbine
        at every other turbine's location.

        Args:
            ws_eff: A JAX numpy array of the effective wind speeds at each
                turbine.
            ti_eff: An optional JAX numpy array of the effective turbulence
                intensities at each turbine.
            ctx: The simulation context.

        Returns:
            A tuple containing the wake deficit matrix and the wake radius.
        """
        raise NotImplementedError
