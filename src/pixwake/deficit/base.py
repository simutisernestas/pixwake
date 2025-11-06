from abc import ABC, abstractmethod

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
        ws_deficit_m, wake_radius = (
            self.rotor_avg_model(self.compute, ws_eff, ti_eff, ctx)
            if self.rotor_avg_model
            else self.compute(ws_eff, ti_eff, ctx)
        )

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
