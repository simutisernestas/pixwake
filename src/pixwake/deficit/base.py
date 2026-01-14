from abc import ABC, abstractmethod

import jax.numpy as jnp

from pixwake.rotor_avg import RotorAvg
from pixwake.superposition import SquaredSum, Superposition

from ..core import SimulationContext


class WakeDeficit(ABC):
    """Abstract base class for all wake deficit models.

    This class provides the basic structure for wake deficit models, which are
    responsible for calculating the velocity deficit caused by upstream wind
    turbines. The main logic for superposition of deficits is implemented in the
    `__call__` method, which in turn calls the `compute` method that must be
    implemented by subclasses.

    Attributes:
        superposition: A `Superposition` model to combine wake deficits from
            multiple upstream turbines.
    """

    def __init__(
        self,
        use_radius_mask: bool = True,
        rotor_avg_model: RotorAvg | None = None,
        superposition: Superposition | None = None,
    ) -> None:
        """Initializes the `WakeDeficit` model.

        Args:
            use_radius_mask: A boolean indicating whether to use a radius-based
                mask to exclude points that are clearly outside the wake.
            rotor_avg_model: An optional rotor averaging model.
            superposition: A `Superposition` model to combine wake deficits.
                Defaults to `SquaredSum` if not provided.
        """
        self.use_radius_mask = use_radius_mask
        self.rotor_avg_model = rotor_avg_model
        self.superposition = (
            superposition if superposition is not None else SquaredSum()
        )

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
        wake_radius_for_exclude: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, SimulationContext]:
        """Calculates the effective wind speed after considering wake effects.

        This method orchestrates the wake deficit calculation by calling the
        `_deficit` method to get the deficit from each turbine and then
        using the configured superposition model to combine them.

        If a `rotor_avg_model` is provided, it will be used to compute the
        rotor-averaged deficit.

        Args:
            ws_eff: A JAX numpy array of the effective wind speeds at each
                turbine.
            ti_eff: An optional JAX numpy array of the effective turbulence
                intensities at each turbine.
            ctx: The simulation context.
            wake_radius_for_exclude: Optional wake radius from wake model, used
                by blockage models to determine wake exclusion zones. Ignored
                by standard wake deficit models.

        Returns:
            A tuple containing the updated effective wind speeds and the
            simulation context with added wake radius.
        """
        ctx.wake_radius = self._wake_radius(ws_eff, ti_eff, ctx)

        ws_deficit_m = (
            self.rotor_avg_model(self._deficit, ws_eff, ti_eff, ctx)
            if self.rotor_avg_model
            else self._deficit(ws_eff, ti_eff, ctx)
        )

        in_wake_mask = ctx.dw > 0.0
        if self.use_radius_mask:
            in_wake_mask &= jnp.abs(ctx.cw) < ctx.wake_radius

        # Apply mask and use configurable superposition
        masked_deficit = jnp.where(in_wake_mask, ws_deficit_m, 0.0)
        new_eff_ws = self.superposition(ctx.ws, masked_deficit)
        return jnp.maximum(0.0, new_eff_ws), ctx

    @abstractmethod
    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:  # pragma: no cover
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

    @abstractmethod
    def _wake_radius(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:  # pragma: no cover
        """Calculates the wake radius for each turbine.

        This abstract method must be implemented by all subclasses. It is
        responsible for calculating the wake radius for each turbine based on
        the simulation context.

        Args:
            ctx: The simulation context.

        Returns:
            A JAX numpy array representing the wake radius for each turbine.
        """
        raise NotImplementedError
