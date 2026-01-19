"""Base classes for deficit models.

This module defines the abstract base classes for wake and blockage deficit models:
    - DeficitModelBase: Common interface for all deficit models
    - WakeDeficit: Base class for downstream wake deficit models
    - BlockageDeficit: Base class for upstream blockage deficit models
"""

from abc import ABC, abstractmethod
from dataclasses import replace

import jax.numpy as jnp

from pixwake.rotor_avg import RotorAvg
from pixwake.superposition import LinearSum, SquaredSum, Superposition

from ..core import SimulationContext


class DeficitModelBase(ABC):
    """Abstract base class for all deficit models (wake and blockage).

    This class defines the common interface that both wake deficit models
    (which operate downstream) and blockage deficit models (which operate
    upstream) must implement.

    Attributes:
        superposition: A `Superposition` model to combine deficits from
            multiple turbines.
        use_radius_mask: Whether to use radius-based masking.
        rotor_avg_model: Optional rotor averaging model.
    """

    def __init__(
        self,
        use_radius_mask: bool = True,
        rotor_avg_model: RotorAvg | None = None,
        superposition: Superposition | None = None,
    ) -> None:
        """Initializes the deficit model.

        Args:
            use_radius_mask: Whether to use a radius-based mask to exclude
                points outside the effect zone.
            rotor_avg_model: An optional rotor averaging model.
            superposition: A `Superposition` model to combine deficits.
        """
        self.use_radius_mask = use_radius_mask
        self.rotor_avg_model = rotor_avg_model
        self.superposition = superposition

    @abstractmethod
    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
        wake_radius_for_exclude: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, SimulationContext]:  # pragma: no cover
        """Calculates the effective wind speed after applying deficit effects.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.
            wake_radius_for_exclude: Optional wake radius for exclusion checks.

        Returns:
            A tuple of (updated effective wind speeds, updated context).
        """
        raise NotImplementedError

    @abstractmethod
    def _deficit(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:  # pragma: no cover
        """Computes the deficit from each turbine.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            Deficit matrix (n_receivers, n_sources) in m/s.
        """
        raise NotImplementedError

    @abstractmethod
    def _wake_radius(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> jnp.ndarray:  # pragma: no cover
        """Calculates the effect radius for each turbine.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.

        Returns:
            Effect radius array.
        """
        raise NotImplementedError


class WakeDeficit(DeficitModelBase):
    """Base class for downstream wake deficit models.

    This class provides the implementation for wake deficit models, which
    calculate the velocity deficit caused by upstream wind turbines in
    the downstream direction.

    Wake deficit models:
    - Operate downstream (dw > 0)
    - Reduce wind speed in the wake region
    - Use SquaredSum superposition by default
    """

    def __init__(
        self,
        use_radius_mask: bool = True,
        rotor_avg_model: RotorAvg | None = None,
        superposition: Superposition | None = None,
    ) -> None:
        """Initializes the WakeDeficit model.

        Args:
            use_radius_mask: Whether to use a radius-based mask to exclude
                points outside the wake cone.
            rotor_avg_model: An optional rotor averaging model.
            superposition: A `Superposition` model to combine wake deficits.
                Defaults to `SquaredSum` if not provided.
        """
        super().__init__(
            use_radius_mask=use_radius_mask,
            rotor_avg_model=rotor_avg_model,
            superposition=superposition if superposition is not None else SquaredSum(),
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
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.
            wake_radius_for_exclude: Ignored for wake models (used by blockage).

        Returns:
            A tuple of (updated effective wind speeds, updated context).
        """
        _ = wake_radius_for_exclude  # unused by wake models

        wake_radius = self._wake_radius(ws_eff, ti_eff, ctx)
        ctx = replace(ctx, wake_radius=wake_radius)

        ws_deficit_m = (
            self.rotor_avg_model(self._deficit, ws_eff, ti_eff, ctx)
            if self.rotor_avg_model
            else self._deficit(ws_eff, ti_eff, ctx)
        )

        # Only apply to downstream positions (dw > 0)
        in_wake_mask = ctx.dw > 0.0
        if self.use_radius_mask:
            assert ctx.wake_radius is not None
            in_wake_mask &= jnp.abs(ctx.cw) < ctx.wake_radius

        # Apply mask and use configurable superposition
        masked_deficit = jnp.where(in_wake_mask, ws_deficit_m, 0.0)
        assert self.superposition is not None
        new_eff_ws = self.superposition(ctx.ws, masked_deficit)
        return jnp.maximum(0.0, new_eff_ws), ctx


class BlockageDeficit(DeficitModelBase):
    """Base class for upstream blockage deficit models.

    This class provides the implementation for blockage (induction) deficit
    models, which calculate the velocity reduction upstream of wind turbines
    due to the pressure field induced by the rotor's thrust.

    Blockage deficit models:
    - Operate upstream (dw < 0) with optional downstream speedup
    - Reduce wind speed due to induction effects
    - Use LinearSum superposition by default (preserves sign for speedup)
    """

    def __init__(
        self,
        use_radius_mask: bool = False,
        rotor_avg_model: RotorAvg | None = None,
        superposition: Superposition | None = None,
        exclude_downstream_speedup: bool = False,
    ) -> None:
        """Initializes the BlockageDeficit model.

        Args:
            use_radius_mask: Whether to use radius-based masking. Typically
                False for blockage models which handle masking internally.
            rotor_avg_model: An optional rotor averaging model.
            superposition: A `Superposition` model to combine blockage deficits.
                Defaults to `LinearSum` if not provided.
            exclude_downstream_speedup: If True, exclude downstream speedup
                effect when inside the wake region. Used in combined
                wake+blockage mode to match PyWake behavior.
        """
        super().__init__(
            use_radius_mask=use_radius_mask,
            rotor_avg_model=rotor_avg_model,
            superposition=superposition if superposition is not None else LinearSum(),
        )
        self.exclude_downstream_speedup = exclude_downstream_speedup

    def __call__(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
        wake_radius_for_exclude: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, SimulationContext]:
        """Calculate effective wind speed after blockage/induction effects.

        Blockage models apply signed deficits based on position:
        - Upstream (dw < 0): positive deficit (blockage, reduces wind speed)
        - Downstream (dw > 0): negative deficit (speedup, increases wind speed)

        When exclude_downstream_speedup=True (for combined wake+blockage mode),
        speedup is excluded when the receiver is in the wake of the source
        (dw > 0 AND |cw| < wake_radius). This matches PyWake's exclude_wake
        behavior.

        Args:
            ws_eff: Effective wind speeds at each turbine.
            ti_eff: Effective turbulence intensities at each turbine.
            ctx: The simulation context.
            wake_radius_for_exclude: Optional wake radius from wake model,
                used for exclude_wake check in combined mode.

        Returns:
            A tuple of (updated effective wind speeds, updated context).
        """
        # Compute blockage effect radius
        wake_radius = self._wake_radius(ws_eff, ti_eff, ctx)
        ctx = replace(ctx, wake_radius=wake_radius)

        # Compute deficit (positive = reduction in wind speed)
        ws_deficit_m = (
            self.rotor_avg_model(self._deficit, ws_eff, ti_eff, ctx)
            if self.rotor_avg_model
            else self._deficit(ws_eff, ti_eff, ctx)
        )

        if self.exclude_downstream_speedup:
            # In combined wake+blockage mode, use PyWake's exclude_wake logic:
            # - Upstream (dw < 0): apply blockage (positive deficit)
            # - Downstream in wake (dw > 0, |cw| < wake_radius): exclude (0)
            # - Downstream outside wake (dw > 0, |cw| >= wake_radius): apply speedup
            exclude_radius = (
                wake_radius_for_exclude
                if wake_radius_for_exclude is not None
                else ctx.wake_radius
            )

            assert exclude_radius is not None
            in_wake = (ctx.dw > 0.0) & (jnp.abs(ctx.cw) < exclude_radius)

            signed_deficit = jnp.where(
                ctx.dw < 0.0,
                ws_deficit_m,  # upstream: blockage
                jnp.where(
                    in_wake, 0.0, -ws_deficit_m
                ),  # downstream: speedup if not in wake
            )
        else:
            # Apply sign based on location:
            # - Upstream (dw < 0): positive deficit (blockage)
            # - Downstream (dw > 0): negative deficit (speedup)
            signed_deficit = jnp.where(ctx.dw < 0.0, ws_deficit_m, -ws_deficit_m)

        # Apply superposition
        assert self.superposition is not None
        new_eff_ws = self.superposition(ctx.ws, signed_deficit)

        return jnp.maximum(0.0, new_eff_ws), ctx
