from typing import Any, Callable

import jax.numpy as jnp

from ..core import SimulationContext
from ..jax_utils import get_float_eps
from ..utils import ct2a_madsen
from .base import WakeDeficit


class NOJDeficit(WakeDeficit):
    """Implements the N.O. Jensen (NOJ) wake deficit model.

    This is a classic and simple analytical model that assumes a linearly
    expanding wake with a top-hat profile for the velocity deficit.

    Attributes:
        k: The wake expansion coefficient, which determines how quickly the
            wake expands with downwind distance.
        ct2a: A callable that converts the thrust coefficient (`Ct`) to the
            induction factor (`a`).
    """

    def __init__(
        self,
        k: float = 0.1,
        ct2a: Callable = ct2a_madsen,
        use_radius_mask: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the `NOJDeficit` model.

        Args:
            k: The wake expansion coefficient.
            ct2a: A callable to convert `Ct` to the induction factor.
            use_radius_mask: A boolean indicating whether to use a radius-based
                mask.
        """
        super().__init__(use_radius_mask, **kwargs)
        self.k = k
        self.ct2a = ct2a

    def compute(
        self,
        ws_eff: jnp.ndarray,
        ti_eff: jnp.ndarray | None,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the wake deficit using the NOJ model.

        Args:
            ws_eff: A JAX numpy array of the effective wind speeds at each
                turbine.
            ti_eff: An optional JAX numpy array of the effective turbulence
                intensities (not used in this model).
            ctx: The simulation context.

        Returns:
            A tuple containing the wake deficit matrix and the wake radius.
        """
        _ = ti_eff  # unused

        wt = ctx.turbine
        rr = wt.rotor_diameter / 2
        wake_radius = (rr) + self.k * ctx.dw
        all2all_deficit_matrix = (
            2 * self.ct2a(wt.ct(ws_eff))
            * (rr / jnp.maximum(wake_radius, get_float_eps())) ** 2
        )  # fmt: skip
        return ctx.ws * all2all_deficit_matrix, wake_radius
