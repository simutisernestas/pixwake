from abc import ABC, abstractmethod

import jax.numpy as jnp


class RotorAvgModel(ABC):
    """Base class for rotor average models."""

    @abstractmethod
    def __call__(self, func, **kwargs):
        raise NotImplementedError


class AreaOverlapAvgModel(RotorAvgModel):
    """Area overlap rotor average model.

    This model calculates the wake effect by considering the overlapping area
    between the wake and the rotor.
    """

    def __call__(self, func, **kwargs):
        """Calculates the rotor-averaged wake effect.

        Args:
            func: The wake deficit or turbulence function.
            **kwargs: Additional keyword arguments for the wake function.

        Returns:
            The rotor-averaged wake effect.
        """
        # This model assumes the function `func` returns the centerline deficit
        # or turbulence, which is then scaled by the overlap factor.
        res, wake_radius = func(**kwargs)
        return res * self.overlapping_area_factor(
            wake_radius, kwargs["cw"], kwargs["D_dst"]
        ), wake_radius

    def overlapping_area_factor(self, wake_radius, cw, D_dst):
        """Calculate overlapping factor.

        Args:
            wake_radius: The radius of the wake.
            cw: The crosswind distance.
            D_dst: The diameter of the downstream turbine.

        Returns:
            The area overlapping factor.
        """
        return self._cal_overlapping_area_factor(wake_radius, D_dst / 2.0, jnp.abs(cw))

    def _cal_overlapping_area_factor(self, R1, R2, d):
        """Calculate the overlapping area of two circles.

        The calculation formula is based on the standard formula for the area
        of intersection of two circles.

        Args:
            R1: Radius of the first circle (wake).
            R2: Radius of the second circle (rotor).
            d: Distance between two centers.

        Returns
        -------
        A_ol_f: array_like
            Area overlapping factor (overlapping_area / rotor_area).
        """
        # Ensure R_big >= R_small for numerical stability
        Rmax = jnp.maximum(R1, R2)
        Rmin = jnp.minimum(R1, R2)

        # Handle non-overlapping case
        no_overlap_mask = d >= (R1 + R2)
        # Handle full overlap case (smaller circle inside larger one)
        full_overlap_mask = d <= jnp.abs(R1 - R2)

        # Partial overlap calculation
        eps = 2 * jnp.finfo(float).eps

        def arccos_lim(x):
            return jnp.arccos(jnp.clip(x, -1.0 + eps, 1.0 - eps))

        # Angles of the sectors in the two circles
        alpha = arccos_lim((R1**2 + d**2 - R2**2) / (2.0 * jnp.maximum(R1 * d, eps)))
        beta = arccos_lim((R2**2 + d**2 - R1**2) / (2.0 * jnp.maximum(R2 * d, eps)))

        # Area of the triangle formed by the centers and an intersection point, multiplied by 2
        triangle_area_x2 = 0.5 * jnp.sqrt(
            jnp.maximum(
                0, (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)
            )
        )

        # Overlapping area
        overlapping_area = alpha * R1**2 + beta * R2**2 - triangle_area_x2
        # Factor is the ratio of overlapping area to the rotor area
        partial_factor = overlapping_area / (jnp.pi * R2**2)

        # Determine the factor for the full overlap case
        # If wake is smaller than rotor, factor is (wake_area / rotor_area)
        # If rotor is smaller than wake, factor is 1.0
        full_overlap_factor = jnp.where(R1 < R2, (R1**2) / (R2**2), 1.0)

        # Combine the cases
        factor = jnp.where(no_overlap_mask, 0.0, partial_factor)
        factor = jnp.where(full_overlap_mask, full_overlap_factor, factor)

        return factor