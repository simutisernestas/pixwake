import jax
import jax.numpy as jnp

from .jax_utils import ssqrt


def ct2a_madsen(ct: jax.Array, ct2ap: tuple = (0.2460, 0.0586, 0.0883)) -> jax.Array:
    """Approximates the axial induction factor `a` from the thrust coefficient `Ct`.

    This function uses an empirical formula from Madsen et al. (2020) to provide
    a polynomial approximation of the relationship between the thrust
    coefficient and the axial induction factor, as derived from Blade Element
    Momentum (BEM) theory.

    Args:
        ct: A JAX numpy array of thrust coefficients.
        ct2ap: A tuple of polynomial coefficients for the approximation.

    Returns:
        A JAX numpy array of the corresponding axial induction factors.

    Reference:
        Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.:
        Implementation of the blade element momentum model on a polar grid and
        its aeroelastic load impact, Wind Energ. Sci., 5, 1–27,
        https://doi.org/10.5194/wes-5-1-2020, 2020.
    """
    return ct * (ct2ap[0] + ct * (ct2ap[1] + ct * ct2ap[2]))


def ct2a_mom1d(ct: jax.Array) -> jax.Array:
    """1D momentum, CT = 4a(1-a), with CT forced to below 1."""
    return 0.5 * (1.0 - ssqrt(1.0 - jnp.minimum(1, ct)))
