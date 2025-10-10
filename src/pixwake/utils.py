import jax.numpy as jnp


def ct2a_madsen(
    ct: jnp.ndarray, ct2ap: tuple = (0.2460, 0.0586, 0.0883)
) -> jnp.ndarray:
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
