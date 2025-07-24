from flax.struct import dataclass

import jax.numpy as jnp


@dataclass
class Curve:
    wind_speed: jnp.ndarray
    values: jnp.ndarray


@dataclass
class Turbine:
    rotor_diameter: float
    hub_height: float
    power_curve: Curve
    ct_curve: Curve
