import jax.numpy as jnp

from pixwake import Curve, Turbine

# Data extracted from the Vestas V80 XML definition file.
# The performance data points for wind speed, power, and thrust coefficient.
# fmt:off
wind_speeds = jnp.array(
    [
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
    ]
)
# Power output in kilowatts (kW). The original data was in Watts.
power_outputs_kw = jnp.array(
    [
        66.6, 154.0, 282.0, 460.0, 696.0, 996.0, 1341.0, 1661.0, 1866.0, 1958.0, 1988.0, 1997.0,
        1999.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0,
    ]
)
# Thrust coefficients (Ct).
thrust_coefficients = jnp.array(
    [
        0.818, 0.806, 0.804, 0.805, 0.806, 0.807, 0.793, 0.739, 0.709, 0.409, 0.314, 
        0.249, 0.202, 0.167, 0.140, 0.118, 0.101, 0.088, 0.076, 0.067, 0.059, 0.052,
    ]
)
# fmt:on

# Create the performance Curve objects for the turbine.
power_curve = Curve(wind_speed=wind_speeds, values=power_outputs_kw)
ct_curve = Curve(wind_speed=wind_speeds, values=thrust_coefficients)

# Define the Vestas V80 turbine as a pixwake Turbine object.
# This object can be imported and used in wake simulations.
vestas_v80 = Turbine(
    rotor_diameter=80.0,
    hub_height=67.0,
    power_curve=power_curve,
    ct_curve=ct_curve,
)
