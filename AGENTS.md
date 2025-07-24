# pixwake Agent Documentation

This document provides instructions for agents on how to use and test the `pixwake` library.

## Overview

`pixwake` is a JAX-based library for fast and differentiable wind farm wake modeling. It provides implementations of the Jensen NOJ (N.O. Jensen) and a RANS (Reynolds-averaged Navier-Stokes) surrogate model. The library is designed for performance-critical applications, such as wind farm layout optimization.

## Installation

The necessary dependencies for running the `pixwake` library are listed in the `pyproject.toml` file. You can install them using `pip`:

```bash
pip install -e .
```

## Testing

The tests for `pixwake` are located in the `test/` directory. To run the tests, use `pytest`:

```bash
pytest
```

The test suite includes:
- Unit tests for the core functions.
- Tests for the NOJ model, including edge cases.
- Tests for the RANS model, including a gradient check that is expected to fail.
- Tests for the AEP and power calculation functions.
- Tests for equivalence with PyWake.

## Usage

The main entry point for running a wake simulation is the `WakeSimulation` class. This class takes a wake model as input and provides a common interface for running simulations.

The general workflow is as follows:
1. Define the turbine characteristics using the `Turbine` and `Curve` classes.
2. Select a wake model, such as `NOJModel` or `RANSModel`.
3. Instantiate the `WakeSimulation` class with the chosen model.
4. Call the simulation with the turbine layout, wind conditions, and turbine definition.
5. Calculate the power and AEP using the `calculate_power` and `calculate_aep` functions.

### Example

Here is a complete example of how to run a simulation with the NOJ model and calculate the AEP:

```python
import jax.numpy as jnp
from pixwake import (
    NOJModel,
    RANSModel,
    WakeSimulation,
    Turbine,
    Curve,
    calculate_aep,
    calculate_power,
)

# 1. Define turbine characteristics
power_curve = Curve(
    wind_speed=jnp.array([4.0, 10.0, 25.0]),
    values=jnp.array([0.0, 2000.0, 2000.0]),
)
ct_curve = Curve(
    wind_speed=jnp.array([4.0, 10.0, 25.0]),
    values=jnp.array([0.8, 0.8, 0.4]),
)
turbine = Turbine(
    rotor_diameter=100.0,
    hub_height=80.0,
    power_curve=power_curve,
    ct_curve=ct_curve,
)

# 2. Select a wake model
# model = NOJModel(k=0.05)
model = RANSModel(ambient_ti=0.1)

# 3. Instantiate the simulation
sim = WakeSimulation(model)

# 4. Call the simulation
xs = jnp.array([0.0, 500.0])
ys = jnp.array([0.0, 0.0])
ws = jnp.array([10.0, 12.0])
wd = jnp.array([270.0, 270.0])

effective_wind_speeds = sim(xs, ys, ws, wd, turbine)

# 5. Calculate power and AEP
power = calculate_power(effective_wind_speeds, turbine.power_curve)
aep = calculate_aep(effective_wind_speeds, turbine.power_curve)

print("Effective wind speeds:", effective_wind_speeds)
print("Power:", power)
print("AEP:", aep)
```
