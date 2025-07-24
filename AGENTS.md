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
python -m pytest
```

The test suite includes:
- Unit tests for the core functions.
- Tests for the NOJ model, including edge cases.
- Tests for the RANS model, including a gradient check that is expected to fail.
- Tests for the AEP and power calculation functions.
- Tests for equivalence with PyWake.

## Models

### Jensen NOJ Model

The Jensen NOJ model is a simple analytical wake model that is widely used for wind farm layout optimization. It is implemented in the `simulate_case_noj` function.

**Usage Example:**

```python
import jax.numpy as jnp
from pixwake import simulate_case_noj

# Turbine layout
xs = jnp.array([0.0, 500.0])
ys = jnp.array([0.0, 0.0])

# Wind conditions
ws = jnp.array([10.0])
wd = jnp.array([270.0])

# Turbine parameters
D = 100.0
k = 0.05
ct_curve = jnp.array([[0.0, 0.8], [20.0, 0.8]])

# Simulate the case
effective_wind_speeds = simulate_case_noj(xs, ys, ws, wd, D, k, ct_curve)
```

### RANS Surrogate Model

The RANS surrogate model is a neural network that approximates the results of a more complex RANS simulation. It is implemented in the `simulate_case_rans` function.

**Usage Example:**

```python
import jax.numpy as jnp
from pixwake import simulate_case_rans

# Turbine layout
xs = jnp.array([0.0, 500.0])
ys = jnp.array([0.0, 0.0])

# Wind conditions
ws = 9.0
wd = 90.0

# Turbine parameters
D = 178.0
ct_curve = jnp.array([[3.0, 0.8], [25.0, 0.6]])

# Simulate the case
effective_wind_speeds = simulate_case_rans(xs, ys, ws, wd, D, ct_curve)
```

## Energy Calculation

The `pixwake` library provides functions for calculating power and Annual Energy Production (AEP) from effective wind speeds.

### `ws2power`

Calculates the power produced by each turbine for each time step.

**Usage Example:**
```python
import jax.numpy as jnp
from pixwake import ws2power

ws_eff = jnp.array([[10.0, 12.0], [15.0, 20.0]])
power_curve = jnp.array([[0.0, 0.0], [10.0, 1000.0], [25.0, 3000.0]])

power = ws2power(ws_eff, power_curve)
```

### `ws2aep`

Calculates the total Annual Energy Production (AEP) for the wind farm.

**Usage Example:**
```python
import jax.numpy as jnp
from pixwake import ws2aep

ws_eff = jnp.array([[10.0, 12.0], [15.0, 20.0]])
power_curve = jnp.array([[0.0, 0.0], [10.0, 1000.0], [25.0, 3000.0]])

aep = ws2aep(ws_eff, power_curve)
```

The `ws2aep` function can also take a probability distribution for the wind conditions.
