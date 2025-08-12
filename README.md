# pixwake

`pixwake` is a JAX-based Python library for fast and differentiable wind farm wake modeling. It is designed for performance-critical applications, such as wind farm layout optimization, where both speed and gradient information are essential.

The library provides implementations of two common wake models:

-   **NOJ (N.O. Jensen) Model**: A simple analytical model that is computationally efficient.
-   **RANS (Reynolds-averaged Navier-Stokes) Surrogate Model**: A more accurate model based on a neural network trained on high-fidelity CFD data.

## Installation

You can install `pixwake` and its dependencies using `pip`:

```bash
pip install -e .
```

## Usage

The main entry point for running a wake simulation is the `WakeSimulation` class. This class takes a wake model as input and provides a common interface for running simulations.

Here is a complete example of how to run a simulation with the `RANSModel` and calculate the Annual Energy Production (AEP):

```python
import jax.numpy as jnp
from pixwake import (
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

# 2. Select and configure the wake model
model = RANSModel(ambient_ti=0.1)

# 3. Instantiate the simulation
sim = WakeSimulation(model)

# 4. Define the wind farm layout and wind conditions
xs = jnp.array([0.0, 500.0])
ys = jnp.array([0.0, 0.0])
ws = jnp.array([10.0, 12.0])
wd = jnp.array([270.0, 270.0])

# 5. Run the simulation to get the effective wind speeds
effective_wind_speeds = sim(xs, ys, ws, wd, turbine)

# 6. Calculate power and AEP
power = calculate_power(effective_wind_speeds, turbine.power_curve)
aep = calculate_aep(effective_wind_speeds, turbine.power_curve)

print("Effective wind speeds:", effective_wind_speeds)
print("Power:", power)
print("AEP:", aep)
```

## The RANS Surrogate Model

The `RANSModel` is a surrogate model that approximates the results of high-fidelity RANS CFD simulations. It consists of two neural networks:

-   **Wake Deficit Model**: Predicts the velocity deficit in the wake of a turbine.
-   **Wake Added Turbulence Model**: Predicts the added turbulence intensity in the wake.

### The `use_effective` Parameter

The `RANSModel`'s `compute_deficit` method has a boolean parameter called `use_effective`. This parameter significantly changes how the wake deficit is calculated and, consequently, the final solved state of the wind farm.

**`use_effective=True` (Default Behavior)**

When `use_effective` is `True`, the wake deficit is calculated as an absolute reduction in wind speed. The strength of the wake is proportional to the *effective wind speed* at the turbine generating the wake.

The calculation is as follows:

```
deficit_value = model.predict_deficit(...)
absolute_deficit = deficit_value * effective_wind_speed_at_waking_turbine
downstream_wind_speed = free_stream_wind_speed - absolute_deficit
```

This approach is more physically realistic, as the momentum extracted by a turbine (and thus the strength of its wake) depends on the wind speed it actually experiences.

**`use_effective=False`**

When `use_effective` is `False`, the wake deficit is calculated as a fractional reduction relative to the *free-stream wind-speed*.

The calculation is as follows:

```
fractional_deficit = model.predict_deficit(...)
downstream_wind_speed = free_stream_wind_speed * (1.0 - fractional_deficit)
```

In this case, the strength of the wake is independent of the local wind speed at the turbine. This can lead to less accurate results, especially in large wind farms where turbines are deep inside the array and experience significantly reduced wind speeds.

**Impact on Solved Wind Farm State**

The choice of `use_effective` affects the fixed-point iteration process that solves for the stable wind farm state:

-   With `use_effective=True`, the wake deficits are coupled more strongly. A change in wind speed at one turbine will have a more pronounced effect on the wakes it generates, which in turn affects downstream turbines. This can lead to a more complex and potentially more realistic equilibrium state.
-   With `use_effective=False`, the coupling is weaker. The wakes are primarily determined by the free-stream conditions, which can result in a simpler but less accurate solution.

For most applications, it is recommended to use the default setting of `use_effective=True`.

## Code Structure

The `pixwake` library is organized as follows:

-   `src/pixwake/core.py`: Contains the main `WakeSimulation` class, the fixed-point solver, and utility functions for calculating power and AEP.
-   `src/pixwake/models/`: Contains the different wake models.
    -   `base.py`: The abstract base class for all wake models.
    -   `noj.py`: The implementation of the NOJ model.
    -   `rans.py`: The implementation of the RANS surrogate model.
-   `test/`: Contains the unit tests for the library.

## Contributing

Please follow the guidelines in `AGENTS.md` when contributing to this project.
