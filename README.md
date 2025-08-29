# pixwake

`pixwake` is a JAX-based Python library for fast and differentiable wind farm wake modeling. It is designed for applications, such as wind farm layout optimization, where both speed and gradient information are essential.

The library provides implementations of two wake models:

-   **NOJ (N.O. Jensen) Model**: A simple analytical model that is computationally efficient.
-   **RANS (Reynolds-averaged Navier-Stokes) Surrogate Model**: A more accurate model based on a neural network trained on high-fidelity CFD data.

## Installation

You can install `pixwake` and its dependencies using `pip`:

```bash
pip install -e .
```

## Usage

Please refer to test files as they include common usage of the package.

## Code Structure

The `pixwake` library is organized as follows:

-   `src/pixwake/core.py`: Contains the main `WakeSimulation` class, the fixed-point solver, and utility functions for calculating power and AEP.
-   `src/pixwake/models/`: Contains the different wake models.
    -   `base.py`: The abstract base class for all wake models.
    -   `noj.py`: The implementation of the NOJ model.
    -   `rans.py`: The implementation of the RANS surrogate model.
-   `test/`: Contains the unit/integration tests for the library.

## Contributing

Please follow the guidelines in `AGENTS.md` when contributing to this project.
