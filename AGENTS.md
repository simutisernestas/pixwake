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

## mypy

Package must comply with `mypy` type checking:

```bash
pip install mypy
mypy src/
```

## Formatting

Formatting is enforced with `pre-commit`

```bash
pip install pre-commit
pre-commit run --all-files
```

## Usage

The main entry point for running a wake simulation is the `WakeSimulation` class. This class takes a wake model as input and provides a common interface for running winf farm wake simulations.

The general workflow is as follows:
1. Define the turbine characteristics using the `Turbine` and `Curve` classes.
2. Select a wake model, such as `NOJModel` or `RANSModel`, etc.
3. Instantiate the `WakeSimulation` class with the chosen model.
4. Call the simulation with the turbine layout, wind conditions, and turbine definition.
5. Calculate the power and AEP using the `power` and `aep` functions on simulation result object.

### Examples

For examples look into the `test/` directory for package usage examples.



