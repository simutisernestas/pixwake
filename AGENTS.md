# Agent Documentation for pixwake

This document provides instructions for agents on how to use and test the `pixwake` library.

## Overview

`pixwake` is a JAX-based library for fast and differentiable wind farm wake modeling. It provides implementations of the multiple wake deficit models. The library is designed for performance-critical applications, such as wind farm layout optimization.

## Testing (pip)

The tests for `pixwake` are located in the `test/` directory. To run the tests, use `pytest`:

```bash
pip install py_wake pytest memory_profiler pytest-xdist pytest-cov mypy -e .
pytest
```

## Testing (pixi)

The tests for `pixwake` are located in the `test/` directory. To run the tests, use `pytest`:

```bash
pixi run test
```

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
pre-commit install
pre-commit run --all-files
```

## Usage

The main entry point for running a wake simulation is the `WakeSimulation` class. This class takes a wake model as input and provides a common interface for running wind farm wake simulations.

The general workflow is as follows:
- Define the turbine characteristics using the `Turbine` and `Curve` classes.
- Select a wake model, such as `BastankhahGaussianDeficit`, `NOJDeficit`, etc.
- Instantiate the `WakeSimulation` class with the chosen model and turbine.
- Call the simulation with the turbine layout and wind conditions.
- Calculate the power and AEP using the `power` and `aep` functions on simulation result object.

For examples look into the `test/` directory.
