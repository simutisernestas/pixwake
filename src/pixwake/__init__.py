"""Pixwake: A JAX-based wind farm wake simulation library.

Pixwake provides fast, differentiable computation of wind turbine wake effects
using JAX for automatic differentiation and hardware acceleration (CPU/GPU).

Main components:
    - WakeSimulation: Main orchestrator for wake simulations
    - Turbine: Wind turbine physical and performance characteristics
    - Curve: Performance curves (power, thrust coefficient)

Example:
    >>> from pixwake import WakeSimulation, Turbine
    >>> from pixwake.deficit import BastankhahGaussianDeficit
    >>> sim = WakeSimulation(turbine, BastankhahGaussianDeficit())
    >>> result = sim(wt_xs, wt_ys, ws_amb=10.0, wd_amb=270.0)
    >>> print(result.power())
"""

import tempfile
from pathlib import Path

from jax import config as jcfg

from .core import Curve, Turbine, WakeSimulation

__version__ = "0.1.0"

jax_cache_dir = Path(tempfile.gettempdir(), "jax_cache")
jcfg.update("jax_compilation_cache_dir", str(jax_cache_dir))
jcfg.update("jax_persistent_cache_min_entry_size_bytes", -1)
jcfg.update("jax_persistent_cache_min_compile_time_secs", 0.1)
jcfg.update("jax_enable_x64", False)


__all__ = [
    "__version__",
    "WakeSimulation",
    "Curve",
    "Turbine",
]
