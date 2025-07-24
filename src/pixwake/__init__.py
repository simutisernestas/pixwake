import tempfile
from pathlib import Path

from jax import config as jcfg

from .core import WakeSimulation, calculate_power, calculate_aep, Curve, Turbine
from .models.noj import NOJModel
from .models.rans import RANSModel

jax_cache_dir = Path(tempfile.gettempdir(), "jax_cache")
jcfg.update("jax_compilation_cache_dir", str(jax_cache_dir))
jcfg.update("jax_persistent_cache_min_entry_size_bytes", -1)
jcfg.update("jax_persistent_cache_min_compile_time_secs", 1e-2)
jcfg.update("jax_enable_x64", False)


__all__ = [
    "WakeSimulation",
    "Curve",
    "Turbine",
    "NOJModel",
    "RANSModel",
    "calculate_power",
    "calculate_aep",
]
