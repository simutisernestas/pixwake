import tempfile

from jax import config as jcfg

from .core import WakeSimulation, calculate_power, calculate_aep
from .models.noj import NOJModel
from .models.rans import RANSModel

jcfg.update("jax_compilation_cache_dir", tempfile.gettempdir())
jcfg.update("jax_persistent_cache_min_entry_size_bytes", -1)
jcfg.update("jax_persistent_cache_min_compile_time_secs", 1e-2)
jcfg.update("jax_enable_x64", False)


__all__ = [
    "simulate_case",
    "NOJModel",
    "RANSModel",
    "calculate_power",
    "calculate_aep",
]
