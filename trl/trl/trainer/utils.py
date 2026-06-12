# Compatibility shim: re-export symbols that ppov2_config.py and rloo_config.py
# expect at trl.trl.trainer.utils (or trl.trainer.utils via relative import).
from .utils_multi_unified import OnpolicyRuntimeConfig, exact_div

__all__ = ["OnpolicyRuntimeConfig", "exact_div"]
