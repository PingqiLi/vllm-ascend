# ResQ Debug Module
# 
# This module provides transformers-style ResQ model implementation for debugging.
# It allows direct comparison between original Qwen3 and ResQ-quantized models
# by saving intermediate activations.

from .modeling_qwen3_resq import (
    Qwen3ResQConfig,
    Qwen3ResQForCausalLM,
)
from .compare_forward import compare_models

__all__ = [
    "Qwen3ResQConfig",
    "Qwen3ResQForCausalLM", 
    "compare_models",
]
