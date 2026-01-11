# ResQ Debug Module
# 
# This module provides transformers-style ResQ model implementation for debugging.
# It allows direct comparison between original Qwen3 and ResQ-quantized models
# by saving intermediate activations.
#
# The ResQ model is built on top of the original transformers Qwen3,
# replacing attention and MLP modules with ResQ-aware versions that:
# 1. Use dequantized weights from checkpoint A
# 2. Apply Uc rotation after RoPE
# 3. Apply Ud rotation before down_proj
#
# Usage:
#   python -m tools.resq_debug.compare_forward \
#       --original /path/to/qwen3-bf16 \
#       --ckpt-a /path/to/checkpoint_A.pt \
#       --ckpt-b /path/to/transforms_B.pt \
#       --layers 0,1,63

from .modeling_qwen3_resq import Qwen3ResQForCausalLM
from .compare_forward import compare_models

__all__ = [
    "Qwen3ResQForCausalLM", 
    "compare_models",
]
