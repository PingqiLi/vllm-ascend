from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

class Qwen3ResQForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3ResQForCausalLM
    
    This class inherits from vLLM's standard Qwen3ForCausalLM.
    Currently, the standard Qwen3 implementation (supporting QK-Norm) matches
    the requirements for ResQ when UB_FUSED=0 (no Uc/R3 rotation required, 
    no o_proj reordering required).
    
    This class serves as a registered architecture name "Qwen3ResQForCausalLM" 
    allows users to explicitly specify it in config.json.
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
