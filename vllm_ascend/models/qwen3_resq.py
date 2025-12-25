"""Inference-only Qwen3ResQ model compatible with vLLM."""
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config as Qwen3Config

from vllm.config import VllmConfig, CacheConfig
from vllm.attention import AttentionType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import (
    extract_layer_index, maybe_prefix)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader)

from vllm_ascend.quantization.w4a4_resq_dynamic import apply_rotation

from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM


class Qwen3ResQAttention(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ResQ R3 (U_C) Project: Rotation for Query and Key
        self.apply_resq_rotation = True
        
        # Register R3 parameter (initialized as empty, loaded from checkpoint)
        # Note: Shape depends on implementation (head-wise vs full).
        # Assuming full rotation matrix based on reference.
        self.register_parameter("rotation_R3", torch.nn.Parameter(torch.empty(0), requires_grad=False))
        # Use custom loader to handle shape mismatch if needed (e.g. if not in checkpoint, stay empty)
        setattr(self.rotation_R3, "weight_loader", default_weight_loader)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        
        # ResQ R3 (U_C): Post-RoPE Rotation
        if self.apply_resq_rotation:
             # If rotation_R3 is loaded (non-empty), use it. Otherwise use Hadamard if logic dictates (currently Hadamard logic assumes None)
             # However, apply_rotation handles None by falling back to Hadamard.
             # If R3 is missing from checkpoint, rotation_R3 will be empty tensor(0).
             
             rot_mat = None
             if self.rotation_R3.numel() > 0:
                 rot_mat = self.rotation_R3
                 
             q = apply_rotation(q, rot_mat)
             k = apply_rotation(k, rot_mat)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

# Removed Qwen3ResQMLP as U_D is now handled by AscendResQW4A4DynamicLinearMethod

class Qwen3ResQDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix)
        
        # 1. Override Self Attention
        self.self_attn = Qwen3ResQAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=getattr(config, "rope_theta", 1000000),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
        )
        
        # 2. MLP: Use standard Qwen3MLP
        # QuantMethod will handle the R4 rotation internally
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # Explicitly mark down_proj for R4 (Hadamard) rotation and register potential learned parameter
        if hasattr(self.mlp, 'down_proj'):
             setattr(self.mlp.down_proj, 'resq_apply_hadamard', True)
             # Register R4 parameter (initialized as empty)
             # If checkpoint contains rotation_R4, it will be loaded here.
             self.mlp.down_proj.register_parameter("rotation_R4", torch.nn.Parameter(torch.empty(0), requires_grad=False))
             setattr(self.mlp.down_proj.rotation_R4, "weight_loader", default_weight_loader)

        # 3. ResQ Specific: Basis Change omitted for Shared Basis mode

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        # ResQ Step 1: Rotate residual (Shared Basis: Identity -> Omitted)
             
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
            
        # ResQ Step 2: Rotate residual (Shared Basis: Identity -> Omitted)
             
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class Qwen3ResQModel(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        super(Qwen3Model, self).__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3ResQDecoderLayer)
        
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def filtered_weights(weights_iterable):
            for name, tensor in weights_iterable:
                if ".int4." in name:
                    name = name.replace(".int4.", ".weight_int4_") 
                    name = name.replace("weight_int4_weight", "weight_int4")
                    name = name.replace("weight_int4_weight_scale", "weight_scale_int4")
                    name = name.replace("weight_int4_weight_offset", "weight_offset_int4")
                elif ".int8." in name:
                    name = name.replace(".int8.", ".weight_int8_")
                    name = name.replace("weight_int8_weight", "weight_int8")
                    name = name.replace("weight_int8_weight_scale", "weight_scale_int8")
                    name = name.replace("weight_int8_weight_offset", "weight_offset_int8")
                
                # ResQ R3 (U_C) Parameter Mapping: self_attn.R3 (Map to rotation_R3)
                if name.endswith(".self_attn.R3"):
                    name = name.replace(".self_attn.R3", ".self_attn.rotation_R3")
                # ResQ R4 (U_D) Parameter Mapping: mlp.down_proj.R4 (Map to rotation_R4)
                if name.endswith(".mlp.down_proj.R4"):
                    name = name.replace(".mlp.down_proj.R4", ".mlp.down_proj.rotation_R4")

                yield name, tensor

        loaded_params = super().load_weights(filtered_weights(weights))
        return loaded_params

class Qwen3ResQForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model = Qwen3ResQModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
             from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
             self.lm_head = ParallelLMHead(config.vocab_size,
                                           config.hidden_size,
                                           quant_config=quant_config,
                                           prefix=maybe_prefix(prefix, "lm_head"))
