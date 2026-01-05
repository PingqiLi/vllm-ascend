"""Inference-only Qwen3ResQ model compatible with vLLM."""
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config as Qwen3Config

from vllm.config import VllmConfig, CacheConfig
from vllm.attention import AttentionType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import (
    extract_layer_index, maybe_prefix, PPMissingLayer)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm.distributed import get_pp_group

from vllm_ascend.quantization.w4a4_resq_dynamic import apply_rotation

from vllm.model_executor.layers.layernorm import RMSNorm
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
        # Use custom loader to handle shape mismatch (empty -> actual shape)
        def rotation_weight_loader(param, loaded_weight):
            if param.data.shape != loaded_weight.shape:
                param.data = torch.empty_like(loaded_weight)
            param.data.copy_(loaded_weight)
        setattr(self.rotation_R3, "weight_loader", rotation_weight_loader)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Add qk-norm (Copied from Qwen3Attention to ensure consistency)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        
        # ResQ R3 (U_C): Post-RoPE Rotation
        if self.apply_resq_rotation:
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
        # super().__init__(config, cache_config, quant_config, prefix)
        nn.Module.__init__(self)
        
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # 1. Self Attention (Using ResQ wrapper)
        self.self_attn = Qwen3ResQAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        # 2. MLP: Use standard Qwen3MLP
        # QuantMethod will handle the R4 rotation internally
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        
        # Explicitly mark down_proj for R4 (Hadamard) rotation and register potential learned parameter
        if hasattr(self.mlp, 'down_proj'):
             setattr(self.mlp.down_proj, 'resq_apply_hadamard', True)
             # Register R4 parameter (initialized as empty)
             # If checkpoint contains rotation_R4, it will be loaded here.
             self.mlp.down_proj.register_parameter("rotation_R4", torch.nn.Parameter(torch.empty(0), requires_grad=False))
             # Use custom loader to handle shape mismatch (empty -> actual shape)
             def rotation_weight_loader(param, loaded_weight):
                 if param.data.shape != loaded_weight.shape:
                     param.data = torch.empty_like(loaded_weight)
                 param.data.copy_(loaded_weight)
             setattr(self.mlp.down_proj.rotation_R4, "weight_loader", rotation_weight_loader)

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
                # Map low/high keys (from checkpoint) to int4/int8 parameters (expected by QuantMethod)
                if "weight_low" in name:
                    name = name.replace("weight_low", "weight_int4")
                elif "scale_low" in name:
                    name = name.replace("scale_low", "weight_scale_int4")
                elif "offset_low" in name:
                     name = name.replace("offset_low", "weight_offset_int4")

                if "weight_high" in name:
                    name = name.replace("weight_high", "weight_int8")
                elif "scale_high" in name:
                    name = name.replace("scale_high", "weight_scale_int8")
                elif "offset_high" in name:
                    name = name.replace("offset_high", "weight_offset_int8")

                # ResQ R3 (U_C) Parameter Mapping: self_attn.R3 (Map to rotation_R3)
                if name.endswith(".Uc"):
                    name = name.replace(".Uc", ".self_attn.rotation_R3")
                # ResQ R4 (U_D) Parameter Mapping: mlp.R4 (Map to rotation_R4)
                if name.endswith(".Ud"):
                    name = name.replace(".Ud", ".mlp.rotation_R4")

                yield name, tensor

        loaded_params = super().load_weights(filtered_weights(weights))
        return loaded_params

class Qwen3ResQForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Do NOT call super().__init__ to avoid instantiating standard Qwen3Model via parent
        nn.Module.__init__(self)
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        # Instantiate our custom ResQ model
        self.model = Qwen3ResQModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                 from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
                 self.lm_head = ParallelLMHead(config.vocab_size,
                                               config.hidden_size,
                                               quant_config=quant_config,
                                               prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights with ResQ-specific name mappings.
        Uses WeightsMapper to transform checkpoint key names to model parameter names.
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
        
        # Create mapper for ResQ checkpoint format
        # Maps checkpoint names -> model parameter names
        mapper = WeightsMapper(
            orig_to_new_prefix={
                # ResQ checkpoint uses "resq.layer." (singular), model uses "model.layers." (plural)
                "resq.layer.": "model.layers.",
            },
            orig_to_new_substr={
                # Weight mappings (ResQ uses _low/_high suffix instead of _int4/_int8)
                "weight_low": "weight_int4",
                "weight_high": "weight_int8",
                "scale_low": "weight_scale_int4",
                "scale_high": "weight_scale_int8",
                "offset_low": "weight_offset_int4",
                "offset_high": "weight_offset_int8",
                # Rotation matrix mappings
                ".Uc": ".self_attn.rotation_R3",
                ".Ud": ".mlp.down_proj.rotation_R4",
            }
        )
        
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=mapper)

