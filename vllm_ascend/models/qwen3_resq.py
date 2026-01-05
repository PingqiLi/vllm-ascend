"""
Qwen3 ResQ Model Definition (Simplified for Preprocessed BF16 Weights)

This module defines Qwen3 model variants with ResQ rotation support.
Assumes weights have been preprocessed to bf16 using preprocess_resq_weights.py.

Key differences from standard Qwen3:
1. Qwen3ResQAttention: Applies R3 rotation to Q/K after RoPE
2. Qwen3ResQDecoderLayer: Marks down_proj for R4 rotation, registers rotation parameters
3. Qwen3ResQForCausalLM: Uses WeightsMapper for rotation parameter naming
"""
from typing import Iterable, Optional
import torch
import torch.nn as nn

from transformers import Qwen2Config as Qwen3Config

from vllm.config import VllmConfig, CacheConfig, QuantizationConfig
from vllm.model_executor.models.qwen3 import (
    Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import maybe_prefix, PPMissingLayer
from vllm.distributed.parallel_state import get_pp_group
from vllm.attention import Attention

# Import rotation function
from vllm_ascend.quantization.w4a4_resq_dynamic import apply_rotation


class Qwen3ResQAttention(Qwen3Attention):
    """
    Qwen3 Attention with ResQ R3 rotation applied after RoPE.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_resq_rotation = True
        
        # Register R3 parameter (will be loaded from checkpoint)
        self.register_parameter("rotation_R3", nn.Parameter(torch.empty(0), requires_grad=False))
        # Custom loader to handle empty -> actual shape
        def rotation_loader(param, loaded_weight):
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        setattr(self.rotation_R3, "weight_loader", rotation_loader)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # QK norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Apply R3 rotation (post-RoPE)
        if self.apply_resq_rotation:
            rot_mat = None
            if self.rotation_R3.numel() > 0:
                rot_mat = self.rotation_R3
            q = apply_rotation(q, rot_mat)
            k = apply_rotation(k, rot_mat)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3ResQMLP(Qwen3MLP):
    """
    Qwen3 MLP with ResQ R4 rotation applied before down_proj.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Register R4 parameter (will be loaded from checkpoint)
        self.register_parameter("rotation_R4", nn.Parameter(torch.empty(0), requires_grad=False))
        # Custom loader to handle empty -> actual shape
        def rotation_loader(param, loaded_weight):
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        setattr(self.rotation_R4, "weight_loader", rotation_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up_proj: [batch, seq, hidden] -> [batch, seq, 2 * intermediate]
        gate_up, _ = self.gate_up_proj(x)
        
        # Split and apply activation
        i = gate_up.shape[-1] // 2
        gate = gate_up[..., :i]
        up = gate_up[..., i:]
        
        # SiLU(gate) * up
        from vllm.model_executor.layers.activation import SiluAndMul
        intermediate = torch.nn.functional.silu(gate) * up
        
        # Apply R4 rotation before down_proj
        if self.rotation_R4.numel() > 0:
            intermediate = apply_rotation(intermediate, self.rotation_R4)
        
        # down_proj
        output, _ = self.down_proj(intermediate)
        return output


class Qwen3ResQDecoderLayer(Qwen3DecoderLayer):
    """
    Qwen3 Decoder Layer with ResQ rotation support.
    """
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        
        from vllm.attention import AttentionType
        attn_type = AttentionType.DECODER
        dual_chunk_attention_config = None
        
        # Use ResQ Attention
        self.self_attn = Qwen3ResQAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        # Use ResQ MLP with R4 rotation
        self.mlp = Qwen3ResQMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3ResQModel(Qwen3Model):
    """Qwen3 Model using ResQ decoder layers."""
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        super(Qwen3Model, self).__init__(
            vllm_config=vllm_config, 
            prefix=prefix, 
            decoder_layer_type=Qwen3ResQDecoderLayer
        )


class Qwen3ResQForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3 Causal LM with ResQ rotation support.
    
    Uses WeightsMapper to handle rotation parameter naming from preprocessed checkpoint.
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        # Use ResQ model
        self.model = Qwen3ResQModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head")
                )
        else:
            self.lm_head = PPMissingLayer()

        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights with rotation parameter name mapping.
        
        Preprocessed checkpoint uses:
        - model.layers.X.self_attn.rotation_R3
        - model.layers.X.mlp.rotation_R4
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader
        
        # Custom weight loader that handles empty rotation parameters
        def rotation_param_loader(param, loaded_weight):
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        
        # Set custom loader on rotation parameters after loading
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        
        return loader.load_weights(weights)
