"""Inference-only Qwen3ResQ model compatible with vLLM."""
import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from vllm.attention import AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import (Qwen3Attention, Qwen3DecoderLayer,
                                               Qwen3ForCausalLM, Qwen3MLP,
                                               Qwen3Model)
from vllm.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)

from vllm_ascend.quantization.w4a4_resq_dynamic import apply_rotation


# ---------------------------------------------------------------------------
# Shared rotation weight loader (avoid duplicate definitions)
# ---------------------------------------------------------------------------
def rotation_weight_loader(param: torch.nn.Parameter, 
                           loaded_weight: torch.Tensor, 
                           shard_id: Optional[int] = None) -> None:
    """
    Weight loader for rotation matrices (R3/Uc, R4/Ud).
    Handles empty param -> actual shape resize.
    shard_id is accepted but ignored (rotation matrices are not sharded).
    """
    if param.data.shape != loaded_weight.shape:
        param.data = loaded_weight.clone()
    else:
        param.data.copy_(loaded_weight)


# ---------------------------------------------------------------------------
# ResQ weight name mapping utilities
# ---------------------------------------------------------------------------
def map_resq_weight_name(name: str) -> str:
    """
    Map checkpoint weight names to model parameter names.
    
    Handles:
    - layers0 -> layers.0 (layer numbering fix)
    - resq.layer.X.Uc -> model.layers.X.self_attn.rotation_R3
    - resq.layer.X.Ud -> model.layers.X.mlp.down_proj.rotation_R4
    - weight_low/high -> weight_int4/int8
    - scale_low/high -> weight_scale_int4/int8
    - offset_low/high -> weight_offset_int4/int8
    """
    # 1. Fix layer numbering: layers0 -> layers.0
    name = re.sub(r'layers(\d+)', r'layers.\1', name)
    
    # 2. ResQ rotation matrices: resq.layer.X.Uc/Ud -> model.layers.X...
    # Handle "resq.layer.X.Uc" format
    name = re.sub(r'^resq\.layer\.(\d+)\.Uc$', 
                  r'model.layers.\1.self_attn.rotation_R3', name)
    name = re.sub(r'^resq\.layer\.(\d+)\.Ud$', 
                  r'model.layers.\1.mlp.down_proj.rotation_R4', name)
    
    # Handle inline ".Uc" and ".Ud" (fallback for other formats)
    if '.Uc' in name:
        name = name.replace('.Uc', '.self_attn.rotation_R3')
    if '.Ud' in name:
        name = name.replace('.Ud', '.mlp.down_proj.rotation_R4')
    
    # 3. Quantization weight mappings
    # weight_low -> weight_int4, weight_high -> weight_int8
    name = name.replace('.weight_low', '.weight_int4')
    name = name.replace('.weight_high', '.weight_int8')
    name = name.replace('.scale_low', '.weight_scale_int4')
    name = name.replace('.scale_high', '.weight_scale_int8')
    name = name.replace('.offset_low', '.weight_offset_int4')
    name = name.replace('.offset_high', '.weight_offset_int8')
    
    # 4. Handle .int4. and .int8. format (alternative checkpoint format)
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
    
    # 5. Legacy mapping for R3/R4
    if name.endswith(".self_attn.R3"):
        name = name.replace(".self_attn.R3", ".self_attn.rotation_R3")
    if name.endswith(".mlp.down_proj.R4"):
        name = name.replace(".mlp.down_proj.R4", ".mlp.down_proj.rotation_R4")
    
    return name


# ---------------------------------------------------------------------------
# ResQ Attention with post-RoPE rotation (R3 / U_C)
# ---------------------------------------------------------------------------
class Qwen3ResQAttention(Qwen3Attention):
    """Qwen3 Attention with ResQ R3 (U_C) rotation applied after RoPE."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_resq_rotation = True
        
        # Register R3 parameter (initialized as empty, loaded from checkpoint)
        self.register_parameter(
            "rotation_R3", 
            nn.Parameter(torch.empty(0), requires_grad=False)
        )
        setattr(self.rotation_R3, "weight_loader", rotation_weight_loader)

    def forward(self, positions: torch.Tensor, 
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # QK norm (required for Qwen3)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, 
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, 
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # Apply RoPE
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


# ---------------------------------------------------------------------------
# ResQ Decoder Layer
# ---------------------------------------------------------------------------
class Qwen3ResQDecoderLayer(Qwen3DecoderLayer):
    """Qwen3 Decoder Layer with ResQ rotations (R3 for attention, R4 for MLP)."""
    
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Don't call super().__init__() to avoid creating duplicate layers
        nn.Module.__init__(self)
        
        self.hidden_size = config.hidden_size
        
        # 1. Self Attention with ResQ R3 rotation
        self.self_attn = Qwen3ResQAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 1000000),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
            dual_chunk_attention_config=getattr(
                config, "dual_chunk_attention_config", None),
        )
        
        # 2. MLP (R4 rotation handled in quantization method)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # 3. Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, 
                                        eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, 
                                                 eps=config.rms_norm_eps)
        
        # 4. Mark down_proj for R4 rotation and register rotation parameter
        if hasattr(self.mlp, 'down_proj'):
            setattr(self.mlp.down_proj, 'resq_apply_hadamard', True)
            self.mlp.down_proj.register_parameter(
                "rotation_R4", 
                nn.Parameter(torch.empty(0), requires_grad=False)
            )
            setattr(self.mlp.down_proj.rotation_R4, "weight_loader", 
                    rotation_weight_loader)

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
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


# ---------------------------------------------------------------------------
# ResQ Model (complete rewrite of __init__ to avoid inheritance issues)
# ---------------------------------------------------------------------------
class Qwen3ResQModel(nn.Module):
    """
    Qwen3 Model with ResQ support.
    
    Complete rewrite of __init__ to properly initialize all components
    and use Qwen3ResQDecoderLayer instead of Qwen3DecoderLayer.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # 1. Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # 2. Decoder layers with ResQ support
        def get_layer(layer_prefix: str) -> Qwen3ResQDecoderLayer:
            return Qwen3ResQDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=layer_prefix,
            )
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        
        # 3. Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 4. Intermediate tensors factory
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[dict] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        else:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights with ResQ-specific name mapping."""
        
        # Define stacked parameter mappings
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", 0),
            ("qkv_proj", "k_proj", 1),
            ("qkv_proj", "v_proj", 2),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        
        for name, loaded_weight in weights:
            # Apply ResQ weight name mapping
            name = map_resq_weight_name(name)
            
            # Handle stacked parameters
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name in name:
                    name = name.replace(shard_name, param_name)
                    # Check if param exists
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", 
                                            default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                # Not a stacked param, load directly
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", 
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        
        return loaded_params


# ---------------------------------------------------------------------------
# ResQ CausalLM wrapper
# ---------------------------------------------------------------------------
class Qwen3ResQForCausalLM(nn.Module):
    """
    Qwen3 CausalLM with ResQ support.
    
    Complete rewrite to properly integrate Qwen3ResQModel.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.vllm_config = vllm_config
        
        # Create ResQ Model
        self.model = Qwen3ResQModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        
        # LM head
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[dict] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                        sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Forward to model's load_weights with ResQ mapping."""
        # Collect weights into a list so we can iterate multiple times if needed
        weights_list = list(weights)
        
        # Load model weights
        loaded = self.model.load_weights(iter(weights_list))
        
        # Load lm_head weights separately if not tied
        if not self.config.tie_word_embeddings:
            lm_head_params = dict(self.lm_head.named_parameters())
            for name, loaded_weight in weights_list:
                name = map_resq_weight_name(name)
                if "lm_head" in name:
                    # Extract param name after lm_head
                    param_name = name.split("lm_head.")[-1] if "lm_head." in name else "weight"
                    if param_name in lm_head_params:
                        param = lm_head_params[param_name]
                        weight_loader = getattr(param, "weight_loader", 
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded.add(name)
        
        return loaded

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        next_tokens = self.logits_processor.sample(logits, sampling_metadata)
        return next_tokens
