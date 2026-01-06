"""
Qwen3 ResQ Model Definition (Simplified for Preprocessed BF16 Weights)

This module defines Qwen3 model variants with ResQ rotation support.
Assumes weights have been preprocessed to bf16 using preprocess_resq_weights.py.

Key differences from standard Qwen3:
1. Qwen3ResQAttention: Applies R3 rotation to Q/K after RoPE
2. Qwen3ResQMLP: Applies R4 rotation before down_proj
3. Optional W4A4 fake quantization for accuracy testing

Usage:
    # Without fake quantization (for debugging rotations):
    RESQ_FAKE_QUANT=0 vllm serve ...
    
    # With fake quantization (for W4A4 accuracy testing, default):
    RESQ_FAKE_QUANT=1 vllm serve ...
"""
from typing import Iterable, Optional
import torch
import torch.nn as nn
import logging
import os

from transformers import Qwen2Config as Qwen3Config

from vllm.config import VllmConfig, CacheConfig, QuantizationConfig
from vllm.model_executor.models.qwen3 import (
    Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import maybe_prefix, PPMissingLayer
from vllm.distributed.parallel_state import get_pp_group
from vllm.attention import Attention

# Setup logger for ResQ debugging
logger = logging.getLogger(__name__)

# Configuration flags (can be set via environment variables)
RESQ_DEBUG = os.environ.get("RESQ_DEBUG", "0") == "1"
RESQ_FAKE_QUANT = os.environ.get("RESQ_FAKE_QUANT", "1") == "1"
RESQ_HIGH_BITS = int(os.environ.get("RESQ_HIGH_BITS", "8"))
RESQ_LOW_BITS = int(os.environ.get("RESQ_LOW_BITS", "4"))
RESQ_HIGH_FRACTION = float(os.environ.get("RESQ_HIGH_FRACTION", "0.125"))


# ============================================================================
# Utility Functions for ResQ
# ============================================================================

def apply_rotation(x: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply rotation to input tensor using block-wise approach.
    
    If rotation_matrix is None, return x unchanged.
    If rotation_matrix is provided, apply block-wise MatMul:
        1. Reshape x from [..., N] to [..., N/K, K] where K is rotation block size
        2. Matmul with R [K, K]
        3. Reshape back to [..., N]
    
    This is mathematically equivalent to multiplying by a block-diagonal matrix.
    """
    if rotation_matrix is None:
        return x
    
    R = rotation_matrix.to(device=x.device, dtype=x.dtype)
    K = R.shape[0]  # Block size (e.g., 128 or 256)
    
    original_shape = x.shape
    N = original_shape[-1]
    
    if N == K:
        return torch.matmul(x, R)
    
    if N % K != 0:
        raise ValueError(f"Feature dim {N} must be divisible by rotation block size {K}")
    
    num_blocks = N // K
    x_blocked = x.view(*original_shape[:-1], num_blocks, K)
    x_rotated = torch.matmul(x_blocked, R)
    return x_rotated.view(*original_shape)


def fake_quantize_per_token(x: torch.Tensor, bits: int = 8, sym: bool = True) -> torch.Tensor:
    """
    Apply per-token fake quantization to tensor.
    
    Args:
        x: Input tensor [..., hidden_dim]
        bits: Number of bits for quantization (default 8)
        sym: Whether to use symmetric quantization (default True)
    
    Returns:
        Fake-quantized tensor (same shape and dtype as input)
    """
    if bits >= 16:
        return x
    
    original_dtype = x.dtype
    x_float = x.float()
    
    if sym:
        qmax = (1 << (bits - 1)) - 1
        x_abs_max = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scale = x_abs_max / qmax
        x_quant = (x_float / scale).round().clamp(-qmax, qmax)
        x_dequant = x_quant * scale
    else:
        qmax = (1 << bits) - 1
        x_min = x_float.amin(dim=-1, keepdim=True)
        x_max = x_float.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-10) / qmax
        zero_point = (-x_min / scale).round()
        x_quant = ((x_float / scale) + zero_point).round().clamp(0, qmax)
        x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant.to(original_dtype)


def apply_mixed_precision_fake_quant(x: torch.Tensor, high_fraction: float = 0.125,
                                      high_bits: int = 8, low_bits: int = 4) -> torch.Tensor:
    """
    Apply mixed-precision fake quantization to activation tensor.
    
    Channel layout (matching msit quantization code):
    - First (1 - high_fraction) channels: low precision (4-bit)
    - Last high_fraction channels: high precision (8-bit)
    
    Args:
        x: Input tensor [..., hidden_dim]
        high_fraction: Fraction of channels using high precision (default 0.125 = 12.5%)
        high_bits: Bits for high-precision part (default 8)
        low_bits: Bits for low-precision part (default 4)
    
    Returns:
        Fake-quantized tensor (same shape as input)
    """
    hidden_dim = x.shape[-1]
    k_high = int(hidden_dim * high_fraction)
    k_low = hidden_dim - k_high
    
    if k_high > 0 and k_low > 0:
        # Split: first part is low (4-bit), last part is high (8-bit)
        # This matches msit: weight_low = weight[:, :low_dim], weight_high = weight[:, low_dim:]
        x_low = fake_quantize_per_token(x[..., :k_low], bits=low_bits)
        x_high = fake_quantize_per_token(x[..., k_low:], bits=high_bits)
        return torch.cat([x_low, x_high], dim=-1)
    else:
        # No split, use single precision
        return fake_quantize_per_token(x, bits=high_bits)


class Qwen3ResQAttention(Qwen3Attention):
    """
    Qwen3 Attention with ResQ R3 rotation applied after RoPE.
    Optionally applies W4A4 fake quantization when RESQ_FAKE_QUANT=1.
    """
    _debug_logged = False  # Class-level flag to log only once
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_resq_rotation = True
        
        # Register R3 parameter (will be loaded from checkpoint)
        self.register_parameter("rotation_R3", nn.Parameter(torch.empty(0), requires_grad=False))
        # Custom loader to handle empty -> actual shape
        def rotation_loader(param, loaded_weight):
            if RESQ_DEBUG:
                logger.warning(f"[ResQ] Loading rotation_R3: loaded_weight.shape={loaded_weight.shape}")
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        setattr(self.rotation_R3, "weight_loader", rotation_loader)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Debug logging (only once)
        if RESQ_DEBUG and not Qwen3ResQAttention._debug_logged:
            Qwen3ResQAttention._debug_logged = True
            logger.warning(f"[ResQ] Qwen3ResQAttention.forward called")
            logger.warning(f"[ResQ]   RESQ_FAKE_QUANT={RESQ_FAKE_QUANT}, HIGH_BITS={RESQ_HIGH_BITS}, LOW_BITS={RESQ_LOW_BITS}")
            logger.warning(f"[ResQ]   rotation_R3.numel()={self.rotation_R3.numel()}, shape={self.rotation_R3.shape}")
            logger.warning(f"[ResQ]   hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
            if self.rotation_R3.numel() > 0:
                logger.warning(f"[ResQ]   rotation_R3 stats: min={self.rotation_R3.min():.4f}, max={self.rotation_R3.max():.4f}")
            if hasattr(self.qkv_proj, 'weight'):
                w = self.qkv_proj.weight
                logger.warning(f"[ResQ]   qkv_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}")
        
        # Apply fake quantization to input (simulating dynamic activation quantization)
        if RESQ_FAKE_QUANT:
            hidden_states = apply_mixed_precision_fake_quant(
                hidden_states, RESQ_HIGH_FRACTION, RESQ_HIGH_BITS, RESQ_LOW_BITS
            )
        
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
        
        # Apply fake quantization to o_proj input
        if RESQ_FAKE_QUANT:
            attn_output = apply_mixed_precision_fake_quant(
                attn_output, RESQ_HIGH_FRACTION, RESQ_HIGH_BITS, RESQ_LOW_BITS
            )
        
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3ResQMLP(Qwen3MLP):
    """
    Qwen3 MLP with ResQ R4 rotation applied before down_proj.
    Optionally applies W4A4 fake quantization when RESQ_FAKE_QUANT=1.
    """
    _debug_logged = False  # Class-level flag to log only once
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Register R4 parameter (will be loaded from checkpoint)
        self.register_parameter("rotation_R4", nn.Parameter(torch.empty(0), requires_grad=False))
        # Custom loader to handle empty -> actual shape
        def rotation_loader(param, loaded_weight):
            if RESQ_DEBUG:
                logger.warning(f"[ResQ] Loading rotation_R4: loaded_weight.shape={loaded_weight.shape}")
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        setattr(self.rotation_R4, "weight_loader", rotation_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Debug logging (only once)
        if RESQ_DEBUG and not Qwen3ResQMLP._debug_logged:
            Qwen3ResQMLP._debug_logged = True
            logger.warning(f"[ResQ] Qwen3ResQMLP.forward called")
            logger.warning(f"[ResQ]   RESQ_FAKE_QUANT={RESQ_FAKE_QUANT}")
            logger.warning(f"[ResQ]   rotation_R4.numel()={self.rotation_R4.numel()}, shape={self.rotation_R4.shape}")
            if self.rotation_R4.numel() > 0:
                logger.warning(f"[ResQ]   rotation_R4 stats: min={self.rotation_R4.min():.4f}, max={self.rotation_R4.max():.4f}")
            if hasattr(self.gate_up_proj, 'weight'):
                w = self.gate_up_proj.weight
                logger.warning(f"[ResQ]   gate_up_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}")
            if hasattr(self.down_proj, 'weight'):
                w = self.down_proj.weight
                logger.warning(f"[ResQ]   down_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}")
        
        # Apply fake quantization to gate_up_proj input
        if RESQ_FAKE_QUANT:
            x = apply_mixed_precision_fake_quant(
                x, RESQ_HIGH_FRACTION, RESQ_HIGH_BITS, RESQ_LOW_BITS
            )
        
        # gate_up_proj: [batch, seq, hidden] -> [batch, seq, 2 * intermediate]
        gate_up, _ = self.gate_up_proj(x)
        
        # Split and apply activation
        i = gate_up.shape[-1] // 2
        gate = gate_up[..., :i]
        up = gate_up[..., i:]
        
        # SiLU(gate) * up
        intermediate = torch.nn.functional.silu(gate) * up
        
        # Apply R4 rotation before down_proj
        if self.rotation_R4.numel() > 0:
            intermediate = apply_rotation(intermediate, self.rotation_R4)
        
        # Apply fake quantization to down_proj input (after R4 rotation)
        if RESQ_FAKE_QUANT:
            intermediate = apply_mixed_precision_fake_quant(
                intermediate, RESQ_HIGH_FRACTION, RESQ_HIGH_BITS, RESQ_LOW_BITS
            )
        
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
        
        if RESQ_DEBUG:
            logger.warning("[ResQ] load_weights called")
            # Log first few weight names and shapes
            weights_list = list(weights)
            logger.warning(f"[ResQ] Total weights in checkpoint: {len(weights_list)}")
            
            # Check for rotation weights
            rotation_count = 0
            sample_weights = []
            for name, tensor in weights_list[:20]:
                sample_weights.append(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            for name, tensor in weights_list:
                if "rotation" in name:
                    rotation_count += 1
                    logger.warning(f"[ResQ] Found rotation weight: {name}, shape={tensor.shape}")
            logger.warning(f"[ResQ] Sample weights (first 20):")
            for s in sample_weights:
                logger.warning(f"[ResQ] {s}")
            logger.warning(f"[ResQ] Total rotation weights found: {rotation_count}")
            
            # Convert back to iterator for loader
            weights = iter(weights_list)
        
        # Set custom loader on rotation parameters after loading
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        
        loaded_keys = loader.load_weights(weights)
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ] Loaded {len(loaded_keys)} weight keys")
            # Check if rotation parameters were loaded by checking their sizes
            for name, param in self.named_parameters():
                if "rotation" in name:
                    logger.warning(f"[ResQ] After loading - {name}: numel={param.numel()}, shape={param.shape}")
        
        return loaded_keys
