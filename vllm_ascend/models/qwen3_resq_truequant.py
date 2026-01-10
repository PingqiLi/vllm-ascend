"""
Qwen3 ResQ True Quantization Model

This module implements ResQ with direct loading of msmodelslim quantized weights.
Designed for TP=1 to avoid complex tensor parallelism in Hadamard rotation.

Weight Format from msmodelslim (权重A):
======================================
- resq.Hd                           # [K, K] Hadamard block matrix
- resq.Hd_K                         # scalar, K value  
- resq.intermediate_size            # scalar
- resq.down_proj_blocksize          # scalar (typically 256)
- resq.layer.*.Uc                   # [head_dim, head_dim] Q/K post-RoPE rotation
- resq.layer.*.Pd                   # [blocksize, blocksize] MLP rotation

For each linear layer (q/k/v/o_proj, gate/up/down_proj):
- weight_low                        # int4 quantized, [out_dim, in_dim_low]
- weight_high                       # int8 quantized, [out_dim, in_dim_high]  
- scale_low                         # per-row scales for int4
- scale_high                        # per-row scales for int8
- offset_low                        # (optional) zero points for int4
- offset_high                       # (optional) zero points for int8

Note: in_dim_low + in_dim_high = in_features, typically 87.5% low + 12.5% high

This implementation:
1. Directly loads int4/int8 weights without dequantization
2. Performs dequantization during forward (or uses fused kernel)
3. Applies online U_d rotation for MLP and U_c rotation for attention
4. Only supports TP=1 to simplify implementation
"""

from typing import Iterable, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import logging
import os

from transformers import Qwen2Config as Qwen3Config

from vllm.config import VllmConfig, CacheConfig, QuantizationConfig
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix, PPMissingLayer
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed import get_tensor_model_parallel_world_size

logger = logging.getLogger(__name__)

RESQ_DEBUG = os.environ.get("RESQ_DEBUG", "0") == "1"

# Fix for msmodelslim rearrange/quantize mismatch in o_proj:
# - rearrange_columns uses: model_dim * high_fraction (e.g., 5120 * 0.125 = 640 -> 10 cols/head)
# - calibrator uses: weight.shape[1] * high_fraction (e.g., 8192 * 0.125 = 1024 -> 16 cols/head)
# When enabled, use the rearrange logic (10 cols/head) to match the actual column order
RESQ_O_PROJ_REARRANGE_FIX = os.environ.get("RESQ_O_PROJ_REARRANGE_FIX", "0") == "1"


# ============================================================================
# Hadamard Transform Utilities
# ============================================================================

def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    assert is_pow2(n), f"Last dimension must be power of 2, got {n}"
    
    original_shape = u.shape
    x = u.reshape(-1, n).clone()
    
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    
    return x.view(original_shape)


# ============================================================================
# Mixed-Precision Linear Layer
# ============================================================================

class ResQMixedPrecisionLinear(nn.Module):
    """
    Linear layer with ResQ mixed-precision quantization using NPU fused kernels.
    
    Uses torch_npu.npu_quant_matmul for efficient quantized matmul:
    - Separate matmul for int4 (low) and int8 (high) parts
    - Sum results to get final output
    - No bf16 dequantization, avoids OOM
    
    Layout: [out_features, in_features] where in_features = in_low + in_high
    
    Reference: 
    - https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_quant_matmul.md
    - vllm_ascend/ops/resq_quant_matmul.py
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        high_fraction: float = 0.125,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_fraction = high_fraction
        
        self.in_high = int(in_features * high_fraction)
        self.in_low = in_features - self.in_high
        
        # Quantized weights - will be loaded from checkpoint
        # weight_low: int4 stored as int8, [out, in_low]
        # weight_high: int8, [out, in_high]
        self.register_buffer('weight_low', torch.empty(0))
        self.register_buffer('weight_high', torch.empty(0))
        self.register_buffer('scale_low', torch.empty(0))    # [out] or [out, 1]
        self.register_buffer('scale_high', torch.empty(0))   # [out] or [out, 1]
        self.register_buffer('offset_low', torch.empty(0))   # optional
        self.register_buffer('offset_high', torch.empty(0))  # optional
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward using quantized matmul for both int4 and int8 parts.
        
        y = x_low @ W_low^T * scale_low * lxScale + x_high @ W_high^T * scale_high * rxScale
        """
        from vllm_ascend.ops.resq_quant_matmul import resq_quant_matmul
        
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Flatten batch dimensions: (..., in_features) -> (M, in_features)
        x_2d = x.view(-1, self.in_features)
        M = x_2d.shape[0]
        
        # Split input along K dimension
        x_low = x_2d[:, :self.in_low]    # (M, in_low) for int4 weights
        x_high = x_2d[:, self.in_low:]   # (M, in_high) for int8 weights
        
        # Per-token dynamic quantization
        # int4 part: qmax=7
        x_low_abs_max = x_low.abs().amax(dim=-1, keepdim=True)
        lxScale = (x_low_abs_max / 7.0).clamp(min=1e-10).squeeze(-1).to(torch.float32)
        x_low_int8 = torch.round(x_low / lxScale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        
        # int8 part: qmax=127
        x_high_abs_max = x_high.abs().amax(dim=-1, keepdim=True)
        rxScale = (x_high_abs_max / 127.0).clamp(min=1e-10).squeeze(-1).to(torch.float32)
        x_high_int8 = torch.round(x_high / rxScale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        
        # Concatenate quantized input: (M, K)
        x_quant = torch.cat([x_low_int8, x_high_int8], dim=-1)
        
        # Concatenate weights: (in_low, out) + (in_high, out) -> (K, N)
        # Note: stored as (out, in_*), need to transpose
        weight = torch.cat([self.weight_low.T, self.weight_high.T], dim=0)  # (K, N)
        
        # Call quantized matmul
        output = resq_quant_matmul(
            x_quant,
            weight,
            self.scale_low,
            self.scale_high,
            lxScale,
            rxScale,
            self.in_low,
            groupList=None,  # E=1
            outDtype=original_dtype,
        )
        
        if self.bias is not None:
            output = output + self.bias
        
        # Restore shape
        output_shape = list(original_shape[:-1]) + [self.out_features]
        return output.view(output_shape)


# ============================================================================
# Rotation Functions
# ============================================================================

def apply_block_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal rotation."""
    if R is None or R.numel() == 0:
        return x
    
    K = R.shape[0]
    original_shape = x.shape
    N = original_shape[-1]
    
    if N == K:
        return torch.matmul(x.float(), R.float()).to(x.dtype)
    
    if N % K != 0:
        raise ValueError(f"Dimension {N} must be divisible by block size {K}")
    
    num_blocks = N // K
    x_blocked = x.float().reshape(*original_shape[:-1], num_blocks, K)
    x_rotated = torch.matmul(x_blocked, R.float())
    return x_rotated.reshape(original_shape).to(x.dtype)


def apply_ud_rotation(
    x: torch.Tensor,
    Pd: torch.Tensor,
    Hd: Optional[torch.Tensor],
    K: int,
    blocksize: int,
) -> torch.Tensor:
    """
    Apply Ud rotation before down_proj for ResQ inference.
    
    During quantization (rotate_mlp_output_hadamard), weights are fused with:
        Wd_new = Ua.T @ Wd_old @ Ud
        where Ud = block_diag(Pd).T @ H  (code order: Pd.T first, then H)
    
    At inference, activation x needs to match the weight's input space:
        x_transformed = x @ Ud = x @ block_diag(Pd).T @ H
    
    The correct order is:
        1. Apply block_diag(Pd).T to each block
        2. Apply Hadamard H = Hd ⊗ H_butterfly
    
    Note: 
        - K = Hd dimension (e.g., 100 for Qwen3-32B)
        - blocksize = down_proj_blocksize (e.g., 256)
        - intermediate_size = K * blocksize (e.g., 25600)
        - num_blocks = K (since K = intermediate_size / blocksize)
    
    The Hadamard transform H = Hd ⊗ H_butterfly where:
        - Hd: [K, K] block matrix
        - H_butterfly: applied via fast butterfly algorithm on blocksize dimension
    
    This matches msmodelslim's matmul_hadU_cpu implementation.
    """
    original_shape = x.shape
    n = x.shape[-1]  # intermediate_size
    
    # Validate dimensions
    if n != K * blocksize:
        raise ValueError(
            f"Dimension mismatch: intermediate_size={n} != K*blocksize={K}*{blocksize}={K*blocksize}. "
            f"Check resq.Hd_K and resq.down_proj_blocksize configuration."
        )
    
    original_dtype = x.dtype
    x = x.float()
    
    # Reshape: (..., n) -> (..., K, blocksize) where K = num_blocks
    x = x.reshape(*original_shape[:-1], K, blocksize)
    
    # Step 1: Apply block_diag(Pd).T block-wise (x @ Pd.T for each block)
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = torch.matmul(x, Pd_f32.T)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    # First apply butterfly Hadamard on the blocksize dimension (last dim)
    x = hadamard_transform(x.contiguous())
    
    # Then apply Hd on the K dimension (second-to-last dim)
    # This matches matmul_hadU_cpu: hadK @ input_tensor where input has shape [batch, K, n//K]
    if Hd is not None and K > 1:
        batch_shape = x.shape[:-2]
        batch_size = 1
        for d in batch_shape:
            batch_size *= d
        x = x.reshape(batch_size, K, blocksize)
        
        Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
        # Apply Hd: [K, K] @ [batch, K, blocksize] -> [batch, K, blocksize]
        x = torch.einsum('ij,bjk->bik', Hd_f32, x)
        
        x = x.reshape(*batch_shape, K, blocksize)
    
    # Normalize: divide by sqrt(n) to match msmodelslim's matmul_hadU_cpu
    x = x / math.sqrt(n)
    
    # Compensate for msmodelslim bug: missing K factor in weight fusion
    # 
    # Original paper (project-resq) uses: K * hadK in matmul_hadU_cuda
    # But msmodelslim uses: hadK (without K factor)
    # 
    # This means msmodelslim's fused weights are scaled by 1/K compared to paper.
    # To compensate, we need to scale activations by K.
    # 
    # Combined with Hd normalization (elements ~1/sqrt(K)), the total compensation
    # needed is K * sqrt(K) = K^1.5, but since matmul_hadU includes /sqrt(n) on
    # both sides, the effective scale mismatch is just K.
    #
    # For now, apply K compensation to make scale = 1.0
    if Hd is not None and K > 1:
        hd_max = Hd.abs().max().item()
        if hd_max < 0.5:  # Normalized Hd (elements ~±1/sqrt(K))
            x = x * K
    
    return x.reshape(original_shape).to(original_dtype)


# ============================================================================
# Attention with ResQ
# ============================================================================

class Qwen3ResQTrueQuantAttention(nn.Module):
    """Qwen3 Attention with ResQ mixed-precision and U_c rotation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        rope_scaling: Optional[Dict],
        prefix: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        
        # Projections with mixed-precision quantization
        self.q_proj = ResQMixedPrecisionLinear(hidden_size, self.q_size)
        self.k_proj = ResQMixedPrecisionLinear(hidden_size, self.kv_size)
        self.v_proj = ResQMixedPrecisionLinear(hidden_size, self.kv_size)
        self.o_proj = ResQMixedPrecisionLinear(self.q_size, hidden_size)
        
        # QK Norm with custom weight_loader for proper device handling
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        
        # Set weight_loader for norm weights (like resq_fake_v2 pattern)
        # This lets vLLM's standard weight loading handle device placement
        def norm_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)
        setattr(self.q_norm.weight, "weight_loader", norm_weight_loader)
        setattr(self.k_norm.weight, "weight_loader", norm_weight_loader)
        
        # RoPE
        from vllm.model_executor.layers.rotary_embedding import get_rope
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=32768,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Attention
        from vllm.attention import Attention, AttentionType
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )
        
        # U_c (R3) rotation for Q/K after RoPE
        self.register_buffer('rotation_R3', torch.empty(0))
        
        # O_proj input column reordering
        # msmodelslim's rearrange_o_proj reorders columns to [mid | high]
        # We need to reorder attn_output to match this layout
        self.register_buffer('o_proj_column_order', torch.empty(0, dtype=torch.long))
    
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Q/K/V projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # QK norm
        q = self.q_norm(q.reshape(*q.shape[:-1], -1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(*k.shape[:-1], -1, self.head_dim)).reshape(k.shape)
        
        # RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # U_c rotation after RoPE
        if self.rotation_R3.numel() > 0:
            q = apply_block_rotation(q, self.rotation_R3)
            k = apply_block_rotation(k, self.rotation_R3)
        
        # Attention
        attn_output = self.attn(q, k, v)
        
        # Reorder attn_output columns to match o_proj weight layout [mid | high]
        if self.o_proj_column_order.numel() > 0:
            attn_output = attn_output[..., self.o_proj_column_order]
        
        # O projection
        return self.o_proj(attn_output)


# ============================================================================
# MLP with ResQ
# ============================================================================

class Qwen3ResQTrueQuantMLP(nn.Module):
    """Qwen3 MLP with ResQ mixed-precision and U_d rotation."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Note: gate and up are separate in true quant (not fused)
        self.gate_proj = ResQMixedPrecisionLinear(hidden_size, intermediate_size)
        self.up_proj = ResQMixedPrecisionLinear(hidden_size, intermediate_size)
        self.down_proj = ResQMixedPrecisionLinear(intermediate_size, hidden_size)
        
        # U_d rotation parameters
        self.register_buffer('rotation_Pd', torch.empty(0))
        self.shared_Hd: Optional[torch.Tensor] = None
        self.shared_Hd_K: int = 1
        self.blocksize: int = 256
    
    def set_shared_hadamard(self, Hd: Optional[torch.Tensor], Hd_K: int, blocksize: int):
        self.shared_Hd = Hd
        self.shared_Hd_K = Hd_K
        self.blocksize = blocksize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        intermediate = torch.nn.functional.silu(gate) * up
        
        # U_d rotation before down_proj
        if self.rotation_Pd.numel() > 0:
            intermediate = apply_ud_rotation(
                intermediate,
                Pd=self.rotation_Pd,
                Hd=self.shared_Hd,
                K=self.shared_Hd_K,
                blocksize=self.blocksize,
            )
        
        return self.down_proj(intermediate)


# ============================================================================
# Decoder Layer
# ============================================================================

class Qwen3ResQTrueQuantDecoderLayer(nn.Module):
    """Qwen3 Decoder Layer with ResQ true quantization."""
    
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3ResQTrueQuantAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, 'rope_scaling', None),
            prefix=f"{prefix}.self_attn",
        )
        
        self.mlp = Qwen3ResQTrueQuantMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Set weight_loader for layer norm weights (like resq_fake_v2 pattern)
        def norm_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)
        setattr(self.input_layernorm.weight, "weight_loader", norm_weight_loader)
        setattr(self.post_attention_layernorm.weight, "weight_loader", norm_weight_loader)
    
    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


# ============================================================================
# Full Model
# ============================================================================

class Qwen3ResQTrueQuantForCausalLM(nn.Module):
    """
    Qwen3 with ResQ true quantization (direct int4/int8 weights).
    
    This model directly loads msmodelslim quantized weights (权重A format):
    - weight_low (int4), weight_high (int8)
    - scale_low, scale_high, offset_low, offset_high
    - rotation matrices Uc, Pd, Hd
    
    Usage:
        # No --quantization flag needed! The model handles quantization internally.
        vllm serve /path/to/resq_checkpoint --tensor-parallel-size 1 \\
            --model-type Qwen3ResQTrueQuantForCausalLM
    
    TP=1 only - for TP>1, use the fake quantization version.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            raise ValueError(
                f"Qwen3ResQTrueQuantForCausalLM only supports TP=1, got TP={tp_size}. "
                "For TP>1, use the fake quantization version with preprocessed bf16 weights."
            )
        
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.quant_config = quant_config
        
        # Embedding
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3ResQTrueQuantDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.model.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Set weight_loader for final norm (like resq_fake_v2 pattern)
        def norm_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)
        setattr(self.norm.weight, "weight_loader", norm_weight_loader)
        
        # LM head
        if get_pp_group().is_last_rank:
            from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.lm_head = PPMissingLayer()
        
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)
        
        # ResQ global parameters
        self.register_buffer('resq_Hd', torch.empty(0))
        self.resq_Hd_K = 1
        self.resq_blocksize = 256
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
    
    def _ckpt_to_model_key(self, ckpt_key: str) -> str:
        """Convert checkpoint key to model state_dict key.
        
        Checkpoint uses 'model.' prefix, model state_dict doesn't.
        e.g., 'model.layers.0.input_layernorm.weight' -> 'layers.0.input_layernorm.weight'
        """
        if ckpt_key.startswith('model.'):
            return ckpt_key[6:]  # Remove 'model.' prefix
        return ckpt_key
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load weights from msmodelslim ResQ checkpoint.
        
        Uses standard vLLM weight loading for regular parameters (norms, embed, lm_head)
        and custom loading for ResQ quantized linear layers.
        
        Returns None to skip vLLM's strict weight check (since we use custom quantization).
        """
        # Step 1: Get target device and move model FIRST
        # vllm-ascend requires torch_npu, no fallback needed
        import torch_npu
        target_device = torch.device(f'npu:{torch_npu.npu.current_device()}')
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ TrueQuant] target_device={target_device}")
        
        # Move entire model to target device before any weight loading
        # This ensures all parameters (including RMSNorm weights) are on NPU
        self.to(target_device)
        
        # DEBUG: Check device after self.to()
        if RESQ_DEBUG or True:  # Always log for debugging
            # Check a specific norm weight
            for name, param in self.named_parameters():
                if 'q_norm' in name or 'input_layernorm' in name:
                    logger.warning(f"[ResQ DEBUG] After self.to(): {name} device={param.device}")
                    break
        
        # Step 2: Build params_dict (now model is on correct device)
        params_dict = dict(self.named_parameters())
        
        # Step 3: Normalize keys and collect weights
        weights_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            key = self._ckpt_to_model_key(name)
            weights_dict[key] = tensor
        
        # Load global ResQ parameters (stored as attributes, not in state_dict)
        if 'resq.Hd' in weights_dict:
            self.resq_Hd = weights_dict['resq.Hd'].to(target_device)
        if 'resq.Hd_K' in weights_dict:
            self.resq_Hd_K = int(weights_dict['resq.Hd_K'].item())
        if 'resq.down_proj_blocksize' in weights_dict:
            self.resq_blocksize = int(weights_dict['resq.down_proj_blocksize'].item())
        
        # Load standard parameters using default_weight_loader (handles device correctly)
        # This includes: embed_tokens, all norms, lm_head
        for ckpt_key, loaded_weight in weights_dict.items():
            # Skip ResQ-specific keys
            if ckpt_key.startswith('resq.'):
                continue
            # Skip quantized linear layer keys (weight_low, weight_high, scale_*, offset_*)
            if any(suffix in ckpt_key for suffix in ['weight_low', 'weight_high', 'scale_low', 'scale_high', 'offset_low', 'offset_high']):
                continue
            
            # Map to param name
            param_name = ckpt_key
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                # DEBUG: Log norm weight loading
                if 'q_norm' in param_name or 'input_layernorm' in param_name:
                    logger.warning(f"[ResQ DEBUG] Loading {param_name}: param.device={param.device}, loaded_weight.device={loaded_weight.device}")
                weight_loader(param, loaded_weight)
                if 'q_norm' in param_name or 'input_layernorm' in param_name:
                    logger.warning(f"[ResQ DEBUG] After load {param_name}: param.device={param.device}")
        
        # Load layers - ResQ specific parts
        for i, layer in enumerate(self.layers):
            prefix = f'layers.{i}'
            
            # Load rotations (stored as attributes, move to device)
            uc_key = f'resq.layer.{i}.Uc'
            if uc_key in weights_dict:
                layer.self_attn.rotation_R3 = weights_dict[uc_key].to(target_device)
            
            pd_key = f'resq.layer.{i}.Pd'
            if pd_key in weights_dict:
                layer.mlp.rotation_Pd = weights_dict[pd_key].to(target_device)
            
            layer.mlp.set_shared_hadamard(self.resq_Hd, self.resq_Hd_K, self.resq_blocksize)
            
            # Initialize o_proj column reordering for mixed-precision layout
            # msmodelslim's rearrange_o_proj reorders columns to [mid | high]
            # 
            # BUG in msmodelslim: rearrange and quantize use different high_bits_length:
            # - rearrange_columns: high_bits_length = model_dim * high_fraction = 5120 * 0.125 = 640
            #   -> high_length_per_head = 640 / 64 = 10
            # - calibrator: high_bits_length = in_dim * high_fraction = 8192 * 0.125 = 1024
            #   -> high_length_per_head = 1024 / 64 = 16
            #
            # The actual column order in ckpt A follows rearrange logic (10 cols/head),
            # but weight_low/weight_high split follows quantize logic (16 cols/head).
            #
            # RESQ_O_PROJ_REARRANGE_FIX=1: Use rearrange logic to match actual column order
            # RESQ_O_PROJ_REARRANGE_FIX=0: Use quantize logic (original, incorrect)
            
            head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else self.config.hidden_size // self.config.num_attention_heads
            num_attention_heads = self.config.num_attention_heads
            hidden_size = self.config.hidden_size
            in_dim = num_attention_heads * head_dim  # o_proj input dim
            high_fraction = 0.125
            
            if RESQ_O_PROJ_REARRANGE_FIX:
                # Use rearrange logic: high_bits_length based on model_dim (hidden_size)
                high_bits_length = int(hidden_size * high_fraction)  # 640 for hidden_size=5120
                high_length_per_head = high_bits_length // num_attention_heads  # 10
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] o_proj REARRANGE_FIX enabled: "
                                   f"high_bits_length={high_bits_length} (from model_dim={hidden_size}), "
                                   f"high_length_per_head={high_length_per_head}")
            else:
                # Original logic: high_bits_length based on in_dim (incorrect for rearranged weights)
                high_bits_length = int(in_dim * high_fraction)  # 1024 for in_dim=8192
                high_length_per_head = high_bits_length // num_attention_heads  # 16
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] o_proj using quantize logic: "
                                   f"high_bits_length={high_bits_length} (from in_dim={in_dim}), "
                                   f"high_length_per_head={high_length_per_head}")
            
            low_length_per_head = head_dim - high_length_per_head
            
            # Build column reorder: original -> [mid | high]
            # Original layout: [head0_all, head1_all, ...] where head_all has head_dim columns
            # Target layout: [all_mid, all_high] where:
            #   - mid = first (head_dim - high_length_per_head) columns of each head
            #   - high = last high_length_per_head columns of each head
            column_order = []
            # First: all mid parts (first low_length_per_head of each head)
            for h in range(num_attention_heads):
                base = h * head_dim
                for j in range(low_length_per_head):
                    column_order.append(base + j)
            # Then: all high parts (last high_length_per_head of each head)
            for h in range(num_attention_heads):
                base = h * head_dim + low_length_per_head
                for j in range(high_length_per_head):
                    column_order.append(base + j)
            layer.self_attn.o_proj_column_order = torch.tensor(column_order, dtype=torch.long, device=target_device)
            
            # Load attention projections (quantized)
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name)
                proj_prefix = f'{prefix}.self_attn.{proj_name}'
                self._load_resq_linear(proj, proj_prefix, weights_dict, target_device)
            
            # Load MLP projections (quantized)
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, proj_name)
                proj_prefix = f'{prefix}.mlp.{proj_name}'
                self._load_resq_linear(proj, proj_prefix, weights_dict, target_device)
        

        # Final device sync at end of load_weights
        self.to(target_device)
        
        # Debug: verify device
        for name, param in self.named_parameters():
            if 'q_norm' in name:
                logger.warning(f"[ResQ] FINAL load_weights: {name} device={param.device}")
                break
        if RESQ_DEBUG:
            logger.warning(f"[ResQ TrueQuant] Loaded weights, Hd_K={self.resq_Hd_K}, blocksize={self.resq_blocksize}")
    
    def _load_resq_linear(
        self,
        linear: ResQMixedPrecisionLinear,
        prefix: str,
        weights_dict: Dict[str, torch.Tensor],
        target_device: torch.device,
    ) -> None:
        """Load weights for a ResQ mixed-precision linear layer."""
        for suffix in ['weight_low', 'weight_high', 'scale_low', 'scale_high', 'offset_low', 'offset_high']:
            key = f'{prefix}.{suffix}'
            if key in weights_dict:
                tensor = weights_dict[key].to(target_device)
                buffer = getattr(linear, suffix)
                if buffer.numel() == 0 or buffer.shape != tensor.shape:
                    linear.register_buffer(suffix, tensor)
                else:
                    buffer.copy_(tensor)

        # Update in_low/in_high from actual loaded weight shapes
        # This handles the rearrange/quantize mismatch in msmodelslim
        if linear.weight_low.numel() > 0:
            linear.in_low = linear.weight_low.shape[1]
        if linear.weight_high.numel() > 0:
            linear.in_high = linear.weight_high.shape[1]
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ] {prefix}: in_low={linear.in_low}, in_high={linear.in_high}")
