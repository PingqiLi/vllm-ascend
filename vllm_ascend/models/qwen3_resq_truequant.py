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

def resq_log(msg: str):
    print(msg, flush=True)
    log_file_env = os.environ.get("RESQ_LOG_FILE")
    if log_file_env:
        root, ext = os.path.splitext(log_file_env)
        log_file = f"{root}_resqv1{ext}"
        try:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"WARN: Failed to write to {log_file}: {e}")

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
UB_FUSED = os.environ.get("UB_FUSED", "1") == "1"

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
import torch_npu

# ============================================================================
# Helper Functions from Debug Script
# ============================================================================

def pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """
    x: int8 tensor, shape (E, K, N)，值域 ∈ [-8, 7]
    return: int8 tensor, shape (E, K, N/2)，每个元素打包两个有符号 int4
    """
    assert x.dtype == torch.int8
    K, N = x.shape
    assert N % 2 == 0
    
    # 转成无符号补码 [0, 15]
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]   # 偶数 -> 低 4 位
    high = x_unsigned[..., 1::2]  # 奇数 -> 高 4 位

    out = (low | (high << 4)).to(torch.int8)
    return out


# ============================================================================
# Mixed-Precision Linear Layer
# ============================================================================

class ResQMixedPrecisionLinear(nn.Module):
    """
    Linear layer with ResQ mixed-precision quantization.
    Logic matches tools/resq_debug/modeling_qwen3_resq_truequant.py (ResQTrueQuantLinear).
    
    Stores int8 weights, unpacks/converts during forward, and uses manual NPU/CPU quantization kernels.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        high_fraction: float = 0.125,
        bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.in_features = in_features
        self.out_features = out_features
        self.high_fraction = high_fraction
        
        # Calculate expected dims for validation
        self.in_high = int(in_features * high_fraction)
        self.in_low = in_features - self.in_high
        
        # Initialize as empty tensors to save memory until load_weights
        # This avoids OOM caused by having both placeholder and loaded weight in GPU memory simultaneously
        self.register_buffer('weight_low', torch.empty(0, dtype=torch.int8))
        self.register_buffer('weight_high', torch.empty(0, dtype=torch.int8))
        self.register_buffer('scale_low', torch.empty(0, dtype=torch.float32))
        self.register_buffer('scale_high', torch.empty(0, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward using logic from ResQTrueQuantLinear (True Quantization).
        """
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features).float()
        M = x_2d.shape[0]
        
        # Split input
        x_low = x_2d[:, :self.in_low].to(torch.float16).npu()
        x_high = x_2d[:, self.in_low:].to(torch.float16).npu()
        
        # NPU implement
        # Use quint4x2 for low part to match debug script behavior
        x_low_abs_max, lxScale = torch_npu.npu_dynamic_quant(x_low, dst_type=torch.quint4x2)
        x_high_abs_max, rxScale = torch_npu.npu_dynamic_quant(x_high, dst_type=torch.int8)
        
        # Pack weight_low (which is stored as int8) to int4x2 format needed by npu_quant_matmul
        weight_low = pack_int4_to_int8_signed(self.weight_low)
        weight_low = weight_low.view(torch.int32).transpose(-1, -2).npu()
        
        output_low = torch_npu.npu_quant_matmul(
            x_low_abs_max, 
            weight_low, 
            self.scale_low.to(torch.float).npu(), 
            pertoken_scale=lxScale, 
            output_dtype=torch.float16
        )

        weight_nz_high = torch_npu.npu_format_cast(self.weight_high.npu(), 29).transpose(-1,-2)
        output_high = torch_npu.npu_quant_matmul(
            x_high_abs_max, 
            weight_nz_high, 
            self.scale_high.to(torch.float).npu(), 
            pertoken_scale=rxScale, 
            output_dtype=torch.float16
        )

        output = torch.add(output_low, output_high)
        
        if RESQ_DEBUG and "layers.0." in self.prefix:
            resq_log(f"DEBUG [Ref] {self.prefix} Output Low mean: {output_low.float().mean()}")
            resq_log(f"DEBUG [Ref] {self.prefix} Output High mean: {output_high.float().mean()}")
            resq_log(f"DEBUG [Ref] {self.prefix} Final Output mean: {output.float().mean()}")

        if self.bias is not None:
            output = output + self.bias
        
        # Restore shape to match input (vLLM passes [num_tokens, hidden_size])
        output_shape = list(original_shape[:-1]) + [self.out_features]
        # Convert to bfloat16 to match RMSNorm weights dtype
        # NPU's npu_rms_norm requires x and gamma to have compatible dtypes
        # Supported combos: (fp16,fp16), (bf16,bf16), (fp16,fp32), (bf16,fp32), (fp32,fp32)
        # Since model weights are loaded as bfloat16, output must also be bfloat16
        return output.view(output_shape).to(torch.bfloat16)



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
    h_butterfly: Optional[torch.Tensor], # Add pre-computed matrix
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
    
    # Step 1: Apply block_diag(Pd) block-wise (x @ Pd for each block)
    # Matches verified debug script: Pd_f32 = self.Pd; x = torch.matmul(x, Pd_f32)
    # Pd is [blocksize, blocksize], x is [..., K, blocksize]
    # x @ Pd operates on the last dim (blocksize) -> [..., K, blocksize]
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = torch.matmul(x, Pd_f32)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    # First apply butterfly Hadamard on the blocksize dimension (last dim)
    
    # Optimization: Use pre-computed H_butterfly matrix if available
    # x @ h_butterfly is equivalent to hadamard_transform(x) along last dim
    if h_butterfly is not None:
        h_butterfly_f32 = h_butterfly.to(device=x.device, dtype=torch.float32)
        x = torch.matmul(x, h_butterfly_f32)
    else:
        # Fallback to loop if not pre-computed (shouldn't happen with correct initialization)
        x = hadamard_transform(x.contiguous())
    
    # Then apply Hd on the K dimension (second-to-last dim)
    # Reference: x @ Hd.T (right-multiply with Hd transpose)
    # x: [batch, K, blocksize], Hd: [K, K]
    if Hd is not None and K > 1:
        batch_shape = x.shape[:-2]
        batch_size = 1
        for d in batch_shape:
            batch_size *= d
        x = x.reshape(batch_size, K, blocksize)
        
        Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
        # Match reference: transpose, right-multiply Hd.T, transpose back
        # x = x.transpose(-1, -2) @ Hd.T then transpose back
        x = x.transpose(-1, -2)  # [batch, blocksize, K]
        x = torch.matmul(x, Hd_f32.t())  # [batch, blocksize, K] @ [K, K]
        x = x.transpose(-1, -2)  # [batch, K, blocksize]
        
        x = x.reshape(*batch_shape, K, blocksize)
    
    # Normalize: divide by sqrt(n) AND multiply by K (matching debug script)
    # Debug script: x = x * self.K / math.sqrt(n)
    x = x * K / math.sqrt(n)
    
    # Note: msmodelslim now correctly uses K * hadK in weight fusion (matching original paper).
    # No additional compensation needed here - the Ud rotation is now a proper orthogonal transform.
    
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
        self.q_proj = ResQMixedPrecisionLinear(hidden_size, self.q_size, prefix=f"{prefix}.q_proj")
        self.k_proj = ResQMixedPrecisionLinear(hidden_size, self.kv_size, prefix=f"{prefix}.k_proj")
        self.v_proj = ResQMixedPrecisionLinear(hidden_size, self.kv_size, prefix=f"{prefix}.v_proj")
        self.o_proj = ResQMixedPrecisionLinear(self.q_size, hidden_size, prefix=f"{prefix}.o_proj")
        
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
        # if self.rotation_R3.numel() > 0:
        #     q = apply_block_rotation(q, self.rotation_R3)
        #     k = apply_block_rotation(k, self.rotation_R3)
        
        # Attention
        attn_output = self.attn(q, k, v)
        
        # Reorder attn_output columns to match o_proj weight layout [mid | high]
        # Controlled by UB_FUSED (default=True). 
        # If True: weights fused with Ub -> Requires reordering.
        # If False: weights not fused -> Skip reordering.
        if self.o_proj_column_order.numel() > 0 and UB_FUSED:
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
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Note: gate and up are separate in true quant (not fused)
        self.gate_proj = ResQMixedPrecisionLinear(hidden_size, intermediate_size, prefix=f"{prefix}.gate_proj")
        self.up_proj = ResQMixedPrecisionLinear(hidden_size, intermediate_size, prefix=f"{prefix}.up_proj")
        self.down_proj = ResQMixedPrecisionLinear(intermediate_size, hidden_size, prefix=f"{prefix}.down_proj")
        
        # U_d rotation parameters
        self.register_buffer('rotation_Pd', torch.empty(0))
        # H_butterfly buffer for optimization
        self.register_buffer('h_butterfly', torch.empty(0))
        
        self.shared_Hd: Optional[torch.Tensor] = None
        self.shared_Hd_K: int = 1
        self.blocksize: int = 256
    
    def set_shared_hadamard(self, Hd: Optional[torch.Tensor], Hd_K: int, blocksize: int):
        self.shared_Hd = Hd
        self.shared_Hd_K = Hd_K
        self.blocksize = blocksize
        
        # Pre-compute H_butterfly matrix if not already waiting
        if self.h_butterfly.numel() == 0 or self.h_butterfly.shape[0] != blocksize:
            # Create identity matrix and apply transform to get the matrix form
            eye = torch.eye(blocksize, dtype=torch.float32)
            # We need to apply transform to columns, so we can use the existing function
            # or just apply to rows since it's symmetric
            h_matrix = hadamard_transform(eye)
            self.h_butterfly = h_matrix.to(self.rotation_Pd.device if self.rotation_Pd.numel() > 0 else 'cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        intermediate = torch.nn.functional.silu(gate) * up
        
        # U_d rotation before down_proj
        # Match reference: check both Pd and Hd exist
        if self.rotation_Pd.numel() > 0 and self.shared_Hd is not None:
            # Ensure h_butterfly is on the correct device
            if self.h_butterfly.device != x.device:
                self.h_butterfly = self.h_butterfly.to(x.device)

            intermediate = apply_ud_rotation(
                intermediate,
                Pd=self.rotation_Pd,
                Hd=self.shared_Hd,
                h_butterfly=self.h_butterfly,
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
            prefix=f"{prefix}.mlp",
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

class Qwen3ResQForCausalLM(nn.Module):
    """
    Qwen3 with ResQ true quantization (direct int4/int8 weights).
    
    This model directly loads msmodelslim quantized weights (权重A format):
    - weight_low (int4), weight_high (int8)
    - scale_low, scale_high, offset_low, offset_high
    - rotation matrices Uc, Pd, Hd
    
    Usage:
        # No --quantization flag needed! The model handles quantization internally.
        vllm serve /path/to/resq_checkpoint --tensor-parallel-size 1 \\
            --model-type Qwen3ResQForCausalLM
    
    TP=1 only - for TP>1, use the fake quantization version.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            raise ValueError(
                f"Qwen3ResQForCausalLM only supports TP=1, got TP={tp_size}. "
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
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from msmodelslim ResQ checkpoint.
        
        Strategy: Load EVERYTHING into CPU memory first, then move to NPU at the end.
        This replicates the behavior of the debug script and avoids NPU memory fragmentation.
        Assumes System RAM > 33GB.
        """
        import torch_npu
        
        # We will move to this device ONLY at the very end
        final_device = torch.device(f'npu:{torch_npu.npu.current_device()}')
        
        # Ensure model is on CPU to start with (should be default, but be safe)
        if RESQ_DEBUG:
            logger.warning(f"[ResQ TrueQuant] Moving model to CPU for loading...")
        self.cpu()
        
        # Pre-calculate o_proj column order (on CPU)
        for layer in self.layers:
            self._setup_o_proj_column_order(layer.self_attn, device=torch.device('cpu'))

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        # ResQ doesn't use standard stacked params (qkv, gate_up) in existing impl, 
        # but we keep the structure for consistency if needed later.
        stacked_params_mapping = [] 

        for name, loaded_weight in weights:
            # Ensure loaded_weight is on CPU (it should be coming from safetensors/loader)
            loaded_weight = loaded_weight.cpu()
            
            # --- 1. Global & ResQ Specific Handling ---
            if name == 'resq.Hd':
                self.resq_Hd = loaded_weight
                for layer in self.layers:
                    layer.mlp.set_shared_hadamard(self.resq_Hd, self.resq_Hd_K, self.resq_blocksize)
                loaded_params.add(name)
                continue
            
            if name == 'resq.Hd_K':
                self.resq_Hd_K = int(loaded_weight.item())
                for layer in self.layers:
                    layer.mlp.set_shared_hadamard(self.resq_Hd, self.resq_Hd_K, self.resq_blocksize)
                loaded_params.add(name)
                continue
                
            if name == 'resq.down_proj_blocksize':
                self.resq_blocksize = int(loaded_weight.item())
                for layer in self.layers:
                    layer.mlp.set_shared_hadamard(self.resq_Hd, self.resq_Hd_K, self.resq_blocksize)
                loaded_params.add(name)
                continue
                
            if name.startswith('resq.layer.'):
                try:
                    parts = name.split('.')
                    layer_idx = int(parts[2])
                    param_type = parts[3]
                    if param_type == 'Uc':
                        self.layers[layer_idx].self_attn.rotation_R3 = loaded_weight
                    elif param_type == 'Pd':
                        self.layers[layer_idx].mlp.rotation_Pd = loaded_weight
                    loaded_params.add(name)
                except (IndexError, ValueError) as e:
                    logger.warning(f"Failed to parse ResQ key {name}: {e}")
                continue

            # --- 2. Quantized Linear Weights Custom Load ---
            # Remove 'model.' prefix for strict matching
            model_key = self._ckpt_to_model_key(name)
            
            if any(suffix in model_key for suffix in ['weight_low', 'weight_high', 'scale_low', 'scale_high']):
                self._load_quant_weight_streaming(model_key, loaded_weight)
                loaded_params.add(name)
                continue

            # --- 3. Standard Weights with Mapping Support ---
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Logic for stacked params would go here if needed
                break
            else:
                # Standard parameter loading
                if model_key not in params_dict:
                    if not any(s in model_key for s in ['offset_low', 'offset_high']):
                         pass 
                    continue
                
                param = params_dict[model_key]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # Both param and loaded_weight are on CPU, direct copy safe
                weight_loader(param, loaded_weight)
                loaded_params.add(model_key)
        
        # --- FINAL STEP: Move everything to NPU ---
        if RESQ_DEBUG:
            logger.warning(f"[ResQ TrueQuant] All weights loaded to CPU. Moving model to {final_device}...")
        
        self.to(final_device)
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ TrueQuant] Model successfully moved to NPU.")
            
        return loaded_params

    def _load_quant_weight_streaming(self, key: str, tensor: torch.Tensor):
        """Streaming load for a single quantized weight tensor (CPU mode)."""
        parts = key.split('.')
        try:
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            
            if not isinstance(module, ResQMixedPrecisionLinear):
                return
            
            suffix = parts[-1] 
            buffer = getattr(module, suffix)
            
            # Ensure correct dtype
            if suffix.startswith('weight'):
                tensor = tensor.to(torch.int8)
            else:
                tensor = tensor.float().squeeze()
                if tensor.dim() > 1:
                    tensor = tensor.flatten()
            
            if buffer.numel() == 0 or buffer.shape != tensor.shape:
                # Overwrite buffer with new CPU tensor
                setattr(module, suffix, tensor)
                
                if suffix == 'weight_low':
                    module.in_low = tensor.shape[1]
                elif suffix == 'weight_high':
                    module.in_high = tensor.shape[1]
            else:
                buffer.copy_(tensor)

        except AttributeError:
            logger.warning(f"Could not load quantized weight: {key}")

    def _setup_o_proj_column_order(self, attn, device):
        """Setup o_proj column reorder (matches debug script logic)"""
        head_dim = attn.head_dim
        num_heads = attn.num_heads
        in_dim = num_heads * head_dim
        high_fraction = 0.125
        
        high_bits_length = int(in_dim * high_fraction)
        high_per_head = high_bits_length // num_heads
        
        chunk_starts = torch.arange(0, in_dim, head_dim, device=device)
        high_precision_columns = torch.arange(head_dim - high_per_head, head_dim, device=device)
        
        columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()
        
        all_columns = torch.arange(in_dim, device=device)
        mask = torch.ones(in_dim, dtype=torch.bool, device=device)
        mask[columns_to_end] = False
        remaining_columns = all_columns[mask]
        
        new_column_order = torch.cat([remaining_columns, columns_to_end])
        
        if attn.o_proj_column_order.numel() == 0 or attn.o_proj_column_order.shape != new_column_order.shape:
             attn.o_proj_column_order = new_column_order
        else:
             attn.o_proj_column_order.copy_(new_column_order)
