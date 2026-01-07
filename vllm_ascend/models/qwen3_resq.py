"""
Qwen3 ResQ Model Definition (Simplified for Preprocessed BF16 Weights)

This module defines Qwen3 model variants with ResQ rotation support.
Assumes weights have been preprocessed to bf16 using preprocess_resq_weights.py.

Key differences from standard Qwen3:
1. Qwen3ResQAttention: Applies R3 rotation to Q/K after RoPE
2. Qwen3ResQMLP: Applies Hadamard rotation before down_proj
3. Optional W4A4 fake quantization for accuracy testing

Usage:
    # Without fake quantization (for debugging rotations):
    RESQ_FAKE_QUANT=0 vllm serve ...
    
    # With fake quantization (for W4A4 accuracy testing, default):
    RESQ_FAKE_QUANT=1 vllm serve ...
"""
from typing import Iterable, Optional, Tuple
import math
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
RESQ_SKIP_ROTATION = os.environ.get("RESQ_SKIP_ROTATION", "0") == "1"  # Skip all rotation for debugging
RESQ_SKIP_PD = os.environ.get("RESQ_SKIP_PD", "0") == "1"  # Skip Pd, only apply Hadamard
RESQ_SKIP_HADAMARD = os.environ.get("RESQ_SKIP_HADAMARD", "0") == "1"  # Skip Hadamard, only apply Pd
RESQ_SKIP_R3 = os.environ.get("RESQ_SKIP_R3", "0") == "1"  # Skip R3/Uc rotation in Attention (Q/K post-RoPE)
RESQ_HIGH_BITS = int(os.environ.get("RESQ_HIGH_BITS", "8"))
RESQ_LOW_BITS = int(os.environ.get("RESQ_LOW_BITS", "4"))
RESQ_HIGH_FRACTION = float(os.environ.get("RESQ_HIGH_FRACTION", "0.125"))
RESQ_LOG_FILE = os.environ.get("RESQ_LOG_FILE", "/tmp/resq_debug.log")  # Debug log file path

# Debug log file handle (only rank 0 writes)
_resq_log_file = None

def resq_log(msg: str, rank: int = 0, all_ranks: bool = False) -> None:
    """Write debug message to per-rank log file.
    
    Each rank writes to its own file: /tmp/resq_debug_rank{N}.log
    
    Args:
        msg: Message to log
        rank: Rank filter - only log if current rank matches (unless all_ranks=True)
        all_ranks: If True, log from all ranks
    """
    global _resq_log_file
    if not RESQ_DEBUG:
        return
    
    from vllm.distributed import get_tensor_model_parallel_rank
    current_rank = get_tensor_model_parallel_rank()
    
    if not all_ranks and rank != current_rank:
        return
    
    if _resq_log_file is None:
        try:
            # Each rank gets its own log file
            log_path = RESQ_LOG_FILE.replace('.log', f'_rank{current_rank}.log')
            if not log_path.endswith(f'_rank{current_rank}.log'):
                log_path = f"{RESQ_LOG_FILE}_rank{current_rank}"
            _resq_log_file = open(log_path, 'w')
            _resq_log_file.write(f"=== ResQ Debug Log (Rank {current_rank}) ===\n\n")
        except Exception as e:
            logger.warning(f"[ResQ] Failed to open log file: {e}")
            return
    
    try:
        _resq_log_file.write(msg + "\n")
        _resq_log_file.flush()
    except Exception:
        pass


# ============================================================================
# Hadamard Transform Utilities
# ============================================================================

def is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """
    Fast Hadamard transform using butterfly algorithm.
    
    Works on CPU/NPU without CUDA kernels.
    
    Args:
        u: Input tensor with last dimension being power of 2
    
    Returns:
        Hadamard transformed tensor (unnormalized)
    """
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


def matmul_hadU(x: torch.Tensor, hadK: Optional[torch.Tensor], K: int) -> torch.Tensor:
    """
    Apply structured Hadamard transform.
    
    Computes X @ (hadK ⊗ H)^T where H is Hadamard of n/K dimension.
    
    For n = K * 2^m:
    - hadK is [K, K] block matrix
    - H_butterfly is [2^m, 2^m] Hadamard via butterfly algorithm
    - Result is X @ (hadK ⊗ H_butterfly)
    
    Note: This function auto-detects if hadK is normalized (elements ~±1/sqrt(K))
    or unnormalized (elements ~±1) and adjusts normalization accordingly.
    
    Args:
        x: Input tensor of shape [..., n]
        hadK: Block matrix [K, K] (may be None for pure power-of-2)
        K: Block size
    
    Returns:
        Transformed tensor
    """
    n = x.shape[-1]
    blocksize = n // K if K > 1 else n
    
    if K == 1 or hadK is None:
        # Pure power-of-2: use butterfly Hadamard only
        return hadamard_transform(x.contiguous()) / math.sqrt(n)
    
    # Check if hadK is already normalized
    hadk_max = hadK.abs().max().item()
    hadk_is_normalized = hadk_max < 0.5  # If max < 0.5, it's normalized
    
    # Reshape to apply block-wise transform: [..., n] -> [-1, K, n/K]
    original_shape = x.shape
    input_tensor = x.reshape(-1, K, blocksize)
    
    # Apply fast Hadamard to the blocksize dimension (butterfly algorithm)
    input_tensor = hadamard_transform(input_tensor.contiguous())
    
    # Normalization depends on whether hadK is already normalized
    if hadk_is_normalized:
        # hadK already has 1/sqrt(K) normalization, only need 1/sqrt(blocksize) for H_butterfly
        input_tensor = input_tensor / math.sqrt(blocksize)
    else:
        # hadK is unnormalized, need full 1/sqrt(n) normalization
        input_tensor = input_tensor / math.sqrt(n)
    
    # Apply hadK block matrix: [K, K] @ [-1, K, blocksize]
    hadK = hadK.to(device=input_tensor.device, dtype=input_tensor.dtype)
    input_tensor = hadK @ input_tensor
    
    return input_tensor.reshape(original_shape)


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
    Uses float32 for rotation to maintain precision.
    """
    if rotation_matrix is None:
        return x
    
    original_dtype = x.dtype
    K = rotation_matrix.shape[0]  # Block size (e.g., 128 or 256)
    
    # Use float32 for rotation precision
    R = rotation_matrix.to(device=x.device, dtype=torch.float32)
    x_f32 = x.float()
    
    original_shape = x.shape
    N = original_shape[-1]
    
    if N == K:
        return torch.matmul(x_f32, R).to(original_dtype)
    
    if N % K != 0:
        raise ValueError(f"Feature dim {N} must be divisible by rotation block size {K}")
    
    num_blocks = N // K
    x_blocked = x_f32.reshape(*original_shape[:-1], num_blocks, K)
    x_rotated = torch.matmul(x_blocked, R)
    return x_rotated.reshape(*original_shape).to(original_dtype)


def apply_resq_hadamard_rotation_tp(
    x: torch.Tensor,
    Pd: Optional[torch.Tensor] = None,
    Hd: Optional[torch.Tensor] = None,
    Hd_K: int = 1,
    blocksize: int = 256,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply ResQ Hadamard rotation with Tensor Parallelism support.
    
    Weight transformation in msmodelslim:
        W'_d = Ua.T @ Wd @ block_diag(Pd.T) @ H
    
    For correct forward pass with TP:
        1. Apply block_diag(Pd.T) locally (each block is independent)
        2. Apply H_butterfly locally to each blocksize-dim block
        3. All-gather across TP ranks
        4. Apply Hd to mix all K blocks
        5. Slice back to local portion
    
    Args:
        x: Input tensor [..., local_intermediate_size]
        Pd: Block rotation matrix [blocksize, blocksize]
        Hd: Hadamard block matrix [K, K]
        Hd_K: Global K for Hadamard factorization
        blocksize: Block size for Pd (e.g., 256)
        tp_size: Tensor parallelism size
        tp_rank: Current TP rank
    
    Returns:
        Rotated tensor [..., local_intermediate_size]
    """
    if Pd is None and Hd is None:
        return x
    
    # Skip rotation entirely for debugging
    if RESQ_SKIP_ROTATION:
        return x
    
    original_shape = x.shape
    n_local = original_shape[-1]  # local intermediate_size
    num_local_blocks = n_local // blocksize
    
    # Debug: track input stats
    if RESQ_DEBUG:
        x_in_stats = f"min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}"
    
    # Step 1: Apply block_diag(Pd.T) locally
    # Each blocksize-dim block is independent, so this works with TP
    # Use float32 for rotation to avoid bfloat16 precision loss
    if Pd is not None and not RESQ_SKIP_PD:
        original_dtype = x.dtype
        Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
        x_f32 = x.float()
        
        # Reshape: [..., n_local] -> [..., num_local_blocks, blocksize]
        x_f32 = x_f32.reshape(*original_shape[:-1], num_local_blocks, blocksize)
        # Apply Pd.T to each block
        x_f32 = torch.matmul(x_f32, Pd_f32.T)
        # Reshape back and convert to original dtype
        x = x_f32.reshape(*original_shape).to(original_dtype)
        
        if RESQ_DEBUG:
            x_after_pd_stats = f"min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}"
            # Log per-rank stats after Pd.T (all ranks write)
            x_norm_after_pd = x.norm().item()
            resq_log(f"  [Rank {tp_rank}] After Pd.T: norm={x_norm_after_pd:.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}", all_ranks=True)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly with TP handling
    # Use float32 for all Hadamard operations to maintain precision
    #
    # IMPORTANT: msmodelslim's Hd is NORMALIZED by 1/sqrt(K), but original ResQ's Hd is not!
    # This means msmodelslim's matmul_hadU_cpu has an extra 1/sqrt(K) scaling factor.
    # We need to multiply by sqrt(K) to compensate.
    #
    if Hd is not None and Hd_K > 1 and not RESQ_SKIP_HADAMARD:
        # H = Hd ⊗ H_butterfly where:
        #   - H_butterfly is [blocksize, blocksize] applied to each block locally
        #   - Hd is [K, K] mixing all K blocks (requires all-gather for TP)
        
        original_dtype = x.dtype
        x = x.float()  # Convert to float32 for precision
        
        # First, apply H_butterfly locally to each blocksize-dim block
        x = x.reshape(*original_shape[:-1], num_local_blocks, blocksize)
        x = hadamard_transform(x.contiguous())  # Apply to last dim (blocksize)
        
        # CRITICAL: msit's matmul_hadU_cpu divides by sqrt(n_global) regardless of whether
        # Hd is normalized. The normalized Hd (elements ±1/sqrt(K)) then provides an 
        # additional 1/sqrt(K) factor, so the complete normalization is:
        # - HadamardTransform / sqrt(n_global) * Hd_normalized = H_full / sqrt(K)
        # To match msit exactly, we must also divide by sqrt(n_global)
        n_global = n_local * tp_size
        x = x / math.sqrt(n_global)
        
        if RESQ_DEBUG:
            x_after_hbutterfly_stats = f"min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}"
            # Log per-rank stats to debug TP differences (all ranks write)
            x_norm_local = x.norm().item()
            resq_log(f"  [Rank {tp_rank}] After H_butterfly: norm={x_norm_local:.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}", all_ranks=True)
        
        if tp_size > 1:
            # All-gather across TP ranks to get full [..., K, blocksize] tensor
            from vllm.distributed.communication_op import tensor_model_parallel_all_gather
            
            # x shape: [..., num_local_blocks, blocksize]
            # Flatten batch dimensions for all-gather
            batch_shape = x.shape[:-2]
            batch_size = 1
            for dim in batch_shape:
                batch_size *= dim
            x_flat = x.reshape(batch_size, num_local_blocks, blocksize)
            
            # All-gather along the blocks dimension (dim=1)
            # Input: [batch, local_blocks, blocksize]
            # Output: [batch, K, blocksize] where K = local_blocks * tp_size
            gathered = tensor_model_parallel_all_gather(x_flat, dim=1)
            
            if RESQ_DEBUG:
                gathered_stats = f"shape={gathered.shape}, min={gathered.min().item():.4f}, max={gathered.max().item():.4f}"
            
            # Apply Hd to mix all K blocks (use float32)
            Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
            # Hd: [K, K], gathered: [batch, K, blocksize]
            # mixed[b, i, k] = sum_j Hd[i, j] * gathered[b, j, k]
            mixed = torch.einsum('ij,bjk->bik', Hd_f32, gathered)
            
            if RESQ_DEBUG:
                mixed_stats = f"shape={mixed.shape}, min={mixed.min().item():.4f}, max={mixed.max().item():.4f}"
            
            # Compensate for msmodelslim's extra 1/sqrt(K) normalization on Hd
            # Original ResQ: Hd elements are ±1, msmodelslim: Hd elements are ±1/sqrt(K)
            # BOTH weights and activations were scaled by 1/sqrt(K), so we need to multiply by K
            # to compensate for the total 1/K scaling in the output
            mixed = mixed * Hd_K
            
            # Slice back to local portion
            blocks_per_rank = Hd_K // tp_size
            start_block = tp_rank * blocks_per_rank
            end_block = start_block + blocks_per_rank
            x = mixed[:, start_block:end_block, :].contiguous()  # Make contiguous after slice
            
            if RESQ_DEBUG:
                x_sliced_stats = f"shape={x.shape}, slice=[{start_block}:{end_block}], min={x.min().item():.4f}, max={x.max().item():.4f}"
            
            # Reshape back to original batch dimensions
            x = x.reshape(*batch_shape, num_local_blocks, blocksize)
        else:
            # No TP, apply Hd directly (use float32)
            Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
            # x: [..., K, blocksize], Hd: [K, K]
            # Reshape for matmul: [..., K, blocksize] -> [batch, K, blocksize]
            batch_shape = x.shape[:-2]
            batch_size = 1
            for dim in batch_shape:
                batch_size *= dim
            x = x.reshape(batch_size, Hd_K, blocksize)
            x = torch.einsum('ij,bjk->bik', Hd_f32, x)
            
            # Compensate for msmodelslim's extra 1/sqrt(K) normalization on Hd
            # BOTH weights and activations were scaled by 1/sqrt(K), so multiply by K
            x = x * Hd_K
            
            x = x.reshape(*batch_shape, Hd_K, blocksize)
        
        # Convert back to original dtype
        x = x.to(original_dtype)
        
        # Reshape back to [..., n_local]
        x = x.reshape(*original_shape)
    
    elif Hd_K == 1 or Hd is None:
        # Pure power-of-2: just apply butterfly Hadamard locally
        n_global = n_local * tp_size
        x = hadamard_transform(x.contiguous()) / math.sqrt(n_global)
    
    # Debug: print rotation stats (only rank 0, write to file)
    if RESQ_DEBUG and tp_rank == 0:
        global _RESQ_ROTATION_STATS_LOGGED
        if not _RESQ_ROTATION_STATS_LOGGED:
            _RESQ_ROTATION_STATS_LOGGED = True
            resq_log(f"[MLP Hadamard Rotation] tp_rank={tp_rank}, tp_size={tp_size}", rank=0)
            resq_log(f"  Input: {x_in_stats}", rank=0)
            if Pd is not None:
                resq_log(f"  After Pd.T: {x_after_pd_stats}", rank=0)
                pd_orth_err = (Pd @ Pd.T - torch.eye(Pd.shape[0], device=Pd.device, dtype=Pd.dtype)).abs().max().item()
                resq_log(f"  Pd orthogonality: |Pd @ Pd.T - I|_max = {pd_orth_err:.6f}", rank=0)
            if Hd is not None and Hd_K > 1:
                hd_max = Hd.abs().max().item()
                resq_log(f"  Hd stats: shape={Hd.shape}, max_abs={hd_max:.4f}", rank=0)
                resq_log(f"  Normalization: 1/sqrt(n_global) = 1/sqrt({n_local * tp_size}) to match msit", rank=0)
                resq_log(f"  After H_butterfly: {x_after_hbutterfly_stats}", rank=0)
                if tp_size > 1:
                    resq_log(f"  Gathered: {gathered_stats}", rank=0)
                    resq_log(f"  After Hd: {mixed_stats}", rank=0)
                    resq_log(f"  Sliced: {x_sliced_stats}", rank=0)
            resq_log(f"  Output: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}", rank=0)
    
    return x


_RESQ_ROTATION_DEBUG_LOGGED = False
_RESQ_ROTATION_STATS_LOGGED = False
_RESQ_R3_DEBUG_LOGGED = False
_RESQ_MLP_DEBUG_LOGGED = False

def apply_resq_hadamard_rotation(
    x: torch.Tensor,
    Pd: Optional[torch.Tensor] = None,
    Hd: Optional[torch.Tensor] = None,
    Hd_K: int = 1,
) -> torch.Tensor:
    """
    Apply ResQ Hadamard rotation for inference (TP-aware version).
    
    Detects TP configuration automatically and routes to appropriate implementation.
    """
    global _RESQ_ROTATION_DEBUG_LOGGED
    
    from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
    
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    
    # Get blocksize from Pd
    blocksize = Pd.shape[0] if Pd is not None else 256
    
    # Debug logging (only rank 0, to file)
    if RESQ_DEBUG and not _RESQ_ROTATION_DEBUG_LOGGED and tp_rank == 0:
        _RESQ_ROTATION_DEBUG_LOGGED = True
        n_local = x.shape[-1]
        num_local_blocks = n_local // blocksize
        n_global = n_local * tp_size
        global_K = Hd_K
        resq_log(f"\n[TP Configuration]", rank=0)
        resq_log(f"  tp_size={tp_size}, tp_rank={tp_rank}", rank=0)
        resq_log(f"  x.shape={x.shape}, n_local={n_local}", rank=0)
        resq_log(f"  blocksize={blocksize}, num_local_blocks={num_local_blocks}", rank=0)
        resq_log(f"  n_global={n_global}, global_K={global_K}", rank=0)
        resq_log(f"  Pd.shape={Pd.shape if Pd is not None else None}", rank=0)
        resq_log(f"  Hd.shape={Hd.shape if Hd is not None else None}", rank=0)
        if Hd is not None and Hd_K > 1:
            blocks_per_rank = Hd_K // tp_size
            resq_log(f"  blocks_per_rank={blocks_per_rank}", rank=0)
            if Hd_K % tp_size != 0:
                resq_log(f"  WARNING: Hd_K ({Hd_K}) not divisible by tp_size ({tp_size})!", rank=0)
    
    return apply_resq_hadamard_rotation_tp(
        x, Pd, Hd, Hd_K, blocksize, tp_size, tp_rank
    )


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
        # Debug logging (only once, rank 0 only, to file)
        if RESQ_DEBUG and not Qwen3ResQAttention._debug_logged:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            if tp_rank == 0:
                Qwen3ResQAttention._debug_logged = True
                resq_log(f"\n[Attention Forward]", rank=0)
                resq_log(f"  RESQ_FAKE_QUANT={RESQ_FAKE_QUANT}", rank=0)
                resq_log(f"  rotation_R3: numel={self.rotation_R3.numel()}, shape={self.rotation_R3.shape}", rank=0)
                resq_log(f"  hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}", rank=0)
                if self.rotation_R3.numel() > 0:
                    resq_log(f"  rotation_R3 stats: min={self.rotation_R3.min():.4f}, max={self.rotation_R3.max():.4f}", rank=0)
                if hasattr(self.qkv_proj, 'weight'):
                    w = self.qkv_proj.weight
                    resq_log(f"  qkv_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}", rank=0)
        
        # Apply fake quantization to input (simulating dynamic activation quantization)
        if RESQ_FAKE_QUANT:
            hidden_states = apply_mixed_precision_fake_quant(
                hidden_states, RESQ_HIGH_FRACTION, RESQ_HIGH_BITS, RESQ_LOW_BITS
            )
        
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # QK norm
        q_shape = q.shape
        q_by_head = q.reshape(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q_shape)
        
        k_shape = k.shape
        k_by_head = k.reshape(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k_shape)

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Apply R3 rotation (post-RoPE)
        # Can be skipped via RESQ_SKIP_R3=1 environment variable
        global _RESQ_R3_DEBUG_LOGGED
        if self.apply_resq_rotation and not RESQ_SKIP_R3:
            rot_mat = None
            if self.rotation_R3.numel() > 0:
                rot_mat = self.rotation_R3
            
            # Debug: log Q/K shape before/after rotation (rank 0 only, to file)
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            should_log_r3 = RESQ_DEBUG and not _RESQ_R3_DEBUG_LOGGED and tp_rank == 0
            
            if should_log_r3:
                q_before_min, q_before_max = q.min().item(), q.max().item()
                k_before_min, k_before_max = k.min().item(), k.max().item()
            
            q = apply_rotation(q, rot_mat)
            k = apply_rotation(k, rot_mat)
            
            if should_log_r3:
                _RESQ_R3_DEBUG_LOGGED = True
                resq_log(f"\n[Attention R3 Rotation]", rank=0)
                resq_log(f"  q.shape={q.shape}, R3.shape={rot_mat.shape if rot_mat is not None else None}", rank=0)
                resq_log(f"  head_dim={self.head_dim}, num_heads={self.num_heads}", rank=0)
                resq_log(f"  q before: min={q_before_min:.4f}, max={q_before_max:.4f}", rank=0)
                resq_log(f"  q after:  min={q.min().item():.4f}, max={q.max().item():.4f}", rank=0)
                resq_log(f"  k before: min={k_before_min:.4f}, max={k_before_max:.4f}", rank=0)
                resq_log(f"  k after:  min={k.min().item():.4f}, max={k.max().item():.4f}", rank=0)
        elif RESQ_SKIP_R3 and RESQ_DEBUG:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            if not _RESQ_R3_DEBUG_LOGGED and tp_rank == 0:
                _RESQ_R3_DEBUG_LOGGED = True
                resq_log(f"\n[Attention R3 Rotation] SKIPPED (RESQ_SKIP_R3=1)", rank=0)

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
    Qwen3 MLP with ResQ Hadamard rotation applied before down_proj.
    Optionally applies W4A4 fake quantization when RESQ_FAKE_QUANT=1.
    
    Hadamard mode: R = block_diag(Pd) @ H
    - Pd: per-layer eigenvector basis [blocksize, blocksize]
    - H = Hd ⊗ H_butterfly (Hd is shared across all layers)
    
    Note: Hd is shared across all layers. Each layer stores a reference to
    the same tensor (set via set_shared_hadamard after loading weights).
    """
    _debug_logged = False  # Class-level flag to log only once
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Hadamard mode: Pd is per-layer, Hd is shared across layers
        # Pd: per-layer eigenvector basis [blocksize, blocksize]
        self.register_parameter("rotation_Pd", nn.Parameter(torch.empty(0), requires_grad=False))
        
        # Shared Hadamard parameters (set by set_shared_hadamard after loading)
        # Using instance variables that will reference the same tensor across layers
        self.shared_Hd: Optional[torch.Tensor] = None
        self.shared_Hd_K: int = 1
        
        # Custom loader for Pd
        def pd_loader(param, loaded_weight):
            if RESQ_DEBUG:
                pd_max = loaded_weight.abs().max().item()
                logger.warning(f"[ResQ] Loading rotation_Pd: shape={loaded_weight.shape}, max_abs={pd_max:.4f}")
            if param.data.shape != loaded_weight.shape:
                param.data = loaded_weight.clone()
            else:
                param.data.copy_(loaded_weight)
        setattr(self.rotation_Pd, "weight_loader", pd_loader)
    
    def set_shared_hadamard(self, Hd: Optional[torch.Tensor], Hd_K: int):
        """Set shared Hadamard matrix reference (same tensor object for all layers)."""
        self.shared_Hd = Hd  # Store reference, not copy
        self.shared_Hd_K = Hd_K
        if RESQ_DEBUG:
            hd_max = Hd.abs().max().item() if Hd is not None else 0
            logger.warning(f"[ResQ] set_shared_hadamard called: Hd_K={Hd_K}, Hd_max_abs={hd_max:.4f}, is_normalized={hd_max < 0.5}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Debug logging (only once, rank 0 only, to file)
        if RESQ_DEBUG and not Qwen3ResQMLP._debug_logged:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            if tp_rank == 0:
                Qwen3ResQMLP._debug_logged = True
                resq_log(f"\n[MLP Forward]", rank=0)
                resq_log(f"  RESQ_FAKE_QUANT={RESQ_FAKE_QUANT}", rank=0)
                resq_log(f"  rotation_Pd: numel={self.rotation_Pd.numel()}, shape={self.rotation_Pd.shape}", rank=0)
                resq_log(f"  shared_Hd: shape={self.shared_Hd.shape if self.shared_Hd is not None else None}", rank=0)
                resq_log(f"  shared_Hd_K={self.shared_Hd_K}", rank=0)
                if self.rotation_Pd.numel() > 0:
                    resq_log(f"  rotation_Pd stats: min={self.rotation_Pd.min():.4f}, max={self.rotation_Pd.max():.4f}", rank=0)
                if hasattr(self.gate_up_proj, 'weight'):
                    w = self.gate_up_proj.weight
                    resq_log(f"  gate_up_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}", rank=0)
                if hasattr(self.down_proj, 'weight'):
                    w = self.down_proj.weight
                    resq_log(f"  down_proj.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}, row_norm_mean={w.norm(dim=1).mean():.4f}", rank=0)
        
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
        
        # Debug: log intermediate stats before rotation
        global _RESQ_MLP_DEBUG_LOGGED
        if RESQ_DEBUG and not _RESQ_MLP_DEBUG_LOGGED:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                _RESQ_MLP_DEBUG_LOGGED = True
                resq_log(f"\n[MLP Intermediate Activation Debug]", rank=0)
                resq_log(f"  Before rotation: shape={intermediate.shape}", rank=0)
                resq_log(f"  Before rotation: min={intermediate.min().item():.6f}, max={intermediate.max().item():.6f}, mean={intermediate.mean().item():.6f}", rank=0)
                resq_log(f"  Before rotation: norm={intermediate.norm().item():.6f}", rank=0)
        
        # Apply Hadamard rotation before down_proj: R = block_diag(Pd) @ H
        if self.rotation_Pd.numel() > 0:
            # Use shared Hd (same tensor reference for all layers)
            intermediate = apply_resq_hadamard_rotation(
                intermediate,
                Pd=self.rotation_Pd,
                Hd=self.shared_Hd,
                Hd_K=self.shared_Hd_K,
            )
        
        # Debug: log intermediate stats after rotation
        if RESQ_DEBUG and _RESQ_MLP_DEBUG_LOGGED:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                resq_log(f"  After rotation: min={intermediate.min().item():.6f}, max={intermediate.max().item():.6f}, mean={intermediate.mean().item():.6f}", rank=0)
                resq_log(f"  After rotation: norm={intermediate.norm().item():.6f}", rank=0)
        
        # Apply fake quantization to down_proj input (after rotation)
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
        
        # Use ResQ MLP with Hadamard rotation
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
        - model.layers.X.self_attn.rotation_R3 (Uc)
        - model.layers.X.mlp.rotation_Pd (per-layer eigenvector basis)
        - resq.Hd (global Hadamard block matrix)
        - resq.Hd_K (global Hadamard block size)
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader
        
        # Collect weights and extract global ResQ parameters
        weights_list = list(weights)
        global_Hd = None
        global_Hd_K = 1
        filtered_weights = []
        
        # Detect Hadamard mode from checkpoint
        has_rotation_Pd = False
        
        for name, tensor in weights_list:
            if name == "resq.Hd":
                global_Hd = tensor
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Found global resq.Hd: shape={tensor.shape}")
            elif name == "resq.Hd_K":
                global_Hd_K = int(tensor.item())
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Found global resq.Hd_K: {global_Hd_K}")
            elif name.startswith("resq."):
                # Skip other resq.* global params (intermediate_size, blocksize)
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Skipping global param: {name}")
            else:
                filtered_weights.append((name, tensor))
                # Check for rotation_Pd (Hadamard mode)
                if "rotation_Pd" in name:
                    has_rotation_Pd = True
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ] Detected Hadamard mode: has_rotation_Pd={has_rotation_Pd}")
        
        if RESQ_DEBUG:
            logger.warning("[ResQ] load_weights called")
            logger.warning(f"[ResQ] Total weights in checkpoint: {len(weights_list)}")
            logger.warning(f"[ResQ] Filtered weights (excluding resq.*): {len(filtered_weights)}")
            
            # Check for rotation weights
            rotation_count = 0
            sample_weights = []
            for name, tensor in filtered_weights[:20]:
                sample_weights.append(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            for name, tensor in filtered_weights:
                if "rotation" in name:
                    rotation_count += 1
                    logger.warning(f"[ResQ] Found rotation weight: {name}, shape={tensor.shape}")
            logger.warning(f"[ResQ] Sample weights (first 20):")
            for s in sample_weights:
                logger.warning(f"[ResQ] {s}")
            logger.warning(f"[ResQ] Total rotation weights found: {rotation_count}")
        
        # Build skip prefixes
        skip_prefixes = []
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        
        # Load standard weights using AutoWeightsLoader
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes if skip_prefixes else None,
        )
        
        loaded_keys = loader.load_weights(iter(filtered_weights))
        
        # Set shared Hadamard parameters (same tensor reference for all MLP layers)
        if global_Hd is not None or global_Hd_K > 1:
            if RESQ_DEBUG:
                logger.warning(f"[ResQ] Setting shared Hd (shape={global_Hd.shape if global_Hd is not None else None}) and Hd_K={global_Hd_K}")
            
            # Pass the same tensor reference to all MLP layers (no copies)
            for layer in self.model.layers:
                if isinstance(layer, Qwen3ResQDecoderLayer):
                    mlp = layer.mlp
                    if isinstance(mlp, Qwen3ResQMLP):
                        mlp.set_shared_hadamard(global_Hd, global_Hd_K)
        
        if RESQ_DEBUG:
            logger.warning(f"[ResQ] Loaded {len(loaded_keys)} weight keys")
            # Check if rotation parameters were loaded by checking their sizes
            for name, param in self.named_parameters():
                if "rotation" in name:
                    logger.warning(f"[ResQ] After loading - {name}: numel={param.numel()}, shape={param.shape}")
        
        return loaded_keys
