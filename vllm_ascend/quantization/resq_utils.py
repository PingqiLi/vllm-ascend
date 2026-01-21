#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

import math
from typing import Optional

import torch
import torch_npu


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


def pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """
    x: int8 tensor, shape (E, K, N)，值域 ∈ [-8, 7]
    return: int8 tensor, shape (E, K, N/2)，每个元素打包两个有符号 int4
    """
    assert x.dtype == torch.int8
    # Handle dynamic shapes if necessary, but typically expect valid input
    if x.dim() < 1:
         return x

    # Ensure last dim is even
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for packing, got {x.shape[-1]}")
    
    # 转成无符号补码 [0, 15]
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]   # 偶数 -> 低 4 位
    high = x_unsigned[..., 1::2]  # 奇数 -> 高 4 位

    out = (low | (high << 4)).to(torch.int8)
    return out


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
    h_butterfly: Optional[torch.Tensor], 
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
    """
    original_shape = x.shape
    n = x.shape[-1]  # intermediate_size
    
    # Validate dimensions
    if n != K * blocksize:
        raise ValueError(
            f"Dimension mismatch: intermediate_size={n} != K*blocksize={K}*{blocksize}={K*blocksize}. "
        )
    
    original_dtype = x.dtype
    x = x.float()
    
    # Reshape: (..., n) -> (..., K, blocksize) where K = num_blocks
    # K is the number of blocks, blocksize is the size of each block
    x = x.reshape(*original_shape[:-1], K, blocksize)
    
    # Step 1: Apply block_diag(Pd) block-wise (x @ Pd for each block)
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = torch.matmul(x, Pd_f32)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    # First apply butterfly Hadamard on the blocksize dimension (last dim)
    if h_butterfly is not None:
        h_butterfly_f32 = h_butterfly.to(device=x.device, dtype=torch.float32)
        x = torch.matmul(x, h_butterfly_f32)
    else:
        # Fallback to loop if not pre-computed
        x = hadamard_transform(x.contiguous())
    
    # Then apply Hd on the K dimension (second-to-last dim)
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
    x = x * K / math.sqrt(n)
    
    return x.reshape(original_shape).to(original_dtype)
