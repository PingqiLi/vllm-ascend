#!/usr/bin/env python3
"""
End-to-end test for ResQ MLP rotation.
Simulates msit's weight fusion and vllm's inference to verify correctness.
"""

import torch
import math
import sys
sys.path.insert(0, '/Users/patrick/Projects/msit/msmodelslim')

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.utils.hadamard_utils import (
    matmul_hadU_cpu, get_hadK
)

# ============= vllm's Hadamard implementation (from qwen3_resq.py) =============

def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)

def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Butterfly Hadamard (unnormalized)."""
    n = u.shape[-1]
    assert is_pow2(n)
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

def vllm_rotation(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, K: int) -> torch.Tensor:
    """vllm's ResQ rotation: x @ block_diag(Pd.T) @ H.T"""
    n = x.shape[-1]
    blocksize = Pd.shape[0]
    num_blocks = n // blocksize
    
    # Step 1: Apply block_diag(Pd.T)
    x_reshape = x.view(-1, num_blocks, blocksize)
    x_pd = torch.matmul(x_reshape, Pd.T)
    
    # Step 2: Apply H.T = (Hd ⊗ H_butterfly).T
    # Using msit's approach: hadK @ (HadamardTransform / sqrt(n))
    x_had = hadamard_transform(x_pd.contiguous())
    x_had = x_had / math.sqrt(n)
    result = torch.einsum('ij,bjk->bik', Hd, x_had)
    
    return result.view(x.shape)

# ============= Test =============

def main():
    print("=" * 70)
    print("End-to-End Test: ResQ MLP Rotation")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Dimensions (smaller for testing)
    hidden_size = 512
    intermediate_size = 1024  # 4 * 256 blocks
    batch = 2
    
    # Get Hadamard matrix
    Hd, K = get_hadK(intermediate_size)
    blocksize = intermediate_size // K
    num_blocks = K
    print(f"\nDimensions:")
    print(f"  hidden_size = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  K (Hd dim) = {K}")
    print(f"  blocksize = {blocksize}")
    print(f"  num_blocks = {num_blocks}")
    print(f"  Hd max = {Hd.abs().max():.4f}")
    
    # Create random matrices
    Ua = torch.eye(hidden_size, dtype=torch.float64)  # Use identity for simplicity
    Pd = torch.randn(blocksize, blocksize, dtype=torch.float64)
    Pd, _ = torch.linalg.qr(Pd)  # Make orthogonal
    
    # Original down_proj weight
    W_orig = torch.randn(hidden_size, intermediate_size, dtype=torch.float64)
    
    # ============= msit's weight fusion =============
    print("\n--- Weight Fusion (msit style) ---")
    W_fused = W_orig.clone()
    
    # Step 1: Apply Ua.T to output dim
    W_fused = torch.matmul(Ua.T, W_fused)
    print(f"After Ua.T: shape={W_fused.shape}")
    
    # Step 2: Apply block_diag(Pd.T) to input dim
    W_fused = W_fused.view(hidden_size, num_blocks, blocksize)
    W_fused = torch.matmul(W_fused, Pd.T)
    W_fused = W_fused.view(hidden_size, intermediate_size)
    print(f"After Pd.T: shape={W_fused.shape}")
    
    # Step 3: Apply H.T via matmul_hadU_cpu
    W_fused = matmul_hadU_cpu(W_fused, Hd, K)
    print(f"After H.T: shape={W_fused.shape}")
    
    # ============= Forward pass tests =============
    print("\n--- Forward Pass Tests ---")
    
    # Create test input
    x = torch.randn(batch, intermediate_size, dtype=torch.float64)
    
    # Ground truth: original forward without any rotation
    y_orig = torch.matmul(x, W_orig.T)
    y_orig = torch.matmul(y_orig, Ua)  # Apply Ua to output (as per msit design)
    print(f"Ground truth output: shape={y_orig.shape}, mean={y_orig.mean():.4f}")
    
    # Test 1: Using vllm's rotation
    x_rotated_vllm = vllm_rotation(x, Pd, Hd, K)
    y_vllm = torch.matmul(x_rotated_vllm, W_fused.T)
    
    diff_vllm = (y_vllm - y_orig).abs()
    print(f"\nTest 1 (vllm rotation):")
    print(f"  y_vllm mean={y_vllm.mean():.4f}")
    print(f"  Diff max={diff_vllm.max():.2e}, mean={diff_vllm.mean():.2e}")
    
    # Test 2: Using msit's matmul_hadU_cpu for rotation
    x_pd = x.view(-1, num_blocks, blocksize)
    x_pd = torch.matmul(x_pd, Pd.T)
    x_pd = x_pd.view(batch, intermediate_size)
    x_rotated_msit = matmul_hadU_cpu(x_pd, Hd, K)
    y_msit = torch.matmul(x_rotated_msit, W_fused.T)
    
    diff_msit = (y_msit - y_orig).abs()
    print(f"\nTest 2 (msit rotation):")
    print(f"  y_msit mean={y_msit.mean():.4f}")
    print(f"  Diff max={diff_msit.max():.2e}, mean={diff_msit.mean():.2e}")
    
    # Test 3: Compare vllm vs msit rotation directly
    diff_rotation = (x_rotated_vllm - x_rotated_msit).abs()
    print(f"\nTest 3 (vllm vs msit rotation):")
    print(f"  Diff max={diff_rotation.max():.2e}, mean={diff_rotation.mean():.2e}")
    
    # ============= Summary =============
    print("\n" + "=" * 70)
    if diff_vllm.max() < 1e-8:
        print("✅ PASS: vllm rotation produces correct output!")
    else:
        print("❌ FAIL: vllm rotation produces incorrect output!")
        
        # Debug: check intermediate values
        print("\n--- Debug Info ---")
        print(f"x stats: min={x.min():.4f}, max={x.max():.4f}")
        print(f"x_rotated_vllm stats: min={x_rotated_vllm.min():.4f}, max={x_rotated_vllm.max():.4f}")
        print(f"x_rotated_msit stats: min={x_rotated_msit.min():.4f}, max={x_rotated_msit.max():.4f}")
        print(f"W_fused stats: min={W_fused.min():.4f}, max={W_fused.max():.4f}")
        
        # Check orthogonality
        pd_orth = (Pd @ Pd.T - torch.eye(blocksize, dtype=torch.float64)).abs().max()
        print(f"Pd orthogonality error: {pd_orth:.2e}")
        
        # Check Hadamard
        H_test = torch.eye(intermediate_size, dtype=torch.float64)
        H_out = matmul_hadU_cpu(H_test, Hd, K)
        H_orth = (H_out @ H_out.T - torch.eye(intermediate_size, dtype=torch.float64)).abs().max()
        print(f"H orthogonality error: {H_orth:.2e}")


if __name__ == "__main__":
    main()

