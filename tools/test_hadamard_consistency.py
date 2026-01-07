#!/usr/bin/env python3
"""
Test to verify that vllm's Hadamard implementation matches msit's matmul_hadU_cpu.
"""

import torch
import math
import sys
sys.path.insert(0, '/Users/patrick/Projects/msit/msmodelslim')

# Import msit's implementation
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.utils.hadamard_utils import (
    matmul_hadU_cpu, get_hadK
)

# vllm's implementation (copied from qwen3_resq.py)
def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)

def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Butterfly Hadamard (unnormalized)."""
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

def vllm_hadamard(x: torch.Tensor, Hd: torch.Tensor, K: int) -> torch.Tensor:
    """vllm's Hadamard implementation (from apply_resq_hadamard_rotation_tp)."""
    n = x.shape[-1]
    blocksize = n // K
    
    # Reshape: [batch, n] -> [batch, K, blocksize]
    x_reshape = x.view(-1, K, blocksize)
    
    # Apply butterfly to blocksize dimension
    x_had = hadamard_transform(x_reshape.contiguous())
    
    # Normalize by sqrt(n)
    x_had = x_had / math.sqrt(n)
    
    # Apply Hd
    result = torch.einsum('ij,bjk->bik', Hd.to(x.dtype), x_had)
    
    return result.view(x.shape)


def main():
    print("=" * 60)
    print("Testing Hadamard Consistency: vllm vs msit")
    print("=" * 60)
    
    # Test parameters (mimicking Qwen3-32B)
    n = 25600  # intermediate_size
    batch = 4
    
    # Get Hadamard matrix from msit
    hadK, K = get_hadK(n)
    print(f"\nParameters:")
    print(f"  n = {n}")
    print(f"  K = {K}")
    print(f"  blocksize = n/K = {n // K}")
    print(f"  hadK shape = {hadK.shape if hadK is not None else None}")
    print(f"  hadK max abs = {hadK.abs().max().item():.4f}")
    
    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch, n, dtype=torch.float64)
    
    # msit's implementation
    msit_result = matmul_hadU_cpu(x, hadK, K)
    
    # vllm's implementation  
    vllm_result = vllm_hadamard(x, hadK, K)
    
    # Compare
    diff = (msit_result - vllm_result).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n--- Results ---")
    print(f"msit output: min={msit_result.min():.4f}, max={msit_result.max():.4f}, mean={msit_result.mean():.4f}")
    print(f"vllm output: min={vllm_result.min():.4f}, max={vllm_result.max():.4f}, mean={vllm_result.mean():.4f}")
    print(f"\nDifference: max={max_diff:.2e}, mean={mean_diff:.2e}")
    
    if max_diff < 1e-10:
        print("\n✅ PASS: vllm and msit Hadamard implementations match!")
    else:
        print("\n❌ FAIL: vllm and msit Hadamard implementations differ!")
        
        # Debug: check intermediate values
        print("\n--- Debug ---")
        x_reshape = x.view(-1, K, n // K)
        x_had_vllm = hadamard_transform(x_reshape.contiguous())
        
        # msit's butterfly
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.utils.hadamard_utils import HadamardTransform
        x_had_msit = HadamardTransform.apply(x_reshape.contiguous())
        
        butterfly_diff = (x_had_vllm - x_had_msit).abs().max().item()
        print(f"Butterfly difference: {butterfly_diff:.2e}")
        
    # Also test that H @ H.T = I (orthogonality)
    print("\n--- Orthogonality Test ---")
    x_identity = torch.eye(n, dtype=torch.float64)
    H_x = matmul_hadU_cpu(x_identity, hadK, K)
    H_Ht = H_x @ H_x.T
    I_diff = (H_Ht - torch.eye(n, dtype=torch.float64)).abs().max().item()
    print(f"|H @ H.T - I|_max = {I_diff:.2e}")
    
    if I_diff < 1e-10:
        print("✅ H is orthogonal")
    else:
        print("⚠️  H may not be perfectly orthogonal (numerical error)")


if __name__ == "__main__":
    main()

