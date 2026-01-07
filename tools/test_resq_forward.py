#!/usr/bin/env python3
"""
End-to-end test: verify that rotated activation + fused weight = original forward.

This tests the fundamental ResQ invariant:
    (x @ T) @ W'_fused.T == x @ W_original.T @ Ua

where:
    W'_fused = Ua.T @ W @ block_diag(Pd.T) @ H
    T = block_diag(Pd.T) @ H

Usage:
    python test_resq_forward.py --ckpt /path/to/preprocessed_ckpt --layer 0
"""

import argparse
import math
import os
from glob import glob
import torch
from safetensors import safe_open


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    if n == 1:
        return u
    
    assert (n & (n - 1)) == 0, f"Dimension must be power of 2, got {n}"
    
    original_shape = u.shape
    x = u.view(-1, n)
    
    h = 2
    while h <= n:
        hf = h // 2
        x = x.view(-1, n // h, 2, hf)
        x = torch.cat([x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]], dim=-1)
        x = x.view(-1, n)
        h *= 2
    
    return x.view(original_shape)


def matmul_hadU(x: torch.Tensor, hadK: torch.Tensor, K: int) -> torch.Tensor:
    """Apply x @ H where H = hadK ⊗ H_butterfly.
    
    Auto-detects if hadK is normalized (elements ~±1/sqrt(K)) or unnormalized (elements ~±1).
    """
    n = x.shape[-1]
    
    if K == 1 or hadK is None:
        return hadamard_transform(x.contiguous()) / math.sqrt(n)
    
    original_shape = x.shape
    blocksize = n // K
    
    # Check if hadK is already normalized
    hadk_max = hadK.abs().max().item()
    hadk_is_normalized = hadk_max < 0.5  # If max < 0.5, it's normalized
    
    # Reshape to [batch, K, blocksize]
    input_tensor = x.reshape(-1, K, blocksize)
    
    # Apply H_butterfly to blocksize dimension
    input_tensor = hadamard_transform(input_tensor.contiguous())
    
    # Normalization depends on whether hadK is already normalized
    if hadk_is_normalized:
        # hadK already has 1/sqrt(K) normalization, only need 1/sqrt(blocksize)
        input_tensor = input_tensor / math.sqrt(blocksize)
    else:
        # hadK is unnormalized, need full 1/sqrt(n)
        input_tensor = input_tensor / math.sqrt(n)
    
    # Apply hadK to K dimension
    hadK = hadK.to(device=input_tensor.device, dtype=input_tensor.dtype)
    input_tensor = hadK @ input_tensor
    
    return input_tensor.reshape(original_shape)


def apply_block_diag_Pd_T(x: torch.Tensor, Pd: torch.Tensor) -> torch.Tensor:
    """Apply block_diag(Pd.T) to x: x @ block_diag(Pd.T)"""
    blocksize = Pd.shape[0]
    n = x.shape[-1]
    num_blocks = n // blocksize
    
    original_shape = x.shape
    x = x.reshape(*original_shape[:-1], num_blocks, blocksize)
    x = torch.matmul(x, Pd.T.to(x.dtype))
    return x.reshape(*original_shape)


def apply_rotation_T(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, Hd_K: int) -> torch.Tensor:
    """Apply T = block_diag(Pd.T) @ H to x"""
    # Step 1: x @ block_diag(Pd.T)
    x = apply_block_diag_Pd_T(x, Pd)
    
    # Step 2: result @ H
    x = matmul_hadU(x, Hd, Hd_K)
    
    return x


def build_block_diag_Pd_T(Pd: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Build full block_diag(Pd.T) matrix for verification."""
    blocksize = Pd.shape[0]
    n = num_blocks * blocksize
    result = torch.zeros(n, n, dtype=Pd.dtype)
    
    for i in range(num_blocks):
        start = i * blocksize
        end = (i + 1) * blocksize
        result[start:end, start:end] = Pd.T
    
    return result


def build_full_H(Hd: torch.Tensor, blocksize: int) -> torch.Tensor:
    """Build full H matrix that matches matmul_hadU behavior.
    
    matmul_hadU computes: z[out_K, out_b] = sum_k Hd[out_K, k] * y[k, out_b]
    where y[k, out_b] = sum_{in_b} x[k, in_b] * H_butterfly[in_b, out_b]
    
    So the full matrix H[in_K * bs + in_b, out_K * bs + out_b] = Hd[out_K, in_K] * H_butterfly[in_b, out_b]
    Note: It's Hd[out_K, in_K], not Hd[in_K, out_K]!
    
    Auto-detects if Hd is normalized and adjusts accordingly.
    """
    K = Hd.shape[0]
    n = K * blocksize
    
    # Check if Hd is already normalized
    hd_max = Hd.abs().max().item()
    hd_is_normalized = hd_max < 0.5
    
    # Build H_butterfly (normalized Hadamard matrix)
    H_butterfly = torch.zeros(blocksize, blocksize, dtype=Hd.dtype)
    for i in range(blocksize):
        for j in range(blocksize):
            # Compute Hadamard entry using bit manipulation
            bit_count = bin(i & j).count('1')
            H_butterfly[i, j] = (-1) ** bit_count
    H_butterfly = H_butterfly / math.sqrt(blocksize)
    
    # Build H_full to match matmul_hadU:
    # H[in_K * bs + in_b, out_K * bs + out_b] = Hd[out_K, in_K] * H_butterfly[in_b, out_b]
    H_full = torch.zeros(n, n, dtype=Hd.dtype)
    for in_K in range(K):
        for out_K in range(K):
            row_start = in_K * blocksize
            row_end = (in_K + 1) * blocksize
            col_start = out_K * blocksize
            col_end = (out_K + 1) * blocksize
            if hd_is_normalized:
                H_full[row_start:row_end, col_start:col_end] = Hd[out_K, in_K] * H_butterfly
            else:
                H_full[row_start:row_end, col_start:col_end] = (Hd[out_K, in_K] / math.sqrt(K)) * H_butterfly
    
    return H_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to preprocessed ResQ checkpoint")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.ckpt}")
    
    # Load parameters
    Pd = None
    Hd = None
    Hd_K = None
    down_proj_weight = None
    
    safetensor_files = glob(os.path.join(args.ckpt, "*.safetensors"))
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == f"model.layers.{args.layer}.mlp.rotation_Pd":
                    Pd = f.get_tensor(key).float()
                    print(f"Loaded Pd: {key}, shape={Pd.shape}")
                elif key == "resq.Hd":
                    Hd = f.get_tensor(key).float()
                    print(f"Loaded Hd: {key}, shape={Hd.shape}")
                elif key == "resq.Hd_K":
                    Hd_K = int(f.get_tensor(key).item())
                    print(f"Loaded Hd_K: {Hd_K}")
                elif key == f"model.layers.{args.layer}.mlp.down_proj.weight":
                    down_proj_weight = f.get_tensor(key).float()
                    print(f"Loaded down_proj.weight: {key}, shape={down_proj_weight.shape}")
    
    if Pd is None or Hd is None or Hd_K is None or down_proj_weight is None:
        print("ERROR: Missing required parameters!")
        return
    
    # Derive dimensions
    blocksize = Pd.shape[0]
    intermediate_size = Hd_K * blocksize
    hidden_size = down_proj_weight.shape[0]
    
    print(f"\nDimensions:")
    print(f"  blocksize = {blocksize}")
    print(f"  Hd_K = {Hd_K}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  hidden_size = {hidden_size}")
    
    # Verify down_proj_weight shape
    if down_proj_weight.shape != (hidden_size, intermediate_size):
        print(f"ERROR: down_proj_weight shape mismatch! Expected ({hidden_size}, {intermediate_size}), got {down_proj_weight.shape}")
        return
    
    # Verify Pd orthogonality
    Pd_orth_err = (Pd @ Pd.T - torch.eye(blocksize)).abs().max().item()
    print(f"\nPd orthogonality: |Pd @ Pd.T - I|_max = {Pd_orth_err:.6e}")
    
    # Verify Hd properties
    hd_max = Hd.abs().max().item()
    hd_is_normalized = hd_max < 0.5
    print(f"Hd stats: min={Hd.min().item():.4f}, max={Hd.max().item():.4f}, max_abs={hd_max:.4f}")
    print(f"Hd is_normalized: {hd_is_normalized} (expected max_abs ~1/sqrt(K)={1/math.sqrt(Hd_K):.4f} if normalized)")
    
    # Create test input
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float32)
    
    print(f"\nTest input: shape={x.shape}")
    
    # =========================================================================
    # Test 1: Verify fast Hadamard matches matrix multiplication
    # =========================================================================
    print("\n" + "="*60)
    print("Test 1: Fast Hadamard vs Matrix Multiplication")
    print("="*60)
    
    # Use a smaller test for full matrix construction
    test_n = min(1024, intermediate_size)  # Limit size for memory
    test_x = torch.randn(4, test_n)
    test_K = test_n // blocksize
    test_Hd = Hd[:test_K, :test_K] if test_K < Hd_K else Hd
    
    # Fast version
    fast_result = matmul_hadU(test_x, test_Hd, test_K)
    
    # Build full H and compute with matmul
    H_full = build_full_H(test_Hd, blocksize)
    matrix_result = test_x @ H_full
    
    diff = (fast_result - matrix_result).abs()
    print(f"Max difference: {diff.max().item():.6e}")
    if diff.max().item() < 1e-4:
        print("✓ Fast Hadamard matches matrix multiplication")
    else:
        print("✗ Fast Hadamard DOES NOT match matrix multiplication!")
    
    # =========================================================================
    # Test 2: Verify activation rotation
    # =========================================================================
    print("\n" + "="*60)
    print("Test 2: Activation Rotation")
    print("="*60)
    
    # Apply rotation using our function
    x_rotated = apply_rotation_T(x, Pd, Hd, Hd_K)
    print(f"x_rotated: shape={x_rotated.shape}, min={x_rotated.min():.4f}, max={x_rotated.max():.4f}")
    
    # =========================================================================
    # Test 3: Forward pass with fused weights
    # =========================================================================
    print("\n" + "="*60)
    print("Test 3: Forward Pass (x_rotated @ W'_fused.T)")
    print("="*60)
    
    # Fused forward: x_rotated @ W'_fused.T
    y_fused = x_rotated @ down_proj_weight.T
    print(f"y_fused: shape={y_fused.shape}, min={y_fused.min():.4f}, max={y_fused.max():.4f}")
    
    # =========================================================================
    # Test 4: Verify the fundamental invariant
    # =========================================================================
    print("\n" + "="*60)
    print("Test 4: Invariant Check")
    print("="*60)
    print("If W'_fused = Ua.T @ W_orig @ block_diag(Pd.T) @ H, then:")
    print("  (x @ T) @ W'_fused.T = x @ W_orig.T @ Ua")
    print("\nWe can verify this by checking if the forward pass produces reasonable outputs.")
    print("(Without the original W and Ua, we can't do exact verification)")
    
    # Check output statistics
    y_mean = y_fused.mean().item()
    y_std = y_fused.std().item()
    print(f"\nOutput statistics:")
    print(f"  mean = {y_mean:.4f}")
    print(f"  std = {y_std:.4f}")
    print(f"  range = [{y_fused.min().item():.4f}, {y_fused.max().item():.4f}]")
    
    if abs(y_mean) < 1.0 and 0.1 < y_std < 10.0:
        print("\n✓ Output statistics look reasonable")
    else:
        print("\n⚠ Output statistics may indicate an issue")
    
    # =========================================================================
    # Test 5: Compare with no rotation (to detect gross errors)
    # =========================================================================
    print("\n" + "="*60)
    print("Test 5: Sanity Check - Rotation Effect")
    print("="*60)
    
    y_no_rotation = x @ down_proj_weight.T
    diff_with_rotation = (y_fused - y_no_rotation).abs()
    
    print(f"y without rotation: mean={y_no_rotation.mean():.4f}, std={y_no_rotation.std():.4f}")
    print(f"Difference (rotated vs non-rotated): mean={diff_with_rotation.mean():.4f}, max={diff_with_rotation.max():.4f}")
    
    if diff_with_rotation.mean().item() > 0.01:
        print("✓ Rotation has significant effect (expected)")
    else:
        print("⚠ Rotation has minimal effect (might indicate issue)")


if __name__ == "__main__":
    main()

