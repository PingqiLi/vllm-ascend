#!/usr/bin/env python3
"""
Test script to verify ResQ Hadamard rotation correctness with TP simulation.

This script:
1. Loads Pd, Hd from a ResQ checkpoint
2. Simulates the rotation with TP=1 (ground truth)
3. Simulates the rotation with TP=4 (distributed version)
4. Compares results

Usage:
    python test_resq_tp_rotation.py --ckpt /path/to/resq_preprocessed_ckpt
"""

import argparse
import math
import torch
from safetensors import safe_open


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    if n == 1:
        return u
    
    # Ensure power of 2
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
    """Apply structured Hadamard transform (TP=1 ground truth version).
    
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
        input_tensor = input_tensor / math.sqrt(blocksize)
    else:
        input_tensor = input_tensor / math.sqrt(n)
    
    # Apply hadK to K dimension: hadK @ input_tensor
    hadK = hadK.to(device=input_tensor.device, dtype=input_tensor.dtype)
    input_tensor = hadK @ input_tensor  # [K, K] @ [batch, K, blocksize]
    
    return input_tensor.reshape(original_shape)


def apply_rotation_tp1(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, Hd_K: int) -> torch.Tensor:
    """Apply ResQ rotation with TP=1 (ground truth)."""
    n = x.shape[-1]
    blocksize = Pd.shape[0]
    num_blocks = n // blocksize
    
    original_shape = x.shape
    
    # Step 1: Apply block_diag(Pd.T)
    x = x.reshape(*original_shape[:-1], num_blocks, blocksize)
    x = torch.matmul(x, Pd.T.to(x.dtype))
    x = x.reshape(*original_shape)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    x = matmul_hadU(x, Hd, Hd_K)
    
    return x


def apply_rotation_tp4_simulated(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, Hd_K: int, tp_size: int = 4) -> torch.Tensor:
    """
    Simulate TP=4 rotation by:
    1. Splitting x into tp_size shards
    2. Applying local rotation to each shard
    3. Simulating all-gather + Hd + slice
    4. Concatenating results
    """
    n = x.shape[-1]
    blocksize = Pd.shape[0]
    n_local = n // tp_size
    num_local_blocks = n_local // blocksize
    
    original_shape = x.shape
    batch_dims = original_shape[:-1]
    
    results = []
    
    for tp_rank in range(tp_size):
        # Get local shard
        start = tp_rank * n_local
        end = (tp_rank + 1) * n_local
        x_local = x[..., start:end]
        
        local_shape = x_local.shape
        
        # Step 1: Apply block_diag(Pd.T) locally
        x_local = x_local.reshape(*batch_dims, num_local_blocks, blocksize)
        x_local = torch.matmul(x_local, Pd.T.to(x_local.dtype))
        x_local = x_local.reshape(*local_shape)
        
        results.append(x_local)
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    # Check if Hd is already normalized
    hd_max = Hd.abs().max().item()
    hd_is_normalized = hd_max < 0.5
    
    # First apply H_butterfly locally to each shard
    for i in range(tp_size):
        results[i] = results[i].reshape(*batch_dims, num_local_blocks, blocksize)
        results[i] = hadamard_transform(results[i].contiguous())
    
    # Normalization depends on whether Hd is already normalized
    if hd_is_normalized:
        norm_factor = math.sqrt(blocksize)
    else:
        norm_factor = math.sqrt(n)  # n_global
    
    for i in range(tp_size):
        results[i] = results[i] / norm_factor
    
    # Simulate all-gather: concatenate all shards along blocks dimension
    # Each shard is [batch_dims, num_local_blocks, blocksize]
    # After all-gather: [batch_dims, K, blocksize]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    gathered_list = [r.reshape(batch_size, num_local_blocks, blocksize) for r in results]
    gathered = torch.cat(gathered_list, dim=1)  # [batch, K, blocksize]
    
    print(f"  Gathered shape: {gathered.shape} (expected [{batch_size}, {Hd_K}, {blocksize}])")
    
    # Apply Hd
    Hd_typed = Hd.to(device=gathered.device, dtype=gathered.dtype)
    mixed = torch.einsum('ij,bjk->bik', Hd_typed, gathered)
    
    print(f"  Mixed shape: {mixed.shape}")
    
    # Slice back for each rank
    final_results = []
    blocks_per_rank = Hd_K // tp_size
    for tp_rank in range(tp_size):
        start_block = tp_rank * blocks_per_rank
        end_block = start_block + blocks_per_rank
        x_sliced = mixed[:, start_block:end_block, :].contiguous()
        x_sliced = x_sliced.reshape(*batch_dims, num_local_blocks, blocksize)
        x_sliced = x_sliced.reshape(*batch_dims, n_local)
        final_results.append(x_sliced)
    
    # Concatenate all ranks' results
    result = torch.cat(final_results, dim=-1)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to preprocessed ResQ checkpoint")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.ckpt}")
    
    # Load rotation matrices
    Pd = None
    Hd = None
    Hd_K = None
    
    import os
    from glob import glob
    
    safetensor_files = glob(os.path.join(args.ckpt, "*.safetensors"))
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == f"model.layers.{args.layer}.mlp.rotation_Pd":
                    Pd = f.get_tensor(key)
                    print(f"Loaded Pd: {key}, shape={Pd.shape}")
                elif key == "resq.Hd":
                    Hd = f.get_tensor(key)
                    print(f"Loaded Hd: {key}, shape={Hd.shape}")
                elif key == "resq.Hd_K":
                    Hd_K = f.get_tensor(key).item()
                    print(f"Loaded Hd_K: {Hd_K}")
    
    if Pd is None:
        print("ERROR: Pd not found!")
        return
    if Hd is None:
        print("ERROR: Hd not found!")
        return
    if Hd_K is None:
        print("ERROR: Hd_K not found!")
        return
    
    # Verify Pd orthogonality
    Pd_orth_err = (Pd @ Pd.T - torch.eye(Pd.shape[0])).abs().max().item()
    print(f"\nPd orthogonality check: |Pd @ Pd.T - I|_max = {Pd_orth_err:.6f}")
    
    # Check Hd properties
    print(f"Hd stats: min={Hd.min().item():.4f}, max={Hd.max().item():.4f}")
    Hd_orth_err = (Hd @ Hd.T / Hd.shape[0] - torch.eye(Hd.shape[0])).abs().max().item()
    print(f"Hd orthogonality check (scaled): |Hd @ Hd.T / K - I|_max = {Hd_orth_err:.6f}")
    
    # Test parameters
    blocksize = Pd.shape[0]
    intermediate_size = Hd_K * blocksize
    batch_size = 4
    seq_len = 128
    
    print(f"\nTest parameters:")
    print(f"  blocksize = {blocksize}")
    print(f"  Hd_K (K) = {Hd_K}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len = {seq_len}")
    
    # Create random input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float32)
    
    print(f"\nInput x: shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")
    
    # TP=1 ground truth
    print("\n--- TP=1 (Ground Truth) ---")
    result_tp1 = apply_rotation_tp1(x, Pd, Hd, Hd_K)
    print(f"Result: shape={result_tp1.shape}, min={result_tp1.min().item():.4f}, max={result_tp1.max().item():.4f}")
    
    # TP=4 simulated
    print("\n--- TP=4 (Simulated) ---")
    result_tp4 = apply_rotation_tp4_simulated(x, Pd, Hd, Hd_K, tp_size=4)
    print(f"Result: shape={result_tp4.shape}, min={result_tp4.min().item():.4f}, max={result_tp4.max().item():.4f}")
    
    # Compare
    print("\n--- Comparison ---")
    diff = (result_tp1 - result_tp4).abs()
    print(f"Max absolute difference: {diff.max().item():.6e}")
    print(f"Mean absolute difference: {diff.mean().item():.6e}")
    
    if diff.max().item() < 1e-4:
        print("\n✓ TP=1 and TP=4 results match!")
    else:
        print("\n✗ TP=1 and TP=4 results DO NOT match!")
        
        # Debug: check where the difference is
        print("\nDebug: Checking per-rank differences...")
        n_local = intermediate_size // 4
        for rank in range(4):
            start = rank * n_local
            end = (rank + 1) * n_local
            rank_diff = (result_tp1[..., start:end] - result_tp4[..., start:end]).abs()
            print(f"  Rank {rank}: max_diff={rank_diff.max().item():.6e}, mean_diff={rank_diff.mean().item():.6e}")


if __name__ == "__main__":
    main()

