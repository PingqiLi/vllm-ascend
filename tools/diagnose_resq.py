#!/usr/bin/env python3
"""
Diagnose ResQ rotation issues.

This script simulates the EXACT same computation as msit's weight fusion
and vllm's inference, allowing step-by-step verification.

Usage:
    python diagnose_resq.py /path/to/checkpoint

Requirements:
    pip install safetensors
"""

import argparse
import torch
import math
import sys
from pathlib import Path

# Add msit to path
sys.path.insert(0, '/Users/patrick/Projects/msit/msmodelslim')


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load all tensors from safetensors checkpoint."""
    from safetensors import safe_open
    
    checkpoint_path = Path(checkpoint_path)
    all_tensors = {}
    
    if checkpoint_path.is_dir():
        for shard_file in sorted(checkpoint_path.glob("*.safetensors")):
            with safe_open(shard_file, framework="pt") as f:
                for key in f.keys():
                    all_tensors[key] = f.get_tensor(key)
    
    return all_tensors


def diagnose_rotation(checkpoint_path: str, layer_idx: int = 0):
    """Diagnose rotation issues by comparing msit and vllm implementations."""
    
    print("=" * 70)
    print("ResQ Rotation Diagnosis")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    tensors = load_checkpoint(checkpoint_path)
    
    # Extract parameters
    Pd_key = f"model.layers.{layer_idx}.mlp.rotation_Pd"
    Pd = tensors.get(Pd_key)
    Hd = tensors.get("resq.Hd")
    Hd_K = tensors.get("resq.Hd_K")
    K = int(Hd_K.item()) if Hd_K is not None else 1
    
    down_proj_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    W_fused = tensors.get(down_proj_key)
    
    if Pd is None or Hd is None or W_fused is None:
        print("ERROR: Missing required parameters!")
        print(f"  Pd: {'found' if Pd is not None else 'MISSING'}")
        print(f"  Hd: {'found' if Hd is not None else 'MISSING'}")
        print(f"  W_fused: {'found' if W_fused is not None else 'MISSING'}")
        return
    
    print(f"\nParameters for layer {layer_idx}:")
    print(f"  Pd: {Pd.shape}, dtype={Pd.dtype}")
    print(f"  Hd: {Hd.shape}, K={K}, dtype={Hd.dtype}")
    print(f"  W_fused: {W_fused.shape}, dtype={W_fused.dtype}")
    
    hidden_size = W_fused.shape[0]
    intermediate_size = W_fused.shape[1]
    blocksize = Pd.shape[0]
    num_blocks = intermediate_size // blocksize
    
    print(f"\nDimensions:")
    print(f"  hidden_size = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  blocksize = {blocksize}")
    print(f"  num_blocks = {num_blocks}")
    print(f"  K = {K}")
    
    # Verify dimensions
    assert num_blocks == K, f"num_blocks ({num_blocks}) != K ({K})"
    assert intermediate_size == K * blocksize, f"intermediate_size mismatch"
    
    # Convert to float64 for precision
    Pd = Pd.to(torch.float64)
    Hd = Hd.to(torch.float64)
    W_fused = W_fused.to(torch.float64)
    
    # Verify orthogonality
    print("\n--- Orthogonality Check ---")
    Pd_orth = (Pd @ Pd.T - torch.eye(blocksize, dtype=torch.float64)).abs().max()
    Hd_orth = (Hd @ Hd.T - torch.eye(K, dtype=torch.float64)).abs().max()
    print(f"  |Pd @ Pd.T - I|_max = {Pd_orth:.2e}")
    print(f"  |Hd @ Hd.T - I|_max = {Hd_orth:.2e}")
    
    # Create test input
    torch.manual_seed(42)
    batch_size = 2
    x = torch.randn(batch_size, intermediate_size, dtype=torch.float64)
    print(f"\nTest input: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    
    # ========== vllm's rotation (current implementation) ==========
    print("\n--- vllm's Rotation ---")
    
    # Import vllm's hadamard_transform
    def hadamard_transform(u):
        n = u.shape[-1]
        assert (n & (n - 1) == 0) and (n > 0)
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
    
    # Step 1: Apply Pd.T
    x_pd = x.view(batch_size, num_blocks, blocksize)
    x_pd = torch.matmul(x_pd, Pd.T)
    x_pd = x_pd.view(batch_size, intermediate_size)
    print(f"  After Pd.T: mean={x_pd.mean():.4f}, std={x_pd.std():.4f}")
    
    # Step 2: Apply Hadamard
    x_reshape = x_pd.view(batch_size, K, blocksize)
    x_had = hadamard_transform(x_reshape.contiguous())
    x_had = x_had / math.sqrt(intermediate_size)
    x_had = torch.einsum('ij,bjk->bik', Hd, x_had)
    x_vllm = x_had.view(batch_size, intermediate_size)
    print(f"  After Hadamard: mean={x_vllm.mean():.4f}, std={x_vllm.std():.4f}")
    
    # Compute output
    y_vllm = x_vllm @ W_fused.T
    print(f"  Output y_vllm: mean={y_vllm.mean():.4f}, std={y_vllm.std():.4f}")
    
    # ========== msit's rotation (using their library) ==========
    print("\n--- msit's Rotation ---")
    try:
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.utils.hadamard_utils import matmul_hadU_cpu
        
        # Apply Pd.T
        x_pd_msit = x.view(batch_size, num_blocks, blocksize)
        x_pd_msit = torch.matmul(x_pd_msit, Pd.T)
        x_pd_msit = x_pd_msit.view(batch_size, intermediate_size)
        
        # Apply Hadamard using msit's function
        x_msit = matmul_hadU_cpu(x_pd_msit, Hd, K)
        print(f"  After Pd.T + Hadamard: mean={x_msit.mean():.4f}, std={x_msit.std():.4f}")
        
        # Compute output
        y_msit = x_msit @ W_fused.T
        print(f"  Output y_msit: mean={y_msit.mean():.4f}, std={y_msit.std():.4f}")
        
        # Compare
        print("\n--- Comparison ---")
        diff_rotation = (x_vllm - x_msit).abs()
        print(f"  |x_vllm - x_msit|: max={diff_rotation.max():.2e}, mean={diff_rotation.mean():.2e}")
        
        diff_output = (y_vllm - y_msit).abs()
        print(f"  |y_vllm - y_msit|: max={diff_output.max():.2e}, mean={diff_output.mean():.2e}")
        
        if diff_rotation.max() < 1e-10:
            print("\n✅ vllm's Hadamard matches msit exactly!")
        else:
            print("\n⚠️  vllm's Hadamard differs from msit")
            
    except ImportError as e:
        print(f"  Could not import msit: {e}")
    
    # ========== Inverse check: can we recover original? ==========
    print("\n--- Inverse Check ---")
    
    # Apply inverse of Hadamard: H @ H^T = I, so H^T = H^{-1}
    # For normalized H, x @ H @ H^T = x
    x_inv_had = x_vllm.view(batch_size, K, blocksize)
    # Apply H^T (transpose of our forward)
    # Forward was: Hd @ (H_butterfly @ x / sqrt(n))
    # Inverse is: H_butterfly^T @ (Hd^T @ x) * sqrt(n)
    x_inv_had = torch.einsum('ji,bjk->bik', Hd, x_inv_had)  # Hd.T @
    x_inv_had = hadamard_transform(x_inv_had.contiguous())  # H_butterfly is symmetric
    x_inv_had = x_inv_had / math.sqrt(intermediate_size)  # Actually we need * sqrt(n) to invert
    x_inv_had = x_inv_had * intermediate_size  # Compensate for double division
    x_inv_had = x_inv_had.view(batch_size, intermediate_size)
    
    # Apply inverse of Pd.T: (Pd.T)^{-1} = Pd (orthogonal)
    x_inv_pd = x_inv_had.view(batch_size, num_blocks, blocksize)
    x_inv_pd = torch.matmul(x_inv_pd, Pd)
    x_recovered = x_inv_pd.view(batch_size, intermediate_size)
    
    diff_recover = (x_recovered - x).abs()
    print(f"  |x_recovered - x_original|: max={diff_recover.max():.2e}, mean={diff_recover.mean():.2e}")
    
    if diff_recover.max() < 1e-8:
        print("  ✅ Rotation is invertible (orthogonal)")
    else:
        print("  ⚠️  Rotation inverse has significant error")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    args = parser.parse_args()
    
    diagnose_rotation(args.checkpoint, args.layer)


if __name__ == "__main__":
    main()

