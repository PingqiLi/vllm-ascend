#!/usr/bin/env python3
"""
Verify ResQ weights and rotation parameters in checkpoint.

This script:
1. Loads the checkpoint directly
2. Verifies rotation parameters (Pd, Hd, R3) are present and valid
3. Tests the rotation chain on a single MLP layer

Usage:
    python verify_resq_weights.py /path/to/resq_checkpoint
"""

import argparse
import torch
import math
from pathlib import Path


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load all tensors from safetensors checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    all_tensors = {}
    
    if checkpoint_path.is_dir():
        # Load from directory with multiple safetensors files
        from safetensors import safe_open
        for shard_file in sorted(checkpoint_path.glob("*.safetensors")):
            print(f"Loading: {shard_file.name}")
            with safe_open(shard_file, framework="pt") as f:
                for key in f.keys():
                    all_tensors[key] = f.get_tensor(key)
    else:
        # Load from single file
        from safetensors.torch import load_file
        all_tensors = load_file(str(checkpoint_path))
    
    return all_tensors


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Butterfly Hadamard (unnormalized)."""
    n = u.shape[-1]
    assert (n & (n - 1) == 0) and (n > 0), f"n must be power of 2, got {n}"
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


def apply_rotation_chain(x, Pd, Hd, K):
    """Apply Pd.T and Hadamard rotation to activation x."""
    n = x.shape[-1]
    blocksize = Pd.shape[0] if Pd is not None else 256
    num_blocks = n // blocksize
    
    # Step 1: Apply block_diag(Pd.T)
    if Pd is not None:
        x_reshape = x.view(-1, num_blocks, blocksize)
        x = torch.matmul(x_reshape, Pd.T.to(x.dtype))
        x = x.view(-1, n)
    
    # Step 2: Apply Hadamard
    if Hd is not None and K > 1:
        x_reshape = x.view(-1, K, n // K)
        x_had = hadamard_transform(x_reshape.contiguous())
        x_had = x_had / math.sqrt(n)
        x = torch.einsum('ij,bjk->bik', Hd.to(x.dtype), x_had)
        x = x.view(-1, n)
    
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint directory or file")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to verify")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ResQ Weight Verification")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    tensors = load_checkpoint(args.checkpoint)
    print(f"Loaded {len(tensors)} tensors")
    
    # Find rotation parameters
    print("\n--- Rotation Parameters ---")
    
    # Global Hd
    Hd = tensors.get("resq.Hd", None)
    Hd_K_tensor = tensors.get("resq.Hd_K", None)
    Hd_K = int(Hd_K_tensor.item()) if Hd_K_tensor is not None else 1
    
    if Hd is not None:
        print(f"resq.Hd: shape={Hd.shape}, max={Hd.abs().max():.4f}")
        # Check orthogonality
        Hd_f64 = Hd.to(torch.float64)
        Hd_orth = (Hd_f64 @ Hd_f64.T - torch.eye(Hd_K, dtype=torch.float64)).abs().max()
        print(f"  Orthogonality: |Hd @ Hd.T - I|_max = {Hd_orth:.2e}")
    else:
        print("resq.Hd: NOT FOUND")
    
    print(f"resq.Hd_K: {Hd_K}")
    
    # Per-layer Pd
    layer = args.layer
    Pd_key = f"model.layers.{layer}.mlp.rotation_Pd"
    Pd = tensors.get(Pd_key, None)
    
    if Pd is not None:
        print(f"\n{Pd_key}: shape={Pd.shape}, max={Pd.abs().max():.4f}")
        # Check orthogonality
        Pd_f64 = Pd.to(torch.float64)
        Pd_orth = (Pd_f64 @ Pd_f64.T - torch.eye(Pd.shape[0], dtype=torch.float64)).abs().max()
        print(f"  Orthogonality: |Pd @ Pd.T - I|_max = {Pd_orth:.2e}")
    else:
        print(f"\n{Pd_key}: NOT FOUND")
    
    # Per-layer R3
    R3_key = f"model.layers.{layer}.self_attn.rotation_R3"
    R3 = tensors.get(R3_key, None)
    
    if R3 is not None:
        print(f"\n{R3_key}: shape={R3.shape}, max={R3.abs().max():.4f}")
        R3_f64 = R3.to(torch.float64)
        R3_orth = (R3_f64 @ R3_f64.T - torch.eye(R3.shape[0], dtype=torch.float64)).abs().max()
        print(f"  Orthogonality: |R3 @ R3.T - I|_max = {R3_orth:.2e}")
    else:
        print(f"\n{R3_key}: NOT FOUND")
    
    # Load down_proj weight
    print("\n--- Weight Verification ---")
    down_proj_key = f"model.layers.{layer}.mlp.down_proj.weight"
    down_proj = tensors.get(down_proj_key, None)
    
    if down_proj is None:
        print(f"{down_proj_key}: NOT FOUND")
        return
    
    print(f"{down_proj_key}: shape={down_proj.shape}")
    print(f"  Stats: min={down_proj.min():.4f}, max={down_proj.max():.4f}, mean={down_proj.mean():.4f}")
    
    # Check if weight looks fused (typically smaller values after rotation)
    weight_std = down_proj.std().item()
    print(f"  Std: {weight_std:.4f}")
    
    # Test rotation chain
    if Pd is not None and Hd is not None:
        print("\n--- Rotation Chain Test ---")
        
        hidden_size = down_proj.shape[0]
        intermediate_size = down_proj.shape[1]
        blocksize = Pd.shape[0]
        
        print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"blocksize={blocksize}, K={Hd_K}")
        
        # Create random input
        torch.manual_seed(42)
        x = torch.randn(4, intermediate_size, dtype=torch.float64)
        
        # Apply rotation
        x_rotated = apply_rotation_chain(x, Pd.to(torch.float64), Hd.to(torch.float64), Hd_K)
        
        # Compute output
        y = torch.matmul(x_rotated, down_proj.T.to(torch.float64))
        
        print(f"\nInput x: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
        print(f"Rotated x: shape={x_rotated.shape}, mean={x_rotated.mean():.4f}, std={x_rotated.std():.4f}")
        print(f"Output y: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Check if rotation is approximately identity when applied twice (should be if correct)
        # H @ H.T = I, so (H @ x).T @ (H @ x) should = x.T @ x (up to numerical error)
        x_xt = (x @ x.T).mean().item()
        xr_xrt = (x_rotated @ x_rotated.T).mean().item()
        print(f"\nEnergy check: x @ x.T mean = {x_xt:.4f}, x_rot @ x_rot.T mean = {xr_xrt:.4f}")
        print(f"  Ratio: {xr_xrt / x_xt:.4f} (should be ~1.0 for orthogonal transform)")
    
    print("\n" + "=" * 70)
    print("Verification complete!")


if __name__ == "__main__":
    main()

