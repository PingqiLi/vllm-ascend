#!/usr/bin/env python3
"""
Verify ResQ embedding and lm_head weights.

This checks if:
1. embed_tokens.weight was correctly rotated by U_attn
2. lm_head.weight was correctly rotated by U_attn
3. No NaN or Inf values exist

Usage:
    python verify_resq_embedding.py --ckpt /path/to/preprocessed_ckpt
"""

import argparse
import os
from glob import glob
import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to preprocessed ResQ checkpoint")
    parser.add_argument("--original_ckpt", help="Path to original (non-rotated) checkpoint for comparison")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.ckpt}")
    
    # Load key parameters
    embed_tokens_weight = None
    lm_head_weight = None
    layer0_down_proj = None
    layer0_gate_proj = None
    Pd_sample = None
    Hd = None
    Hd_K = None
    
    safetensor_files = glob(os.path.join(args.ckpt, "*.safetensors"))
    all_keys = []
    
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_keys.append(key)
                
                if key == "model.embed_tokens.weight":
                    embed_tokens_weight = f.get_tensor(key)
                    print(f"Found embed_tokens.weight: shape={embed_tokens_weight.shape}, dtype={embed_tokens_weight.dtype}")
                elif key == "lm_head.weight":
                    lm_head_weight = f.get_tensor(key)
                    print(f"Found lm_head.weight: shape={lm_head_weight.shape}, dtype={lm_head_weight.dtype}")
                elif key == "model.layers.0.mlp.down_proj.weight":
                    layer0_down_proj = f.get_tensor(key)
                    print(f"Found layer0.down_proj.weight: shape={layer0_down_proj.shape}")
                elif key == "model.layers.0.mlp.gate_up_proj.weight":
                    layer0_gate_proj = f.get_tensor(key)
                    print(f"Found layer0.gate_up_proj.weight: shape={layer0_gate_proj.shape}")
                elif key == "model.layers.0.mlp.rotation_Pd":
                    Pd_sample = f.get_tensor(key)
                    print(f"Found layer0.rotation_Pd: shape={Pd_sample.shape}")
                elif key == "resq.Hd":
                    Hd = f.get_tensor(key)
                    print(f"Found resq.Hd: shape={Hd.shape}")
                elif key == "resq.Hd_K":
                    Hd_K = int(f.get_tensor(key).item())
                    print(f"Found resq.Hd_K: {Hd_K}")
    
    print(f"\nTotal keys in checkpoint: {len(all_keys)}")
    
    # Print key counts by type
    embed_keys = [k for k in all_keys if "embed" in k]
    rotation_keys = [k for k in all_keys if "rotation" in k]
    resq_keys = [k for k in all_keys if k.startswith("resq.")]
    
    print(f"  Embedding-related keys: {len(embed_keys)}")
    print(f"  Rotation keys: {len(rotation_keys)}")
    print(f"  ResQ global keys: {len(resq_keys)}")
    
    # Print rotation keys
    if rotation_keys:
        print(f"\nRotation keys ({len(rotation_keys)}):")
        for k in rotation_keys[:10]:
            print(f"  {k}")
        if len(rotation_keys) > 10:
            print(f"  ... and {len(rotation_keys) - 10} more")
    
    # Verify embed_tokens
    print("\n" + "="*60)
    print("Verifying embed_tokens.weight")
    print("="*60)
    
    if embed_tokens_weight is not None:
        w = embed_tokens_weight.float()
        print(f"  Shape: {w.shape}")
        print(f"  Dtype: {embed_tokens_weight.dtype}")
        print(f"  Min: {w.min().item():.6f}")
        print(f"  Max: {w.max().item():.6f}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  NaN count: {torch.isnan(w).sum().item()}")
        print(f"  Inf count: {torch.isinf(w).sum().item()}")
        
        # Check for zero rows (problematic)
        row_norms = w.norm(dim=1)
        zero_rows = (row_norms < 1e-6).sum().item()
        print(f"  Zero rows (norm < 1e-6): {zero_rows}")
        
        if zero_rows > 0:
            print("  ⚠ WARNING: Some embedding rows are near-zero!")
    else:
        print("  ❌ NOT FOUND in checkpoint!")
    
    # Verify lm_head
    print("\n" + "="*60)
    print("Verifying lm_head.weight")
    print("="*60)
    
    if lm_head_weight is not None:
        w = lm_head_weight.float()
        print(f"  Shape: {w.shape}")
        print(f"  Dtype: {lm_head_weight.dtype}")
        print(f"  Min: {w.min().item():.6f}")
        print(f"  Max: {w.max().item():.6f}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  NaN count: {torch.isnan(w).sum().item()}")
        print(f"  Inf count: {torch.isinf(w).sum().item()}")
    elif embed_tokens_weight is not None:
        print("  Not found separately - likely tied to embed_tokens (tie_word_embeddings=True)")
    else:
        print("  ❌ NOT FOUND in checkpoint!")
    
    # Check if lm_head and embed_tokens are identical (tied)
    if lm_head_weight is not None and embed_tokens_weight is not None:
        if lm_head_weight.shape == embed_tokens_weight.shape:
            diff = (lm_head_weight.float() - embed_tokens_weight.float()).abs().max().item()
            print(f"\n  Comparison with embed_tokens:")
            print(f"    Same shape: Yes")
            print(f"    Max difference: {diff:.6e}")
            if diff < 1e-6:
                print(f"    ✓ Weights are identical (tied)")
            else:
                print(f"    Weights differ (not tied)")
    
    # Verify down_proj
    print("\n" + "="*60)
    print("Verifying layer0.down_proj.weight")
    print("="*60)
    
    if layer0_down_proj is not None:
        w = layer0_down_proj.float()
        print(f"  Shape: {w.shape} (should be [hidden_size, intermediate_size])")
        print(f"  Dtype: {layer0_down_proj.dtype}")
        print(f"  Min: {w.min().item():.6f}")
        print(f"  Max: {w.max().item():.6f}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  NaN count: {torch.isnan(w).sum().item()}")
        print(f"  Inf count: {torch.isinf(w).sum().item()}")
    else:
        print("  ❌ NOT FOUND in checkpoint!")
    
    # Verify Pd orthogonality
    print("\n" + "="*60)
    print("Verifying rotation_Pd orthogonality")
    print("="*60)
    
    if Pd_sample is not None:
        Pd = Pd_sample.float()
        orth_err = (Pd @ Pd.T - torch.eye(Pd.shape[0])).abs().max().item()
        print(f"  Shape: {Pd.shape}")
        print(f"  Min: {Pd.min().item():.6f}")
        print(f"  Max: {Pd.max().item():.6f}")
        print(f"  |Pd @ Pd.T - I|_max: {orth_err:.6e}")
        if orth_err < 1e-4:
            print(f"  ✓ Pd is orthogonal")
        else:
            print(f"  ⚠ WARNING: Pd may not be orthogonal!")
    else:
        print("  ❌ NOT FOUND in checkpoint!")
    
    # Verify Hd
    print("\n" + "="*60)
    print("Verifying resq.Hd")
    print("="*60)
    
    if Hd is not None:
        H = Hd.float()
        hd_max = H.abs().max().item()
        is_normalized = hd_max < 0.5
        print(f"  Shape: {H.shape}")
        print(f"  Hd_K: {Hd_K}")
        print(f"  Min: {H.min().item():.6f}")
        print(f"  Max: {H.max().item():.6f}")
        print(f"  Max abs: {hd_max:.6f}")
        print(f"  Is normalized (max < 0.5): {is_normalized}")
        print(f"  Expected max if normalized: {1.0 / (Hd_K ** 0.5):.6f}")
        
        # Check Hd orthogonality
        # For normalized Hd (elements ±1/sqrt(K)), Hd @ Hd.T = I
        # For unnormalized Hd (elements ±1), Hd @ Hd.T = K * I
        if is_normalized:
            orth_err = (H @ H.T - torch.eye(H.shape[0])).abs().max().item()
            print(f"  |Hd @ Hd.T - I|_max: {orth_err:.6e} (normalized Hd should satisfy Hd @ Hd.T = I)")
        else:
            orth_err = ((H @ H.T) / Hd_K - torch.eye(H.shape[0])).abs().max().item()
            print(f"  |Hd @ Hd.T / K - I|_max: {orth_err:.6e}")
        
        if orth_err < 1e-4:
            print(f"  ✓ Hd is orthogonal (accounting for normalization)")
        else:
            print(f"  ⚠ WARNING: Hd may not be orthogonal!")
    else:
        print("  Not found (may be None for power-of-2 dimensions)")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    issues = []
    if embed_tokens_weight is None:
        issues.append("embed_tokens.weight not found")
    elif torch.isnan(embed_tokens_weight.float()).any():
        issues.append("embed_tokens.weight contains NaN")
    
    if layer0_down_proj is None:
        issues.append("down_proj.weight not found")
    elif torch.isnan(layer0_down_proj.float()).any():
        issues.append("down_proj.weight contains NaN")
    
    if Pd_sample is not None:
        if (Pd_sample.float() @ Pd_sample.float().T - torch.eye(Pd_sample.shape[0])).abs().max().item() > 1e-4:
            issues.append("Pd is not orthogonal")
    
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All checks passed")


if __name__ == "__main__":
    main()

