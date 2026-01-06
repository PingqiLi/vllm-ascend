#!/usr/bin/env python3
"""
Debug script to check ResQ weight values at different stages.
"""
import torch
from safetensors import safe_open
from glob import glob
import sys
import os

def check_preprocessed_checkpoint(path):
    """Check preprocessed bf16 checkpoint."""
    print("\n" + "="*60)
    print("CHECKING PREPROCESSED CHECKPOINT")
    print("="*60)
    
    sf_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not sf_files:
        print(f"ERROR: No safetensors files found in {path}")
        return
    
    print(f"Found {len(sf_files)} safetensors files")
    
    for sf_file in sf_files[:1]:  # Only check first file
        print(f"\nChecking: {sf_file}")
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Total keys: {len(keys)}")
            
            # Check rotation weights
            print("\n--- Rotation Weights ---")
            rotation_count = 0
            for key in keys:
                if "rotation" in key:
                    rotation_count += 1
                    if rotation_count <= 5:
                        t = f.get_tensor(key)
                        print(f"  {key}: shape={t.shape}, min={t.min():.6f}, max={t.max():.6f}")
            print(f"  Total rotation weights: {rotation_count}")
            
            # Check linear weights
            print("\n--- Linear Weights (first few) ---")
            weight_count = 0
            for key in keys:
                if ".weight" in key and "rotation" not in key and weight_count < 10:
                    t = f.get_tensor(key)
                    weight_count += 1
                    has_nan = torch.isnan(t).any().item()
                    has_inf = torch.isinf(t).any().item()
                    print(f"  {key}")
                    print(f"    shape={t.shape}, dtype={t.dtype}")
                    print(f"    min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}")
                    if has_nan:
                        print(f"    WARNING: Contains NaN!")
                    if has_inf:
                        print(f"    WARNING: Contains Inf!")
                    
                    # Check if values are suspiciously small
                    if t.abs().max() < 0.001:
                        print(f"    ⚠️ WARNING: Values are very small! Max abs = {t.abs().max():.8f}")


def check_original_checkpoint(path):
    """Check original ResQ quantized checkpoint."""
    print("\n" + "="*60)
    print("CHECKING ORIGINAL RESQ CHECKPOINT")
    print("="*60)
    
    sf_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not sf_files:
        print(f"ERROR: No safetensors files found in {path}")
        return
    
    print(f"Found {len(sf_files)} safetensors files")
    
    # Collect samples
    weight_samples = {"weight_low": [], "weight_high": [], "scale_low": [], "scale_high": []}
    
    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                for wtype in weight_samples.keys():
                    if wtype in key and len(weight_samples[wtype]) < 3:
                        t = f.get_tensor(key)
                        weight_samples[wtype].append((key, t))
    
    for wtype, samples in weight_samples.items():
        print(f"\n--- {wtype} samples ---")
        for key, t in samples:
            print(f"  {key}")
            print(f"    shape={t.shape}, dtype={t.dtype}")
            print(f"    min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}")
            
            # For scale, check if values are reasonable
            if "scale" in wtype:
                if t.abs().max() < 1e-6:
                    print(f"    ⚠️ WARNING: Scale is very small!")
                elif t.abs().max() > 1e6:
                    print(f"    ⚠️ WARNING: Scale is very large!")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_resq_weights.py <preprocessed_checkpoint_path>")
        print("  python debug_resq_weights.py <preprocessed_path> <original_path>")
        sys.exit(1)
    
    preprocessed_path = sys.argv[1]
    check_preprocessed_checkpoint(preprocessed_path)
    
    if len(sys.argv) >= 3:
        original_path = sys.argv[2]
        check_original_checkpoint(original_path)


if __name__ == "__main__":
    main()

