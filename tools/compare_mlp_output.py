#!/usr/bin/env python3
"""
Compare MLP outputs between original model and ResQ-transformed weights.

This script loads BOTH the original model and ResQ weights, then compares:
1. Original: y = intermediate @ W_down.T
2. ResQ: y' = (intermediate @ Pd.T @ H) @ W_down_resq.T

If the transformation is correct, y and y' should be very close (after accounting for Ua).

Usage:
    python compare_mlp_output.py \
        --original-model /path/to/original/model \
        --resq-weights /path/to/resq/safetensors
"""

import argparse
import math
import torch
from pathlib import Path
from safetensors import safe_open


def load_resq_weights(resq_path):
    """Load ResQ weights from safetensors file."""
    weights = {}
    with safe_open(resq_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def hadamard_transform(x):
    """Apply fast Hadamard transform to last dimension (in-place butterfly)."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Dimension {n} must be power of 2"
    
    x = x.clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j].clone()
                b = x[..., j + h].clone()
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2
    return x


def matmul_hadU_cpu_simple(X, hadK, K):
    """
    Replicate msmodelslim's matmul_hadU_cpu exactly.
    
    From msmodelslim/hadamard_utils.py:
        input_tensor = X.view(-1, K, n // K)
        input_tensor = HadamardTransform.apply(input_tensor) / sqrt(n)
        input_tensor = hadK @ input_tensor
    """
    n = X.shape[-1]
    original_shape = X.shape
    blocksize = n // K
    
    # Reshape: [..., n] -> [-1, K, blocksize]
    X = X.reshape(-1, K, blocksize).float()
    
    # Apply Hadamard to last dim (blocksize)
    X = hadamard_transform(X)
    
    # Normalize by sqrt(n)
    X = X / math.sqrt(n)
    
    # Apply hadK: [K, K] @ [-1, K, blocksize]
    hadK = hadK.float()
    X = torch.einsum('ij,bjk->bik', hadK, X)
    
    return X.reshape(original_shape)


def apply_resq_transform(x, Pd, Hd, Hd_K, blocksize, compensate=True):
    """
    Apply ResQ Hadamard rotation: x' = x @ Pd.T @ H
    
    This should match msmodelslim's weight transformation.
    
    Args:
        compensate: If True, multiply by sqrt(K) to compensate for msmodelslim's
                   extra normalization on Hd. This is needed because msmodelslim's
                   Hd is normalized by 1/sqrt(K) but original ResQ's is not.
    """
    n = x.shape[-1]
    K = Hd_K
    
    original_shape = x.shape
    x = x.float()
    
    # Step 1: Apply Pd.T to each block
    if Pd is not None:
        num_blocks = n // blocksize
        x = x.reshape(*original_shape[:-1], num_blocks, blocksize)
        x = torch.matmul(x, Pd.T.float())
        x = x.reshape(*original_shape)
    
    # Step 2: Apply H using matmul_hadU_cpu
    if Hd is not None and K > 1:
        x = matmul_hadU_cpu_simple(x, Hd, K)
        # Compensate for msmodelslim's extra 1/sqrt(K) normalization on Hd
        # BOTH weights and activations are scaled by 1/sqrt(K), so multiply by K
        if compensate:
            x = x * K
    else:
        # Pure power-of-2 Hadamard
        x = hadamard_transform(x) / math.sqrt(n)
    
    return x


def main():
    parser = argparse.ArgumentParser(description="Compare MLP outputs")
    parser.add_argument("--original-model", type=str, required=True, help="Path to original bf16 model")
    parser.add_argument("--resq-weights", type=str, required=True, help="Path to ResQ safetensors file")
    parser.add_argument("--transforms", type=str, 
                        default="/data1/zhonghan/weights/Qwen3-32B-w4a4-transform/resq_transforms.safetensors",
                        help="Path to resq_transforms.safetensors file containing Ua (R_a)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    args = parser.parse_args()

    print("=" * 80)
    print("MLP Output Comparison: Original vs ResQ")
    print("=" * 80)
    
    # Load original model
    print(f"\nLoading original model from {args.original_model}...")
    from transformers import AutoModelForCausalLM
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Load ResQ weights
    print(f"Loading ResQ weights from {args.resq_weights}...")
    resq_weights = load_resq_weights(args.resq_weights)
    
    # Get layer info
    layer_idx = args.layer
    layer = original_model.model.layers[layer_idx]
    
    # Get original down_proj weight
    W_orig = layer.mlp.down_proj.weight.data.float()  # [hidden_dim, intermediate_size]
    print(f"\nOriginal down_proj weight: {W_orig.shape}")
    print(f"  norm per row (mean): {W_orig.norm(dim=1).mean():.4f}")
    
    # Get ResQ down_proj weight
    resq_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    if resq_key not in resq_weights:
        print(f"ERROR: {resq_key} not found in ResQ weights")
        print(f"Available keys: {[k for k in resq_weights.keys() if 'down_proj' in k][:5]}")
        return
    
    W_resq = resq_weights[resq_key].float()
    print(f"\nResQ down_proj weight: {W_resq.shape}")
    print(f"  norm per row (mean): {W_resq.norm(dim=1).mean():.4f}")
    
    # Get ResQ rotation matrices
    # Try both old and new key formats (preprocess_resq_weights.py renames keys)
    Pd_key_old = f"resq.layer.{layer_idx}.Pd"
    Pd_key_new = f"model.layers.{layer_idx}.mlp.rotation_Pd"
    
    Pd = resq_weights.get(Pd_key_new)
    if Pd is None:
        Pd = resq_weights.get(Pd_key_old)
    if Pd is not None:
        print(f"\nPd matrix: {Pd.shape}")
        # Check orthogonality
        orth_err = (Pd.float() @ Pd.float().T - torch.eye(Pd.shape[0])).abs().max().item()
        print(f"  Orthogonality error: {orth_err:.6f}")
    else:
        print(f"\nWARNING: Pd not found (tried {Pd_key_new} and {Pd_key_old})")
        # List available keys for debugging
        pd_keys = [k for k in resq_weights.keys() if 'Pd' in k or 'rotation' in k]
        print(f"  Available rotation keys: {pd_keys[:10]}")
    
    Hd = resq_weights.get("resq.Hd")
    Hd_K = resq_weights.get("resq.Hd_K")
    if Hd_K is not None:
        Hd_K = Hd_K.item()
    else:
        Hd_K = 1
    
    # Load Ua (global rotation matrix) from separate transforms file
    Ua = None
    if args.transforms and Path(args.transforms).exists():
        print(f"\nLoading Ua from {args.transforms}...")
        transforms = load_resq_weights(args.transforms)
        # Ua is stored as resq.layer.X.R_a (same for all layers, use layer 0)
        Ua_key = f"resq.layer.{layer_idx}.R_a"
        Ua = transforms.get(Ua_key)
        if Ua is not None:
            print(f"Ua matrix ({Ua_key}): {Ua.shape}")
            # Check orthogonality
            orth_err_Ua = (Ua.float() @ Ua.float().T - torch.eye(Ua.shape[0])).abs().max().item()
            print(f"  Orthogonality error: {orth_err_Ua:.6f}")
        else:
            print(f"WARNING: {Ua_key} not found in transforms file")
            print(f"  Available keys: {[k for k in transforms.keys() if 'R_a' in k][:5]}")
    else:
        print(f"\nWARNING: Transforms file not found: {args.transforms}")
    
    if Hd is not None:
        print(f"Hd matrix: {Hd.shape}, K={Hd_K}")
        print(f"  max_abs: {Hd.abs().max():.4f}")
    
    blocksize_tensor = resq_weights.get("resq.down_proj_blocksize")
    blocksize = blocksize_tensor.item() if blocksize_tensor is not None else 256
    print(f"Blocksize: {blocksize}")
    
    # Get dimensions
    hidden_size = W_orig.shape[0]
    intermediate_size = W_orig.shape[1]
    
    # Create a random intermediate activation for testing
    batch_size = 1
    seq_len = 8
    
    print(f"\n" + "-" * 80)
    print("Testing with random intermediate activation")
    print("-" * 80)
    
    # Random activation (similar range to real activations)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, intermediate_size) * 0.1  # Scale to realistic range
    
    print(f"Input x: shape={x.shape}, norm={x.norm():.4f}")
    
    # Original computation: y = x @ W_orig.T
    y_orig = torch.matmul(x.float(), W_orig.T)
    print(f"\nOriginal output: shape={y_orig.shape}")
    print(f"  norm={y_orig.norm():.4f}, min={y_orig.min():.4f}, max={y_orig.max():.4f}")
    
    # Step-by-step ResQ transformation for debugging
    print(f"\n--- Step-by-step transformation ---")
    
    x_f32 = x.float()
    print(f"Step 0 (input): norm={x_f32.norm():.4f}")
    
    # Step 1: Pd.T
    if Pd is not None:
        num_blocks = intermediate_size // blocksize
        x_step1 = x_f32.reshape(batch_size, seq_len, num_blocks, blocksize)
        x_step1 = torch.matmul(x_step1, Pd.T.float())
        x_step1 = x_step1.reshape(batch_size, seq_len, intermediate_size)
        print(f"Step 1 (after Pd.T): norm={x_step1.norm():.4f}")
    else:
        x_step1 = x_f32
    
    # Step 2: H_butterfly
    x_step2 = x_step1.reshape(-1, Hd_K, blocksize)
    x_step2 = hadamard_transform(x_step2)
    print(f"Step 2 (after H_butterfly, before norm): norm={x_step2.norm():.4f}")
    
    x_step2 = x_step2 / math.sqrt(intermediate_size)
    print(f"Step 2 (after /sqrt(n)): norm={x_step2.norm():.4f}")
    
    # Step 3: Hd
    if Hd is not None:
        x_step3 = torch.einsum('ij,bjk->bik', Hd.float(), x_step2)
        print(f"Step 3 (after Hd): norm={x_step3.norm():.4f}")
    else:
        x_step3 = x_step2
    
    x_transformed = x_step3.reshape(batch_size, seq_len, intermediate_size)
    print(f"Final transformed (no compensation): norm={x_transformed.norm():.4f}")
    
    # Apply K compensation for msmodelslim's extra 1/sqrt(K) normalization
    # BOTH weights and activations were scaled by 1/sqrt(K), so multiply by K
    x_transformed_compensated = x_transformed * Hd_K
    print(f"Final transformed (with K={Hd_K} compensation): norm={x_transformed_compensated.norm():.4f}")
    
    print(f"--- End transformation ---\n")
    
    # Use compensated transform for the main comparison
    y_resq = torch.matmul(x_transformed_compensated.float(), W_resq.T)
    print(f"ResQ output: shape={y_resq.shape}")
    print(f"  norm={y_resq.norm():.4f}, min={y_resq.min():.4f}, max={y_resq.max():.4f}")
    
    # Compare (with K compensation)
    print(f"\n" + "=" * 80)
    print(f"Comparison (WITH K={Hd_K} compensation)")
    print("=" * 80)
    
    diff = (y_orig - y_resq).abs()
    relative_diff = diff / (y_orig.abs() + 1e-8)
    
    print(f"Absolute difference:")
    print(f"  max: {diff.max():.6f}")
    print(f"  mean: {diff.mean():.6f}")
    
    print(f"\nRelative difference:")
    print(f"  max: {relative_diff.max():.6f}")
    print(f"  mean: {relative_diff.mean():.6f}")
    
    # Cosine similarity
    y_orig_flat = y_orig.flatten()
    y_resq_flat = y_resq.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(y_orig_flat.unsqueeze(0), y_resq_flat.unsqueeze(0))
    print(f"\nCosine similarity: {cos_sim.item():.6f}")
    
    if cos_sim.item() > 0.99:
        print("\n✓ Outputs are very similar (cosine > 0.99)")
    elif cos_sim.item() > 0.9:
        print("\n⚠ Outputs are somewhat similar (cosine > 0.9)")
    else:
        print("\n✗ Outputs are very different (cosine < 0.9) - TRANSFORMATION IS WRONG!")
    
    # Scale factor analysis
    scale = (y_resq.norm() / y_orig.norm()).item()
    print(f"\nScale factor (ResQ/Original): {scale:.4f}")
    if abs(scale - 1.0) > 0.1:
        print(f"  WARNING: Significant scale difference detected!")
    
    # Additional test: what if we DON'T apply K compensation?
    print(f"\n" + "=" * 80)
    print("Test: Without K compensation (for comparison)")
    print("=" * 80)
    
    y_resq_no_comp = torch.matmul(x_transformed.float(), W_resq.T)  # x_transformed has no compensation
    print(f"y_resq (no compensation): norm={y_resq_no_comp.norm():.4f}")
    scale_no_comp = (y_resq_no_comp.norm() / y_orig.norm()).item()
    print(f"Scale (no compensation): {scale_no_comp:.4f}")
    print(f"  Expected without compensation: 1/K = {1/Hd_K:.4f}")
    
    cos_no_comp = torch.nn.functional.cosine_similarity(
        y_orig.flatten().unsqueeze(0), 
        y_resq_no_comp.flatten().unsqueeze(0)
    )
    print(f"Cosine similarity (no compensation): {cos_no_comp.item():.6f}")
    
    # TEST: Verify that y_resq ≈ y_orig @ Ua
    # Because W_resq = Ua.T @ W_orig @ Pd.T @ H, the output is:
    # y_resq = (x @ Pd.T @ H) @ W_resq.T = (x @ Pd.T @ H) @ (H.T @ Pd @ W_orig.T @ Ua)
    #        = x @ (Pd.T @ H @ H.T @ Pd) @ W_orig.T @ Ua
    #        = x @ W_orig.T @ Ua  (if H @ H.T = I and Pd @ Pd.T = I)
    print(f"\n" + "=" * 80)
    print("KEY TEST: Verify y_resq = y_orig @ Ua")
    print("=" * 80)
    
    if Ua is not None:
        # Test BOTH Ua and Ua.T to determine which is correct
        # The saved R_a might be Ua or Ua.T depending on convention
        
        # Option 1: y_orig @ Ua (if R_a = Ua)
        y_expected_1 = torch.matmul(y_orig.float(), Ua.float())
        cos_1 = torch.nn.functional.cosine_similarity(
            y_expected_1.flatten().unsqueeze(0),
            y_resq.flatten().unsqueeze(0)
        ).item()
        scale_1 = (y_resq.norm() / y_expected_1.norm()).item()
        
        # Option 2: y_orig @ Ua.T (if R_a = Ua.T, i.e., saved as transposed)
        y_expected_2 = torch.matmul(y_orig.float(), Ua.T.float())
        cos_2 = torch.nn.functional.cosine_similarity(
            y_expected_2.flatten().unsqueeze(0),
            y_resq.flatten().unsqueeze(0)
        ).item()
        scale_2 = (y_resq.norm() / y_expected_2.norm()).item()
        
        print(f"Option 1 (y_orig @ Ua):   cosine={cos_1:.6f}, scale={scale_1:.4f}")
        print(f"Option 2 (y_orig @ Ua.T): cosine={cos_2:.6f}, scale={scale_2:.4f}")
        
        if cos_1 > 0.99 and abs(scale_1 - 1.0) < 0.1:
            print("\n✓ SUCCESS! y_resq ≈ y_orig @ Ua (R_a stored as Ua directly)")
        elif cos_2 > 0.99 and abs(scale_2 - 1.0) < 0.1:
            print("\n✓ SUCCESS! y_resq ≈ y_orig @ Ua.T (R_a stored as Ua.T)")
            print("  --> Need to use Ua.T in comparison/inference!")
        else:
            print("\n✗ PROBLEM! Neither matches - check transformation logic")
    else:
        print("Cannot test: Ua not available")
    
    # Test: transform W_resq back to original space
    print(f"\n" + "=" * 80)
    print("Test: Transform W_resq back to check if it matches W_orig")
    print("=" * 80)
    
    # If W_resq = Ua.T @ W_orig @ Pd.T @ H, then to get back:
    # W_orig = Ua @ W_resq @ H^{-1} @ Pd
    # For H where H @ H.T = I/K, we have H^{-1} = K * H.T
    
    print(f"W_orig row norm mean: {W_orig.norm(dim=1).mean():.4f}")
    print(f"W_resq row norm mean: {W_resq.norm(dim=1).mean():.4f}")
    print(f"Ratio: {W_resq.norm(dim=1).mean() / W_orig.norm(dim=1).mean():.4f}")
    
    # Direct weight verification: Ua @ W_resq @ H^{-1} @ Pd should equal W_orig
    # But if R_a = Ua.T, then we need Ua.T @ W_resq @ ... = W_orig
    if Ua is not None and Pd is not None and Hd is not None:
        print(f"\n--- Direct weight transformation verification ---")
        print("Testing both Ua and Ua.T as the first step...")
        
        # Test with Ua
        W_step1_v1 = torch.matmul(Ua.float(), W_resq.float())
        # Test with Ua.T  
        W_step1_v2 = torch.matmul(Ua.T.float(), W_resq.float())
        
        print(f"Step 1a (Ua @ W_resq): norm={W_step1_v1.norm(dim=1).mean():.4f}")
        print(f"Step 1b (Ua.T @ W_resq): norm={W_step1_v2.norm(dim=1).mean():.4f}")
        
        # Continue with both paths
        W_step1 = W_step1_v1  # Default to Ua first
        
        # Step 2: Apply H^{-1} = K * H.T to each row
        # W_step2[i] = W_step1[i] @ (K * H.T) = K * H(W_step1[i])
        # Using our matmul_hadU which applies H, then multiply by K
        W_step2 = matmul_hadU_cpu_simple(W_step1, Hd, Hd_K) * Hd_K
        print(f"Step 2 (@ H^{{-1}} = K*H.T): norm={W_step2.norm(dim=1).mean():.4f}")
        
        # Step 3: Apply Pd to each block
        num_blocks = intermediate_size // blocksize
        W_step3 = W_step2.reshape(hidden_size, num_blocks, blocksize)
        W_step3 = torch.matmul(W_step3, Pd.float())  # @ Pd (not Pd.T!)
        W_step3 = W_step3.reshape(hidden_size, intermediate_size)
        print(f"Step 3 (@ Pd): norm={W_step3.norm(dim=1).mean():.4f}")
        
        # Compare with W_orig
        cos_W = torch.nn.functional.cosine_similarity(
            W_orig.flatten().unsqueeze(0),
            W_step3.flatten().unsqueeze(0)
        )
        print(f"\nCosine similarity (W_recovered vs W_orig): {cos_W.item():.6f}")
        
        scale_W = (W_step3.norm() / W_orig.norm()).item()
        print(f"Scale factor: {scale_W:.4f}")
        
        if cos_W.item() > 0.99:
            print("✓ Weight transformation is CORRECT! (using Ua)")
        else:
            print("✗ Weight transformation with Ua is WRONG!")
            
            # Try with Ua.T instead
            print("\n--- Testing with Ua.T instead ---")
            W_step2_alt = matmul_hadU_cpu_simple(W_step1_v2, Hd, Hd_K) * Hd_K
            W_step3_alt = W_step2_alt.reshape(hidden_size, num_blocks, blocksize)
            W_step3_alt = torch.matmul(W_step3_alt, Pd.float())
            W_step3_alt = W_step3_alt.reshape(hidden_size, intermediate_size)
            cos_W_alt = torch.nn.functional.cosine_similarity(
                W_orig.flatten().unsqueeze(0),
                W_step3_alt.flatten().unsqueeze(0)
            )
            print(f"  Cosine (Ua.T @ W_resq @ H^-1 @ Pd): {cos_W_alt.item():.6f}")
            
            if cos_W_alt.item() > 0.99:
                print("  ✓ Using Ua.T works! R_a is stored as transposed.")
            else:
                # Debug: try different H inverse formulations
                print("\n--- Debug: trying different H^{-1} formulations ---")
                
                # Try H.T directly (no K scaling)
                W_step2_v2 = matmul_hadU_cpu_simple(W_step1, Hd.T, Hd_K)
                W_step3_v2 = W_step2_v2.reshape(hidden_size, num_blocks, blocksize)
                W_step3_v2 = torch.matmul(W_step3_v2, Pd.float())
                W_step3_v2 = W_step3_v2.reshape(hidden_size, intermediate_size)
                cos_W_v2 = torch.nn.functional.cosine_similarity(
                    W_orig.flatten().unsqueeze(0),
                    W_step3_v2.flatten().unsqueeze(0)
                )
                print(f"  Using H.T (no K): cosine={cos_W_v2.item():.6f}")
                
                # Try sqrt(K) scaling
                W_step2_v3 = matmul_hadU_cpu_simple(W_step1, Hd, Hd_K) * math.sqrt(Hd_K)
                W_step3_v3 = W_step2_v3.reshape(hidden_size, num_blocks, blocksize)
                W_step3_v3 = torch.matmul(W_step3_v3, Pd.float())
                W_step3_v3 = W_step3_v3.reshape(hidden_size, intermediate_size)
                cos_W_v3 = torch.nn.functional.cosine_similarity(
                    W_orig.flatten().unsqueeze(0),
                    W_step3_v3.flatten().unsqueeze(0)
                )
                print(f"  Using sqrt(K) scaling: cosine={cos_W_v3.item():.6f}")
    
    # KEY TEST: With compensation (multiply by sqrt(K))
    print(f"\n" + "=" * 80)
    print("KEY FIX TEST: With sqrt(K) compensation")
    print("=" * 80)
    print(f"  msmodelslim's Hd is normalized by 1/sqrt(K), original ResQ's is not.")
    print(f"  BOTH weights and activations are scaled by 1/sqrt(K), total = 1/K")
    print(f"  We need to multiply activations by K = {Hd_K} to compensate")
    
    # Apply transformation WITH compensation
    x_transformed_comp = apply_resq_transform(x, Pd, Hd, Hd_K, blocksize, compensate=True)
    y_resq_comp = torch.matmul(x_transformed_comp, W_resq.T)
    
    print(f"\nWith compensation:")
    print(f"  x_transformed norm: {x_transformed_comp.norm():.4f} (was {x_transformed.norm():.4f})")
    print(f"  y_resq norm: {y_resq_comp.norm():.4f} (was {y_resq.norm():.4f})")
    
    cos_sim_comp = torch.nn.functional.cosine_similarity(
        y_orig.flatten().unsqueeze(0), 
        y_resq_comp.flatten().unsqueeze(0)
    )
    print(f"  Cosine similarity: {cos_sim_comp.item():.6f}")
    
    scale_comp = (y_resq_comp.norm() / y_orig.norm()).item()
    print(f"  Scale factor: {scale_comp:.4f} (should be ~1.0 if Ua is identity)")
    
    if cos_sim_comp.item() > 0.99:
        print("\n  ✓ COMPENSATION WORKS! Outputs match (ignoring Ua rotation)")
    elif cos_sim_comp.item() > 0.9:
        print("\n  ⚠ Partially fixed, but still has issues")
    else:
        print("\n  ✗ Compensation alone doesn't fix it - Ua rotation is the culprit")
        print("    (This is expected - the random input isn't in Ua space)")


if __name__ == "__main__":
    main()

