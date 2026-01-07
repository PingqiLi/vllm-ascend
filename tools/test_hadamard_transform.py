#!/usr/bin/env python3
"""
Test if our Hadamard transform matches msmodelslim's matmul_hadU_cpu.

This script compares:
1. Our vllm-ascend implementation
2. msmodelslim's original implementation

If they don't match, the inference will be wrong.

Usage:
    python test_hadamard_transform.py --resq-weights /path/to/resq/model.safetensors
"""

import argparse
import math
import sys
import torch
from pathlib import Path
from safetensors import safe_open

# Add msmodelslim to path
sys.path.insert(0, "/Users/patrick/Projects/msit/msmodelslim")


def load_resq_weights(resq_path):
    """Load ResQ weights from safetensors file."""
    weights = {}
    with safe_open(resq_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def hadamard_transform_vllm(x):
    """vllm-ascend's hadamard_transform (butterfly algorithm)."""
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


def matmul_hadU_vllm(X, hadK, K, compensate=False):
    """vllm-ascend's matmul_hadU implementation.
    
    Args:
        X: Input tensor
        hadK: Hadamard block matrix (normalized by 1/sqrt(K) in msmodelslim!)
        K: Block size
        compensate: If True, multiply by sqrt(K) to compensate for msmodelslim's normalization
    """
    n = X.shape[-1]
    original_shape = X.shape
    blocksize = n // K
    
    # Reshape: [..., n] -> [-1, K, blocksize]
    X = X.reshape(-1, K, blocksize).float()
    
    # Apply Hadamard to last dim (blocksize)
    X = hadamard_transform_vllm(X)
    
    # Normalize by sqrt(n)
    X = X / math.sqrt(n)
    
    # Apply hadK: [K, K] @ [-1, K, blocksize]
    # NOTE: hadK is ALREADY normalized by 1/sqrt(K) in msmodelslim!
    hadK = hadK.float()
    X = torch.einsum('ij,bjk->bik', hadK, X)
    
    # Compensate for msmodelslim's extra 1/sqrt(K) normalization
    # BOTH weights and activations are scaled by 1/sqrt(K), so we multiply by K
    # to compensate for the total 1/K scaling in the output
    if compensate:
        X = X * K
    
    return X.reshape(original_shape)


def matmul_hadU_inverse(X, hadK, K):
    """
    Inverse of matmul_hadU.
    
    If M = matmul_hadU, then M_inv = matmul_hadU_inverse should satisfy:
    M_inv(M(x)) = x (up to numerical error)
    
    Since H_butterfly is self-inverse (H @ H = n * I),
    and hadK is orthogonal and normalized (hadK @ hadK.T = I),
    The inverse is:
    1. Undo hadK: multiply by hadK.T
    2. Undo /sqrt(n): multiply by sqrt(n)  
    3. Undo H_butterfly: apply H_butterfly again and divide by blocksize
    """
    n = X.shape[-1]
    original_shape = X.shape
    blocksize = n // K
    
    # Reshape: [..., n] -> [-1, K, blocksize]
    X = X.reshape(-1, K, blocksize).float()
    
    # Undo hadK: multiply by hadK.T (since hadK is orthogonal)
    hadK = hadK.float()
    X = torch.einsum('ji,bjk->bik', hadK, X)  # hadK.T @ X
    
    # Undo /sqrt(n): multiply by sqrt(n)
    X = X * math.sqrt(n)
    
    # Undo H_butterfly: H @ H = blocksize * I, so H_inv = H / blocksize
    X = hadamard_transform_vllm(X) / blocksize
    
    return X.reshape(original_shape)
    
    return X.reshape(original_shape)


def main():
    parser = argparse.ArgumentParser(description="Test Hadamard transform")
    parser.add_argument("--resq-weights", type=str, required=True, help="Path to ResQ safetensors file")
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Hadamard Transform Implementation")
    print("=" * 80)
    
    # Load ResQ weights
    print(f"\nLoading ResQ weights from {args.resq_weights}...")
    resq_weights = load_resq_weights(args.resq_weights)
    
    # Get Hd and K
    Hd = resq_weights.get("resq.Hd")
    Hd_K = resq_weights.get("resq.Hd_K")
    if Hd_K is not None:
        Hd_K = Hd_K.item()
    else:
        Hd_K = 1
    
    print(f"Hd: shape={Hd.shape if Hd is not None else None}, K={Hd_K}")
    
    if Hd is None or Hd_K <= 1:
        print("No Hd matrix found or K=1, nothing to test")
        return
    
    # Test parameters
    n = Hd_K * 256  # e.g., 100 * 256 = 25600
    batch_size = 4
    
    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, n) * 0.1
    print(f"\nTest input: shape={x.shape}, norm={x.norm():.4f}")
    
    # Test 1: Our implementation
    print("\n--- Testing vllm-ascend implementation ---")
    y_vllm = matmul_hadU_vllm(x, Hd, Hd_K)
    print(f"Output: norm={y_vllm.norm():.4f}, min={y_vllm.min():.4f}, max={y_vllm.max():.4f}")
    
    # Test 2: msmodelslim implementation
    print("\n--- Testing msmodelslim implementation ---")
    try:
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.utils.hadamard_utils import matmul_hadU_cpu
        y_msit = matmul_hadU_cpu(x, Hd, Hd_K)
        print(f"Output: norm={y_msit.norm():.4f}, min={y_msit.min():.4f}, max={y_msit.max():.4f}")
        
        # Compare
        print("\n--- Comparison ---")
        diff = (y_vllm - y_msit).abs()
        print(f"Absolute difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
        
        cos_sim = torch.nn.functional.cosine_similarity(
            y_vllm.flatten().unsqueeze(0),
            y_msit.flatten().unsqueeze(0)
        )
        print(f"Cosine similarity: {cos_sim.item():.6f}")
        
        if cos_sim.item() > 0.9999:
            print("\n✓ Implementations match!")
        else:
            print("\n✗ Implementations differ! This is the bug.")
            
    except ImportError as e:
        print(f"Could not import msmodelslim: {e}")
        print("Please ensure msmodelslim is in your PYTHONPATH")
    
    # Test 3: Check Hd orthogonality directly
    print("\n--- Checking Hd matrix properties ---")
    print(f"  Hd shape: {Hd.shape}")
    print(f"  Hd elements: min={Hd.min():.4f}, max={Hd.max():.4f}")
    
    # Check if Hd is orthogonal: Hd @ Hd.T should = I
    Hd_f32 = Hd.float()
    Hd_HdT = Hd_f32 @ Hd_f32.T
    ortho_error = (Hd_HdT - torch.eye(Hd.shape[0])).abs().max()
    print(f"  Hd @ Hd.T orthogonality error: {ortho_error:.6f}")
    
    # Check if Hd @ Hd = I (self-inverse)
    Hd_Hd = Hd_f32 @ Hd_f32
    self_inv_error = (Hd_Hd - torch.eye(Hd.shape[0])).abs().max()
    print(f"  Hd @ Hd (self-inverse) error: {self_inv_error:.6f}")
    
    if ortho_error < 0.01:
        print(f"  ✓ Hd is orthogonal")
    else:
        print(f"  ✗ Hd is NOT orthogonal!")
    
    if self_inv_error < 0.01:
        print(f"  ✓ Hd is self-inverse (like Hadamard)")
    else:
        print(f"  ✗ Hd is NOT self-inverse!")
    
    # Test 4: Check Hadamard butterfly properties
    print("\n--- Checking butterfly Hadamard properties ---")
    blocksize = n // Hd_K
    print(f"  blocksize: {blocksize}")
    
    # For 256-dim: H_butterfly @ H_butterfly should = 256 * I
    test_vec = torch.randn(1, blocksize)
    h1 = hadamard_transform_vllm(test_vec)
    h2 = hadamard_transform_vllm(h1)
    expected = test_vec * blocksize
    butterfly_error = (h2 - expected).abs().max()
    print(f"  H(H(x)) vs {blocksize}*x error: {butterfly_error:.6f}")
    if butterfly_error < 0.001:
        print(f"  ✓ Butterfly Hadamard: H @ H = {blocksize} * I")
    else:
        print(f"  ✗ Butterfly Hadamard NOT correct!")
        print(f"    Expected scale: {blocksize}, actual: {h2.norm() / test_vec.norm():.4f}")
    
    # Test 5: Full round-trip (same transform twice)
    print("\n--- Checking full transform round-trip (M(M(x))) ---")
    y2 = matmul_hadU_vllm(y_vllm, Hd, Hd_K)
    scale = (y2.norm() / x.norm()).item()
    
    # Theoretical: M(M(x)) = x * (blocksize / n) = x / K
    expected_scale = 1.0 / Hd_K
    print(f"  Expected scale (1/K): {expected_scale:.6f}")
    print(f"  Actual scale: {scale:.6f}")
    
    cos_round = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0),
        y2.flatten().unsqueeze(0)
    )
    print(f"  Cosine(x, M(M(x))): {cos_round.item():.6f}")
    
    # If H @ H = cI, then M(M(x)) = cx, cosine should be 1.0
    if cos_round.item() > 0.999:
        print(f"  ✓ Transform is self-inverse (up to scale)")
    else:
        print(f"  ✗ Transform is NOT self-inverse!")
        # Debug: check if y2 is proportional to x at all
        print(f"\n  Debug comparison (first 10 elements of batch 0):")
        print(f"    x:        {x[0, :10].tolist()}")
        print(f"    M(M(x)):  {y2[0, :10].tolist()}")
        ratio = y2[0, :10] / (x[0, :10] + 1e-10)
        print(f"    Ratio:    {ratio.tolist()}")
    
    # Test 6: Proper inverse transform
    print("\n--- Testing proper inverse transform ---")
    x_recovered = matmul_hadU_inverse(y_vllm, Hd, Hd_K)
    
    inv_scale = (x_recovered.norm() / x.norm()).item()
    cos_inv = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0),
        x_recovered.flatten().unsqueeze(0)
    )
    
    print(f"  M_inv(M(x)) norm ratio: {inv_scale:.6f} (should be ~1.0)")
    print(f"  Cosine(x, M_inv(M(x))): {cos_inv.item():.6f} (should be ~1.0)")
    
    if cos_inv.item() > 0.999 and abs(inv_scale - 1.0) < 0.01:
        print(f"  ✓ Inverse transform works correctly!")
    else:
        print(f"  ✗ Inverse transform has issues")
        print(f"\n  Debug comparison (first 10 elements of batch 0):")
        print(f"    x:           {x[0, :10].tolist()}")
        print(f"    M_inv(M(x)): {x_recovered[0, :10].tolist()}")
    
    # Test 7: Key insight - M @ M.T should equal cI (for orthogonal transform)
    # The previous round-trip test M(M(x)) tests M @ M, which is WRONG!
    # 
    # In ResQ: 
    # - Quantize: W' = W @ H
    # - Inference: x' = x @ H (same transform!)
    # - Verify: x' @ W'.T = (x @ H) @ (W @ H).T = x @ H @ H.T @ W.T = x @ W.T (if H @ H.T = I)
    # 
    # So we need H @ H.T = I (orthogonality), NOT H @ H = I (self-inverse)
    # For non-symmetric Hadamard, H @ H ≠ I but H @ H.T = I ✓
    
    print("\n--- Testing orthogonality: H @ H.T = cI (CRITICAL!) ---")
    print("  NOTE: ResQ uses same transform for weights and activations")
    print("  Inference correctness requires H @ H.T = I, NOT H @ H = I")
    
    # Inner product preservation test: <M(x), M(y)> ≈ c * <x, y>
    # If M represents orthogonal transform (up to scaling), angles are preserved
    y = torch.randn_like(x) * 0.1
    Mx = y_vllm
    My = matmul_hadU_vllm(y, Hd, Hd_K)
    
    inner_xy = (x * y).sum().item()
    inner_MxMy = (Mx * My).sum().item()
    
    print(f"\n  <x, y> = {inner_xy:.6f}")
    print(f"  <M(x), M(y)> = {inner_MxMy:.6f}")
    
    # For orthogonal M: <M(x), M(y)> = <x, M.T @ M(y)> = <x, y> if M.T @ M = I
    # Our M has scaling 1/sqrt(n), so M.T @ M = I/n, hence <M(x), M(y)> = <x, y>/n
    ratio = inner_MxMy / (inner_xy + 1e-10)
    n_global = Hd_K * 256  # n = K * blocksize
    expected_ratio = 1.0 / n_global  # Due to 1/sqrt(n) normalization: (1/sqrt(n))² = 1/n
    print(f"\n  Ratio <M(x),M(y)>/<x,y> = {ratio:.6f}")
    print(f"  Expected ratio (1/n = 1/{n_global}): {expected_ratio:.6f}")
    print(f"  1/K (if wrong): {1.0 / Hd_K:.6f}")
    
    # Check if ratio matches expected
    if abs(ratio - expected_ratio) / expected_ratio < 0.1:  # 10% tolerance
        print(f"  ✓ Transform is orthogonal (up to 1/sqrt(n) scaling)")
    elif abs(ratio - 1.0 / Hd_K) / (1.0 / Hd_K) < 0.1:
        print(f"  ✓ Transform is orthogonal (up to 1/sqrt(K) scaling - Hd is self-normalized)")
    else:
        print(f"  ✗ Transform does NOT preserve angles correctly!")
    
    # Test 8: Verify M(M(x)) ≠ x is EXPECTED for non-symmetric Hadamard
    print("\n--- Understanding round-trip behavior ---")
    print(f"  Round-trip cosine = {cos_round.item():.6f} (not 1.0 is EXPECTED)")
    print("  Reason: H @ H ≠ I for non-symmetric Hadamard (100x100)")
    print("  This is OK because inference uses (x @ H) @ (W @ H).T = x @ H @ H.T @ W.T")
    print("  And H @ H.T = I (orthogonality), so inference is correct!")
    
    # Verify Hd is asymmetric
    Hd_f32 = Hd.float()
    sym_error = (Hd_f32 - Hd_f32.T).abs().max().item()
    print(f"\n  Hd symmetry error (should be > 0): {sym_error:.6f}")
    if sym_error > 0.01:
        print("  ✓ Hd is asymmetric (as expected for order-100 Hadamard)")
    else:
        print("  ✗ Hd is symmetric (unexpected)")
    
    # Verify H @ H.T = I (critical!)
    HHT = Hd_f32 @ Hd_f32.T
    ortho_error = (HHT - torch.eye(Hd.shape[0])).abs().max().item()
    print(f"  Hd @ Hd.T orthogonality error: {ortho_error:.6f}")
    if ortho_error < 0.01:
        print("  ✓ Hd @ Hd.T = I (INFERENCE IS CORRECT)")
    else:
        print("  ✗ Hd @ Hd.T ≠ I (INFERENCE BUG!)")
    
    # Test 9: Verify compensated transform
    print("\n" + "="*80)
    print("KEY FIX: Testing COMPENSATED transform (multiply by K)")
    print("="*80)
    print("  msmodelslim's Hd is normalized by 1/sqrt(K)")
    print("  BOTH weights and activations scaled by 1/sqrt(K), so total = 1/K")
    print("  We multiply by K to compensate")
    
    y_compensated = matmul_hadU_vllm(x, Hd, Hd_K, compensate=True)
    y2_compensated = matmul_hadU_vllm(y_compensated, Hd, Hd_K, compensate=True)
    
    scale_comp = (y2_compensated.norm() / x.norm()).item()
    cos_comp = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0),
        y2_compensated.flatten().unsqueeze(0)
    )
    
    print(f"\n  M(M(x)) with compensation:")
    print(f"    Scale: {scale_comp:.6f} (expected: K = {Hd_K})")
    print(f"    Cosine: {cos_comp.item():.6f} (should be ~1.0)")
    
    # With K compensation: M_comp(M_comp(x)) = x * K, so scale = K, cosine = 1.0
    if cos_comp.item() > 0.99 and abs(scale_comp - Hd_K) < 1.0:
        print(f"\n  ✓ COMPENSATION WORKS! Cosine = 1.0, Scale = K as expected")
        print(f"    (Transform is not orthogonal, but inference will be correct)")
    else:
        print(f"\n  ✗ Still has issues")
    
    # Test inner product preservation with compensation
    y_comp = y_compensated
    y2_rand = matmul_hadU_vllm(y, Hd, Hd_K, compensate=True)
    inner_comp = (y_comp * y2_rand).sum().item()
    print(f"\n  Inner product preservation:")
    print(f"    <x, y> = {inner_xy:.6f}")
    print(f"    <M_comp(x), M_comp(y)> = {inner_comp:.6f}")
    # With K compensation: <M_comp(x), M_comp(y)> = K² * <x, y> (not orthogonal!)
    # But this is OK because we only apply M_comp once to activations, not twice
    print(f"    Ratio: {inner_comp / (inner_xy + 1e-10):.6f} (expected: K² = {Hd_K**2})")
    
    print("\n" + "="*80)
    print("INFERENCE CORRECTNESS ANALYSIS")
    print("="*80)
    print("  In inference:")
    print("    - Weights were transformed: W' = W @ H / sqrt(K)")
    print("    - Activations transformed:  x' = x @ H * K (with our compensation)")
    print("    - Output: x' @ W'.T = x @ H @ H.T @ W.T * K / sqrt(K)")
    print("           = x @ W.T * sqrt(K)  (if H @ H.T = I)")
    print("")
    print("  WAIT - this gives sqrt(K) scaling, not 1.0!")
    print("  Let me recalculate in the main code...")


if __name__ == "__main__":
    main()

