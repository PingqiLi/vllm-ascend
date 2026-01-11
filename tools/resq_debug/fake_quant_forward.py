"""
Verify ResQ forward using original paper's fake quantization approach

This script loads the original model and applies ResQ transforms dynamically
(fake quantization), then compares with our ResQ model implementation.

The key insight is that project-resq uses:
- Activation: x' = matmul_hadU_cuda(x, hadK, K)  → x @ kron(hadK, H) / sqrt(n)
- Weight: W' = matmul_hadU_cuda(W, K * inv(hadK).T, K)

Usage:
    python -m tools.resq_debug.fake_quant_forward \
        --model /path/to/qwen3-bf16 \
        --transforms /path/to/transforms_B.pt \
        --prompt "Hello" \
        --device cpu
"""

import argparse
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def hadamard_transform_cpu(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform on the last dimension
    
    Input shape: [..., n] where n is a power of 2
    Output: H @ x where H is the Hadamard matrix (unnormalized)
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    
    # Recursive Hadamard via butterfly operations
    h = 1
    while h < n:
        # Split into pairs
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], n)
        h *= 2
    
    return x


def matmul_hadU(X: torch.Tensor, hadK: torch.Tensor, K: int) -> torch.Tensor:
    """
    Original paper's Hadamard transform for activations.
    
    Equivalent to: X @ kron(hadK, H_butterfly).T / sqrt(n)
    
    Args:
        X: Input tensor [..., n]
        hadK: K x K matrix (Hd in our notation)
        K: Number of blocks
    """
    n = X.shape[-1]
    blocksize = n // K
    
    if K == 1:
        return hadamard_transform_cpu(X) / math.sqrt(n)
    
    # Reshape to [batch, K, blocksize]
    input_reshaped = X.view(-1, K, blocksize)
    
    # Apply Hadamard to each block (blocksize dimension)
    input_had = hadamard_transform_cpu(input_reshaped) / math.sqrt(n)
    
    # Apply hadK across K dimension: [K, K] @ [batch, K, blocksize] -> [batch, K, blocksize]
    input_rotated = torch.einsum('ij,bjk->bik', hadK.to(input_had.dtype), input_had)
    
    return input_rotated.reshape(X.shape)


class FakeQuantWrapper(nn.Module):
    """
    Wrapper that applies Hadamard transform before a linear layer (for down_proj)
    This mimics project-resq's ActQuantWrapper with online_full_had=True
    """
    def __init__(self, linear: nn.Linear, hadK: torch.Tensor, K: int):
        super().__init__()
        self.linear = linear
        self.hadK = hadK
        self.K = K
        self.activations = {}
        self._save = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._save:
            self.activations['input'] = x.clone()
        
        # Apply Hadamard transform (like project-resq)
        x_had = matmul_hadU(x, self.hadK, self.K)
        
        if self._save:
            self.activations['after_had'] = x_had.clone()
        
        # Apply linear
        out = self.linear(x_had)
        
        if self._save:
            self.activations['output'] = out.clone()
        
        return out


class QKRotationWrapper(nn.Module):
    """
    Wrapper that applies rotation after RoPE for Q/K
    This mimics project-resq's QKRotationWrapper
    """
    def __init__(self, Uc: torch.Tensor):
        super().__init__()
        self.Uc = Uc
    
    def apply_rotation(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Uc rotation to Q and K after RoPE"""
        q_rot = torch.matmul(q, self.Uc.to(q.dtype))
        k_rot = torch.matmul(k, self.Uc.to(k.dtype))
        return q_rot, k_rot


def inject_resq_transforms(model: nn.Module, transforms_path: str, device: str = "cpu"):
    """
    Inject ResQ transforms into a standard Qwen3 model (fake quantization style)
    
    This modifies the model in-place to add:
    1. Hadamard transforms before down_proj
    2. Uc rotation after RoPE for Q/K
    """
    transforms = torch.load(transforms_path, map_location='cpu')
    
    # Get global Hadamard params
    Hd = transforms.get('resq.Hd')
    K = int(transforms.get('resq.Hd_K', torch.tensor(100)).item())
    blocksize = int(transforms.get('resq.down_proj_blocksize', torch.tensor(256)).item())
    
    print(f"Injecting ResQ transforms: K={K}, blocksize={blocksize}")
    print(f"  Hd shape: {Hd.shape if Hd is not None else 'None'}")
    
    # Store wrappers for access
    model._resq_down_proj_wrappers = {}
    model._resq_uc_matrices = {}
    
    for i, layer in enumerate(model.model.layers):
        # Get per-layer transforms
        Uc = transforms.get(f'resq.layer.{i}.Uc')
        Pd = transforms.get(f'resq.layer.{i}.Pd')  # Not used in simple Hadamard
        
        # Store Uc for manual application
        if Uc is not None:
            model._resq_uc_matrices[i] = Uc.to(device)
        
        # Wrap down_proj with Hadamard transform
        if Hd is not None:
            original_down_proj = layer.mlp.down_proj
            wrapper = FakeQuantWrapper(original_down_proj, Hd.to(device), K)
            layer.mlp.down_proj = wrapper
            model._resq_down_proj_wrappers[i] = wrapper
            print(f"  Layer {i}: wrapped down_proj with Hadamard")
    
    return model


def compare_with_resq_model(
    original_model: nn.Module,
    resq_model_path: str,  # Our ResQ model checkpoint
    transforms_path: str,
    prompt: str,
    device: str = "cpu",
):
    """
    Compare:
    1. Original model + fake quant transforms
    2. Our ResQ model implementation
    """
    from .modeling_qwen3_resq import Qwen3ResQForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(resq_model_path, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: '{prompt}'")
    print("=" * 60)
    
    # === Method 1: Original + Fake Quant ===
    print("\n[1] Original Model + Fake Quant Transforms")
    
    # Inject transforms
    inject_resq_transforms(original_model, transforms_path, device)
    
    # Enable activation saving
    for wrapper in original_model._resq_down_proj_wrappers.values():
        wrapper._save = True
    
    with torch.no_grad():
        out1 = original_model(inputs["input_ids"])
        logits1 = out1.logits if hasattr(out1, 'logits') else out1
    
    # Get activations
    fake_quant_acts = {}
    for i, wrapper in original_model._resq_down_proj_wrappers.items():
        fake_quant_acts[f'layer_{i}_mlp'] = wrapper.activations.copy()
    
    print(f"  Logits shape: {logits1.shape}")
    print(f"  Top-5 tokens: {[tokenizer.decode([t]) for t in logits1[0, -1].topk(5).indices.tolist()]}")
    
    # === Method 2: Our ResQ Model ===
    print("\n[2] Our ResQ Model Implementation")
    print("  (Would need ckpt_a with quantized weights)")
    
    # For now, just show the comparison would happen here
    # resq_model = Qwen3ResQForCausalLM.from_resq_checkpoint(...)
    
    return {
        'fake_quant_logits': logits1.cpu(),
        'fake_quant_activations': fake_quant_acts,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify ResQ forward with fake quantization")
    parser.add_argument("--model", required=True, help="Path to original Qwen3 model")
    parser.add_argument("--transforms", required=True, help="Path to transforms_B.pt")
    parser.add_argument("--prompt", default="Hello", help="Test prompt")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens to generate")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Inject transforms
    print(f"\nInjecting ResQ transforms from {args.transforms}...")
    inject_resq_transforms(model, args.transforms, args.device)
    
    # Enable activation saving
    for wrapper in model._resq_down_proj_wrappers.values():
        wrapper._save = True
    
    # Run forward
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    
    print(f"\nPrompt: '{args.prompt}'")
    print("=" * 60)
    
    with torch.no_grad():
        output = model(inputs["input_ids"])
        logits = output.logits if hasattr(output, 'logits') else output
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Top-5 next tokens: {[tokenizer.decode([t]) for t in logits[0, -1].topk(5).indices.tolist()]}")
    
    # Show activations from first layer
    if 0 in model._resq_down_proj_wrappers:
        acts = model._resq_down_proj_wrappers[0].activations
        print(f"\nLayer 0 down_proj activations:")
        print(f"  input shape: {acts['input'].shape}, norm: {acts['input'].float().norm():.4f}")
        print(f"  after_had shape: {acts['after_had'].shape}, norm: {acts['after_had'].float().norm():.4f}")
        print(f"  output shape: {acts['output'].shape}, norm: {acts['output'].float().norm():.4f}")
        
        # Check scale change from Hadamard
        input_norm = acts['input'].float().norm()
        had_norm = acts['after_had'].float().norm()
        scale_ratio = had_norm / input_norm
        n = acts['input'].shape[-1]
        expected_scale = 1.0  # After normalization by sqrt(n), scale should be ~1
        print(f"\n  Hadamard scale: {scale_ratio:.4f} (expected ~{expected_scale:.4f})")
    
    # Generate tokens
    print(f"\n--- Generation (max {args.max_tokens} tokens) ---")
    generated = []
    current_ids = inputs["input_ids"]
    
    for _ in range(args.max_tokens):
        with torch.no_grad():
            out = model(current_ids)
            logits = out.logits if hasattr(out, 'logits') else out
            next_token = logits[0, -1].argmax().item()
        
        generated.append(next_token)
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=args.device)], dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    print(f"Generated: '{tokenizer.decode(generated, skip_special_tokens=True)}'")


if __name__ == "__main__":
    main()
