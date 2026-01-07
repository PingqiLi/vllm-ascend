#!/usr/bin/env python3
"""
Check MLP intermediate activation norms for original bf16 model.

Usage:
    python check_mlp_activation.py --model /path/to/original/model

This script loads the original model and records MLP intermediate activations
to compare with ResQ quantized model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Check MLP activation norms")
    parser.add_argument("--model", type=str, required=True, help="Path to original bf16 model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?", help="Test prompt")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max new tokens to generate")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Storage for activation stats
    activation_stats = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # For Qwen3 MLP, output is the result after down_proj
            # We want to capture the intermediate activation (after gate * up, before down_proj)
            # This hook captures the input to down_proj
            pass
        return hook
    
    def make_down_proj_input_hook(layer_idx):
        def hook(module, input, output):
            # input[0] is the intermediate activation going into down_proj
            x = input[0]
            stats = {
                'shape': list(x.shape),
                'norm': x.float().norm().item(),
                'min': x.float().min().item(),
                'max': x.float().max().item(),
                'mean': x.float().mean().item(),
            }
            if layer_idx not in activation_stats:
                activation_stats[layer_idx] = []
            activation_stats[layer_idx].append(stats)
        return hook
    
    # Register hooks on down_proj of each layer
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            hook = layer.mlp.down_proj.register_forward_hook(make_down_proj_input_hook(idx))
            hooks.append(hook)
    
    print(f"\nRegistered hooks on {len(hooks)} layers")
    print(f"\nRunning inference with prompt: '{args.prompt}'")
    
    # Tokenize and run inference
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print activation stats
    print("\n" + "=" * 80)
    print("MLP Intermediate Activation Stats (input to down_proj)")
    print("=" * 80)
    
    # We'll show stats for the first forward pass (prefill)
    print("\n[Prefill Phase - First Token]")
    print(f"{'Layer':<8} {'Shape':<25} {'Norm':<15} {'Min':<15} {'Max':<15} {'Mean':<15}")
    print("-" * 93)
    
    for layer_idx in sorted(activation_stats.keys()):
        stats_list = activation_stats[layer_idx]
        if stats_list:
            # First entry is prefill
            stats = stats_list[0]
            print(f"{layer_idx:<8} {str(stats['shape']):<25} {stats['norm']:<15.4f} {stats['min']:<15.6f} {stats['max']:<15.6f} {stats['mean']:<15.6f}")
    
    # Summary
    print("\n[Summary]")
    norms = [activation_stats[idx][0]['norm'] for idx in sorted(activation_stats.keys()) if activation_stats[idx]]
    print(f"  Norm range: {min(norms):.4f} ~ {max(norms):.4f}")
    print(f"  Norm mean: {sum(norms)/len(norms):.4f}")
    
    # Check for abnormal layers
    mean_norm = sum(norms) / len(norms)
    abnormal_layers = [(idx, activation_stats[idx][0]['norm']) 
                       for idx in sorted(activation_stats.keys()) 
                       if activation_stats[idx] and activation_stats[idx][0]['norm'] > 3 * mean_norm]
    
    if abnormal_layers:
        print(f"\n[WARNING] Layers with abnormally high norms (> 3x mean):")
        for idx, norm in abnormal_layers:
            print(f"  Layer {idx}: norm = {norm:.4f}")
    else:
        print(f"\n[OK] All layers have normal activation norms")


if __name__ == "__main__":
    main()

