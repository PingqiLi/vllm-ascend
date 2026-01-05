#!/usr/bin/env python3
"""
Preprocess ResQ quantized weights to standard bf16 format.

This script takes a ResQ checkpoint with:
- weight_low (int4), weight_high (int8)
- scale_low, scale_high, offset_low, offset_high
- Uc, Ud rotation matrices

And produces a standard bf16 checkpoint with:
- weight (dequantized bf16)
- rotation_R3, rotation_R4 (renamed from Uc, Ud)

Usage:
    python preprocess_resq_weights.py --input /path/to/resq_ckpt --output /path/to/output_dir
"""

import argparse
import os
import json
import torch
from glob import glob
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def dequantize_weight(weight_low, weight_high, scale_low, scale_high, 
                      offset_low=None, offset_high=None):
    """
    Dequantize ResQ mixed-precision weights to bf16.
    
    Args:
        weight_low: Int4 weight [out_dim, in_low]
        weight_high: Int8 weight [out_dim, in_high]
        scale_low: Scale for int4 [out_dim, 1] or [out_dim]
        scale_high: Scale for int8 [out_dim, 1] or [out_dim]
        offset_low: Optional offset for int4 (asymmetric)
        offset_high: Optional offset for int8 (asymmetric)
    
    Returns:
        bf16 weight [out_dim, in_high + in_low] (int8 channels first)
    """
    # Reshape scales to [out_dim, 1] for broadcasting
    if scale_low.dim() == 1:
        scale_low = scale_low.view(-1, 1)
    if scale_high.dim() == 1:
        scale_high = scale_high.view(-1, 1)
    
    # Dequantize int8 part
    w_high = weight_high.float()
    if offset_high is not None and offset_high.numel() > 0:
        if offset_high.dim() == 1:
            offset_high = offset_high.view(-1, 1)
        # Ensure shapes match
        if offset_high.shape[0] != w_high.shape[0]:
            offset_high = offset_high[:w_high.shape[0]]
        if scale_high.shape[0] != w_high.shape[0]:
            scale_high = scale_high[:w_high.shape[0]]
        dequant_high = (w_high - offset_high.float()) * scale_high.float()
    else:
        if scale_high.shape[0] != w_high.shape[0]:
            scale_high = scale_high[:w_high.shape[0]]
        dequant_high = w_high * scale_high.float()
    
    # Dequantize int4 part
    w_low = weight_low.float()
    if offset_low is not None and offset_low.numel() > 0:
        if offset_low.dim() == 1:
            offset_low = offset_low.view(-1, 1)
        if offset_low.shape[0] != w_low.shape[0]:
            offset_low = offset_low[:w_low.shape[0]]
        if scale_low.shape[0] != w_low.shape[0]:
            scale_low = scale_low[:w_low.shape[0]]
        dequant_low = (w_low - offset_low.float()) * scale_low.float()
    else:
        if scale_low.shape[0] != w_low.shape[0]:
            scale_low = scale_low[:w_low.shape[0]]
        dequant_low = w_low * scale_low.float()
    
    # Concatenate: int8 (high precision) channels first, then int4
    full_weight = torch.cat([dequant_high, dequant_low], dim=1).to(torch.bfloat16)
    return full_weight


def process_checkpoint(input_dir, output_dir):
    """
    Process all safetensor files in input_dir and save dequantized weights to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all safetensor files
    safetensor_files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {input_dir}")
    
    print(f"Found {len(safetensor_files)} safetensor files")
    
    # Collect all weights
    all_weights = {}
    for sf_file in tqdm(safetensor_files, desc="Loading"):
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(all_weights)} tensors")
    
    # Group weights by layer/param
    layer_weights = {}  # layer_prefix -> {weight_low, weight_high, ...}
    other_weights = {}  # Non-quantized weights
    rotation_weights = {}  # Rotation matrices
    
    for key, tensor in all_weights.items():
        # Handle rotation matrices
        if key.endswith(".Uc"):
            # resq.layer.X.Uc -> model.layers.X.self_attn.rotation_R3
            new_key = key.replace("resq.layer.", "model.layers.")
            new_key = new_key.replace(".Uc", ".self_attn.rotation_R3")
            rotation_weights[new_key] = tensor
            continue
        elif key.endswith(".Ud"):
            # resq.layer.X.Ud -> model.layers.X.mlp.rotation_R4
            # Note: rotation_R4 is registered on Qwen3ResQMLP, not down_proj
            new_key = key.replace("resq.layer.", "model.layers.")
            new_key = new_key.replace(".Ud", ".mlp.rotation_R4")
            rotation_weights[new_key] = tensor
            continue
        
        # Handle quantized weights
        if ".weight_low" in key:
            prefix = key.replace(".weight_low", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["weight_low"] = tensor
        elif ".weight_high" in key:
            prefix = key.replace(".weight_high", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["weight_high"] = tensor
        elif ".scale_low" in key:
            prefix = key.replace(".scale_low", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["scale_low"] = tensor
        elif ".scale_high" in key:
            prefix = key.replace(".scale_high", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["scale_high"] = tensor
        elif ".offset_low" in key:
            prefix = key.replace(".offset_low", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["offset_low"] = tensor
        elif ".offset_high" in key:
            prefix = key.replace(".offset_high", "")
            if prefix not in layer_weights:
                layer_weights[prefix] = {}
            layer_weights[prefix]["offset_high"] = tensor
        else:
            # Standard weight (embed_tokens, lm_head, norm, etc.)
            other_weights[key] = tensor
    
    print(f"Found {len(layer_weights)} quantized layers, {len(rotation_weights)} rotation matrices, {len(other_weights)} other weights")
    
    # Dequantize and create output
    output_weights = {}
    
    # Add rotation matrices
    output_weights.update(rotation_weights)
    
    # Add non-quantized weights
    for key, tensor in other_weights.items():
        # Convert to bf16 if float16/float32
        if tensor.dtype in [torch.float16, torch.float32]:
            tensor = tensor.to(torch.bfloat16)
        output_weights[key] = tensor
    
    # Dequantize quantized layers
    for prefix, weights in tqdm(layer_weights.items(), desc="Dequantizing"):
        if "weight_low" not in weights or "weight_high" not in weights:
            print(f"Warning: Incomplete quantized layer {prefix}, skipping")
            continue
        if "scale_low" not in weights or "scale_high" not in weights:
            print(f"Warning: Missing scales for {prefix}, skipping")
            continue
        
        dequant = dequantize_weight(
            weights["weight_low"],
            weights["weight_high"],
            weights["scale_low"],
            weights["scale_high"],
            weights.get("offset_low"),
            weights.get("offset_high"),
        )
        
        # Output key: original prefix + ".weight"
        output_key = prefix + ".weight"
        output_weights[output_key] = dequant
        print(f"  {output_key}: {dequant.shape}")
    
    # Save output
    output_file = os.path.join(output_dir, "model.safetensors")
    print(f"\nSaving {len(output_weights)} tensors to {output_file}")
    save_file(output_weights, output_file)
    
    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                        "special_tokens_map.json", "quant_model_description.json"]:
        src = os.path.join(input_dir, config_file)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_dir, config_file))
            print(f"Copied {config_file}")
    
    print("\nDone! Preprocessed checkpoint saved to:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ResQ weights to bf16")
    parser.add_argument("--input", "-i", required=True, help="Input ResQ checkpoint directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for bf16 checkpoint")
    args = parser.parse_args()
    
    process_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
