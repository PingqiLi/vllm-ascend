#!/usr/bin/env python3
"""
Preprocess ResQ quantized weights to standard bf16 format.

This script takes a ResQ checkpoint with:
- weight_low (int4), weight_high (int8)
- scale_low, scale_high, offset_low, offset_high
- Uc, Pd rotation matrices (Hadamard mode)
- Global Hd, Hd_K parameters

And produces a standard bf16 checkpoint with:
- weight (dequantized bf16)
- model.layers.X.self_attn.rotation_R3 (renamed from resq.layer.X.Uc)
- model.layers.X.mlp.rotation_Pd (renamed from resq.layer.X.Pd)
- resq.Hd, resq.Hd_K (preserved for Hadamard transform)

Two modes are supported:
- dequant: Dequantize int4/int8 weights using scales and offsets, then concat
- concat: Directly concat fake-quantized float weights (no dequantization needed)

Usage:
    # For real int4/int8 quantized weights:
    python preprocess_resq_weights.py --input /path/to/resq_ckpt --output /path/to/output_dir --mode dequant
    
    # For fake-quantized float weights:
    python preprocess_resq_weights.py --input /path/to/resq_ckpt --output /path/to/output_dir --mode concat
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
        bf16 weight [out_dim, in_low + in_high] (int4 channels first, then int8)
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
    
    # Concatenate: preserve original order - int4 (low) first, then int8 (high)
    # This matches the msit quantization code: weight_low = weight[:, :low_dim], weight_high = weight[:, low_dim:]
    full_weight = torch.cat([dequant_low, dequant_high], dim=1).to(torch.bfloat16)
    return full_weight


def concat_weight(weight_low, weight_high):
    """
    Directly concatenate fake-quantized weights (already in float format).
    
    Use this mode when weights are already fake-quantized (pseudo-quantized)
    and stored as float tensors. No dequantization is needed.
    
    Args:
        weight_low: Fake-quantized low precision weight [out_dim, in_low] (float)
        weight_high: Fake-quantized high precision weight [out_dim, in_high] (float)
    
    Returns:
        bf16 weight [out_dim, in_low + in_high] (int4 channels first, then int8)
    """
    # Concatenate: preserve original order - int4 (low) first, then int8 (high)
    full_weight = torch.cat([weight_low.float(), weight_high.float()], dim=1).to(torch.bfloat16)
    return full_weight


def process_checkpoint(input_dir, output_dir, mode="dequant"):
    """
    Process all safetensor files in input_dir and save processed weights to output_dir.
    
    Args:
        input_dir: Input directory containing ResQ checkpoint files
        output_dir: Output directory for bf16 checkpoint
        mode: Processing mode
            - "dequant": Dequantize int4/int8 weights using scales/offsets, then concat
            - "concat": Directly concat fake-quantized float weights
    """
    print(f"Processing mode: {mode}")
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
        elif key.endswith(".Pd"):
            # Hadamard mode: resq.layer.X.Pd -> model.layers.X.mlp.rotation_Pd
            new_key = key.replace("resq.layer.", "model.layers.")
            new_key = new_key.replace(".Pd", ".mlp.rotation_Pd")
            rotation_weights[new_key] = tensor
            continue
        elif key.endswith(".Ud"):
            # Random mode Ud is not supported, skip
            print(f"  Skipping random mode parameter: {key}")
            continue
        elif key == "resq.Hd" or key == "resq.Hd_K" or key == "resq.intermediate_size" or key == "resq.down_proj_blocksize":
            # Global Hadamard parameters - pass through as-is
            rotation_weights[key] = tensor
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
    
    # Process quantized layers based on mode
    desc = "Dequantizing" if mode == "dequant" else "Concatenating"
    for prefix, weights in tqdm(layer_weights.items(), desc=desc):
        if "weight_low" not in weights or "weight_high" not in weights:
            print(f"Warning: Incomplete quantized layer {prefix}, skipping")
            continue
        
        if mode == "dequant":
            # Dequantize mode: need scales
            if "scale_low" not in weights or "scale_high" not in weights:
                print(f"Warning: Missing scales for {prefix}, skipping")
                continue
            
            processed = dequantize_weight(
                weights["weight_low"],
                weights["weight_high"],
                weights["scale_low"],
                weights["scale_high"],
                weights.get("offset_low"),
                weights.get("offset_high"),
            )
        else:
            # Concat mode: directly concatenate fake-quantized weights
            processed = concat_weight(
                weights["weight_low"],
                weights["weight_high"],
            )
        
        # Output key: original prefix + ".weight"
        output_key = prefix + ".weight"
        output_weights[output_key] = processed
        print(f"  {output_key}: {processed.shape}")
    
    # Filter out intermediate resq.* parameters that shouldn't be in final checkpoint
    # Keep: resq.Hd, resq.Hd_K (needed for Hadamard rotation at inference)
    # Remove: other resq.* params (e.g., resq.intermediate_size, resq.down_proj_blocksize - only used during preprocessing)
    keep_resq_keys = {"resq.Hd", "resq.Hd_K"}
    output_weights = {k: v for k, v in output_weights.items() 
                      if not k.startswith("resq.") or k in keep_resq_keys}

    # Save output
    output_file = os.path.join(output_dir, "model.safetensors")
    print(f"\nSaving {len(output_weights)} tensors to {output_file}")
    save_file(output_weights, output_file)
    
    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                        "special_tokens_map.json", "quant_model_description.json"]:
        src = os.path.join(input_dir, config_file)
        if os.path.exists(src):
            if config_file == "config.json":
                # Special handling: update architectures to use Qwen3ResQForCausalLM
                with open(src, 'r') as f:
                    config_data = json.load(f)
                config_data["architectures"] = ["Qwen3ResQForCausalLM"]
                with open(os.path.join(output_dir, config_file), 'w') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print(f"Updated {config_file} with architectures=['Qwen3ResQForCausalLM']")
            else:
                import shutil
                shutil.copy(src, os.path.join(output_dir, config_file))
                print(f"Copied {config_file}")
    
    print("\nDone! Preprocessed checkpoint saved to:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ResQ weights to bf16")
    parser.add_argument("--input", "-i", required=True, help="Input ResQ checkpoint directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for bf16 checkpoint")
    parser.add_argument("--mode", "-m", choices=["dequant", "concat"], default="dequant",
                        help="Processing mode: 'dequant' for int4/int8 weights with scales, "
                             "'concat' for fake-quantized float weights (default: dequant)")
    args = parser.parse_args()
    
    process_checkpoint(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
