#!/usr/bin/env python3
"""
Preprocess ResQ quantized weights to standard bf16 format.

This script supports two modes:

Mode 1: 'dequant' (v1 - for real int4/int8 quantized weights)
    Input checkpoint has:
    - weight_low (int4), weight_high (int8) 
    - scale_low, scale_high, offset_low, offset_high
    - Uc, Ud rotation matrices
    
    Performs: dequant = (weight - offset) * scale, then concat

Mode 2: 'concat' (v2 - for fake-quantized float weights, DEFAULT)
    Input checkpoint has:
    - weight_low (bf16/fp16/fp32, already fake-quantized)
    - weight_high (bf16/fp16/fp32, already fake-quantized)
    - Uc, Ud rotation matrices
    
    Performs: just concat weight_low and weight_high (no scale/offset needed)

Output checkpoint has:
- weight (bf16, concatenated as [low, high] following project-resq order)
- rotation_R3, rotation_R4 (renamed from Uc, Ud)

Usage:
    # For fake-quantized checkpoints (default):
    python preprocess_resq_weights.py -i /path/to/resq_ckpt -o /path/to/output_dir
    
    # For real int4/int8 quantized checkpoints:
    python preprocess_resq_weights.py -i /path/to/resq_ckpt -o /path/to/output_dir --mode dequant
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
                      offset_low=None, offset_high=None, debug=False, prefix=""):
    """
    Dequantize ResQ mixed-precision weights to bf16.
    
    Following project-resq convention:
        W_l, W_m, W_h = W[:, :low_dim], W[:, low_dim:high_dim], W[:, high_dim:]
    The channel order is: low (int4) -> middle -> high (int8)
    
    Args:
        weight_low: Int4 weight [out_dim, in_low]
        weight_high: Int8 weight [out_dim, in_high]
        scale_low: Scale for int4 [out_dim, 1] or [out_dim]
        scale_high: Scale for int8 [out_dim, 1] or [out_dim]
        offset_low: Optional offset for int4 (asymmetric)
        offset_high: Optional offset for int8 (asymmetric)
        debug: If True, print debug info
        prefix: Prefix for debug output
    
    Returns:
        bf16 weight [out_dim, in_low + in_high] (int4 channels first, then int8)
    """
    if debug:
        print(f"\n[DEBUG] Dequantizing {prefix}")
        print(f"  weight_high: shape={weight_high.shape}, dtype={weight_high.dtype}, min={weight_high.min()}, max={weight_high.max()}")
        print(f"  weight_low: shape={weight_low.shape}, dtype={weight_low.dtype}, min={weight_low.min()}, max={weight_low.max()}")
        print(f"  scale_high: shape={scale_high.shape}, dtype={scale_high.dtype}, min={scale_high.min():.6f}, max={scale_high.max():.6f}")
        print(f"  scale_low: shape={scale_low.shape}, dtype={scale_low.dtype}, min={scale_low.min():.6f}, max={scale_low.max():.6f}")
        if offset_high is not None:
            print(f"  offset_high: shape={offset_high.shape}, min={offset_high.min()}, max={offset_high.max()}")
        if offset_low is not None:
            print(f"  offset_low: shape={offset_low.shape}, min={offset_low.min()}, max={offset_low.max()}")
    
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
    
    if debug:
        print(f"  dequant_low (int4): shape={dequant_low.shape}, min={dequant_low.min():.6f}, max={dequant_low.max():.6f}")
        print(f"  dequant_high (int8): shape={dequant_high.shape}, min={dequant_high.min():.6f}, max={dequant_high.max():.6f}")
    
    # Concatenate following project-resq order: low (int4) first, then high (int8)
    full_weight = torch.cat([dequant_low, dequant_high], dim=1).to(torch.bfloat16)
    return full_weight


def concat_fake_quant_weight(weight_low, weight_high, debug=False, prefix=""):
    """
    Concat fake-quantized ResQ weights to bf16.
    
    For fake-quantized checkpoints, weight_low and weight_high are already
    in float format (bf16/fp16/fp32) after fake quantization. No scale/offset
    dequantization is needed - just concat them.
    
    Following project-resq convention:
        W_l, W_m, W_h = W[:, :low_dim], W[:, low_dim:high_dim], W[:, high_dim:]
    The channel order is: low (int4) -> middle -> high (int8)
    
    Args:
        weight_low: Fake-quantized weight [out_dim, in_low] (already float, int4 precision)
        weight_high: Fake-quantized weight [out_dim, in_high] (already float, int8 precision)
        debug: If True, print debug info
        prefix: Prefix for debug output
    
    Returns:
        bf16 weight [out_dim, in_low + in_high] (low precision first, then high)
    """
    if debug:
        print(f"\n[DEBUG] Concatenating fake-quant weights: {prefix}")
        print(f"  weight_low (int4): shape={weight_low.shape}, dtype={weight_low.dtype}, "
              f"min={weight_low.float().min():.6f}, max={weight_low.float().max():.6f}")
        print(f"  weight_high (int8): shape={weight_high.shape}, dtype={weight_high.dtype}, "
              f"min={weight_high.float().min():.6f}, max={weight_high.float().max():.6f}")
    
    # Concatenate following project-resq order: low (int4) first, then high (int8)
    full_weight = torch.cat([weight_low.to(torch.bfloat16), 
                             weight_high.to(torch.bfloat16)], dim=1)
    
    if debug:
        print(f"  result: shape={full_weight.shape}, dtype={full_weight.dtype}, "
              f"min={full_weight.float().min():.6f}, max={full_weight.float().max():.6f}")
    
    return full_weight


def process_checkpoint(input_dir, output_dir, mode="dequant"):
    """
    Process all safetensor files in input_dir and save processed weights to output_dir.
    
    Args:
        input_dir: Input checkpoint directory
        output_dir: Output directory
        mode: Processing mode
            - "dequant": Dequantize int4/int8 weights using scale/offset (v1)
            - "concat": Just concat fake-quantized float weights (v2)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all safetensor files
    safetensor_files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {input_dir}")
    
    print(f"Found {len(safetensor_files)} safetensor files")
    print(f"Processing mode: {mode}")
    
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
    
    # Process and create output
    output_weights = {}
    
    # Add rotation matrices
    output_weights.update(rotation_weights)
    
    # Add non-quantized weights
    for key, tensor in other_weights.items():
        # Convert to bf16 if float16/float32
        if tensor.dtype in [torch.float16, torch.float32]:
            tensor = tensor.to(torch.bfloat16)
        output_weights[key] = tensor
    
    # Process quantized layers
    debug_count = 0
    desc = "Dequantizing" if mode == "dequant" else "Concatenating"
    
    for prefix, weights in tqdm(layer_weights.items(), desc=desc):
        if "weight_low" not in weights or "weight_high" not in weights:
            print(f"Warning: Incomplete quantized layer {prefix}, skipping")
            continue
        
        # Debug first 3 layers to understand the data
        should_debug = debug_count < 3
        debug_count += 1
        
        if mode == "dequant":
            # V1: Full dequantization with scale/offset
            if "scale_low" not in weights or "scale_high" not in weights:
                print(f"Warning: Missing scales for {prefix}, skipping")
                continue
            
            result = dequantize_weight(
                weights["weight_low"],
                weights["weight_high"],
                weights["scale_low"],
                weights["scale_high"],
                weights.get("offset_low"),
                weights.get("offset_high"),
                debug=should_debug,
                prefix=prefix,
            )
        else:
            # V2: Just concat fake-quantized weights (already float)
            result = concat_fake_quant_weight(
                weights["weight_low"],
                weights["weight_high"],
                debug=should_debug,
                prefix=prefix,
            )
        
        # Output key: original prefix + ".weight"
        output_key = prefix + ".weight"
        output_weights[output_key] = result
        
        # Print summary for all layers
        print(f"  {output_key}: shape={result.shape}, min={result.float().min():.6f}, max={result.float().max():.6f}")
    
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
    parser.add_argument("--mode", "-m", choices=["dequant", "concat"], default="concat",
                        help="Processing mode: "
                             "'dequant' = dequantize int4/int8 with scale/offset (v1), "
                             "'concat' = just concat fake-quantized float weights (v2, default)")
    args = parser.parse_args()
    
    process_checkpoint(args.input, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
