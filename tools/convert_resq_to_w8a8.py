#!/usr/bin/env python3
"""
Convert msmodelslim ResQ weights to vllm-ascend W8A8 format.

ResQ Format (权重A):
- weight_low: [out, in_low] int8 (storing int4 values -8~7)
- weight_high: [out, in_high] int8
- scale_low, scale_high, offset_low, offset_high
- resq.Hd, resq.layer.*.Uc, resq.layer.*.Pd (rotation matrices)

W8A8 Format:
- weight: [out, in] int8
- input_scale: [1] per-tensor
- input_offset: [1] int8
- deq_scale: [out] float32
- quant_bias: [out] int32
- weight_scale: [out, 1]
- weight_offset: [out, 1]

Conversion Strategy:
1. Dequantize ResQ: bf16 = (int - offset) * scale
2. Re-quantize to W8A8: int8 = round(bf16 / new_scale)
3. Compute deq_scale = weight_scale * input_scale

Note: ResQ rotation matrices (Uc, Pd, Hd) are preserved for online rotation.
A custom model is needed to apply these rotations during inference.

Usage:
    python convert_resq_to_w8a8.py /path/to/resq_checkpoint /path/to/output_w8a8
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def dequantize_resq_weight(
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
    offset_low: Optional[torch.Tensor] = None,
    offset_high: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize ResQ mixed-precision weight to bf16."""
    # Low precision part
    w_low = weight_low.float()
    s_low = scale_low.float().view(-1, 1) if scale_low.dim() == 1 else scale_low.float()
    
    if offset_low is not None and offset_low.numel() > 0:
        o_low = offset_low.float().view(-1, 1) if offset_low.dim() == 1 else offset_low.float()
        dequant_low = (w_low - o_low) * s_low
    else:
        dequant_low = w_low * s_low
    
    # High precision part
    w_high = weight_high.float()
    s_high = scale_high.float().view(-1, 1) if scale_high.dim() == 1 else scale_high.float()
    
    if offset_high is not None and offset_high.numel() > 0:
        o_high = offset_high.float().view(-1, 1) if offset_high.dim() == 1 else offset_high.float()
        dequant_high = (w_high - o_high) * s_high
    else:
        dequant_high = w_high * s_high
    
    # Concat: low first, then high
    return torch.cat([dequant_low, dequant_high], dim=1).to(torch.bfloat16)


def quantize_to_w8a8(
    weight_bf16: torch.Tensor,
    input_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Quantize bf16 weight to W8A8 format.
    
    Returns dict with:
    - weight: [out, in] int8
    - input_scale: [1]
    - input_offset: [1] int8
    - deq_scale: [out] float32
    - quant_bias: [out] int32
    - weight_scale: [out, 1]
    - weight_offset: [out, 1]
    """
    out_features, in_features = weight_bf16.shape
    weight_f32 = weight_bf16.float()
    
    # Per-channel symmetric quantization
    # int8 range: -128 to 127
    channel_max = weight_f32.abs().max(dim=1, keepdim=True).values  # [out, 1]
    weight_scale = channel_max / 127.0  # Scale to fit in [-128, 127]
    weight_scale = weight_scale.clamp(min=1e-10)  # Avoid division by zero
    
    # Quantize to int8
    weight_scaled = weight_f32 / weight_scale
    weight_int8 = torch.round(weight_scaled).clamp(-128, 127).to(torch.int8)
    
    # Compute deq_scale for dequantization
    # output = (x_int8 @ weight_int8) * deq_scale
    # deq_scale = weight_scale * input_scale (per-channel)
    deq_scale = (weight_scale.squeeze(1) * input_scale).to(torch.float32)
    
    # quant_bias is typically 0 for symmetric quantization
    quant_bias = torch.zeros(out_features, dtype=torch.int32)
    
    result = {
        "weight": weight_int8,  # [out, in] int8
        "input_scale": torch.tensor([input_scale], dtype=torch.bfloat16),
        "input_offset": torch.zeros(1, dtype=torch.int8),
        "deq_scale": deq_scale,  # [out] float32
        "quant_bias": quant_bias,  # [out] int32
        "weight_scale": weight_scale.to(torch.bfloat16),  # [out, 1]
        "weight_offset": torch.zeros(out_features, 1, dtype=torch.bfloat16),
    }
    
    return result


def convert_checkpoint(
    input_path: str,
    output_path: str,
    input_scale: float = 1.0,
):
    """Convert ResQ checkpoint to W8A8 format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                        "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = input_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    # Update config.json with quantization info
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["quantization_config"] = {
            "quant_method": "ascend",
            "quant_type": "W8A8",
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    # Find safetensor files
    safetensor_files = list(input_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {input_path}")
    
    # Collect all tensors
    all_tensors: Dict[str, torch.Tensor] = {}
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    
    print(f"Loaded {len(all_tensors)} tensors from {len(safetensor_files)} files")
    
    # Identify ResQ linear layers
    resq_layers = set()
    for key in all_tensors:
        if ".weight_low" in key:
            # Extract layer prefix: model.layers.0.self_attn.q_proj
            layer_prefix = key.replace(".weight_low", "")
            resq_layers.add(layer_prefix)
    
    print(f"Found {len(resq_layers)} ResQ linear layers")
    
    # Convert each layer
    output_tensors: Dict[str, torch.Tensor] = {}
    converted = 0
    
    for layer_prefix in tqdm(sorted(resq_layers), desc="Converting layers"):
        # Get ResQ tensors
        weight_low = all_tensors.get(f"{layer_prefix}.weight_low")
        weight_high = all_tensors.get(f"{layer_prefix}.weight_high")
        scale_low = all_tensors.get(f"{layer_prefix}.scale_low")
        scale_high = all_tensors.get(f"{layer_prefix}.scale_high")
        offset_low = all_tensors.get(f"{layer_prefix}.offset_low")
        offset_high = all_tensors.get(f"{layer_prefix}.offset_high")
        
        if weight_low is None or weight_high is None:
            print(f"Warning: Missing weight_low or weight_high for {layer_prefix}")
            continue
        
        if scale_low is None:
            scale_low = torch.ones(weight_low.shape[0])
        if scale_high is None:
            scale_high = torch.ones(weight_high.shape[0])
        
        # Dequantize
        weight_bf16 = dequantize_resq_weight(
            weight_low, weight_high, 
            scale_low, scale_high,
            offset_low, offset_high
        )
        
        # Re-quantize to W8A8
        try:
            w8a8_tensors = quantize_to_w8a8(weight_bf16, input_scale)
        except Exception as e:
            print(f"Warning: Cannot convert {layer_prefix}: {e}")
            # Fallback: store as bf16
            output_tensors[f"{layer_prefix}.weight"] = weight_bf16
            continue
        
        # Add to output with correct names
        output_tensors[f"{layer_prefix}.weight"] = w8a8_tensors["weight"]
        output_tensors[f"{layer_prefix}.input_scale"] = w8a8_tensors["input_scale"]
        output_tensors[f"{layer_prefix}.input_offset"] = w8a8_tensors["input_offset"]
        output_tensors[f"{layer_prefix}.deq_scale"] = w8a8_tensors["deq_scale"]
        output_tensors[f"{layer_prefix}.quant_bias"] = w8a8_tensors["quant_bias"]
        output_tensors[f"{layer_prefix}.weight_scale"] = w8a8_tensors["weight_scale"]
        output_tensors[f"{layer_prefix}.weight_offset"] = w8a8_tensors["weight_offset"]
        
        converted += 1
    
    # Copy non-ResQ tensors (embeddings, norms, etc.)
    for key, tensor in all_tensors.items():
        # Skip ResQ-specific tensors
        if any(suffix in key for suffix in [".weight_low", ".weight_high", 
                                             ".scale_low", ".scale_high",
                                             ".offset_low", ".offset_high"]):
            continue
        
        # Skip if already converted
        if any(key.startswith(prefix) for prefix in resq_layers):
            continue
        
        # Keep ResQ rotation matrices
        if key.startswith("resq."):
            output_tensors[key] = tensor
            continue
        
        # Copy other tensors (embeddings, layernorms, lm_head)
        output_tensors[key] = tensor
    
    print(f"Converted {converted} layers, total {len(output_tensors)} output tensors")
    
    # Save output
    output_file = output_path / "model.safetensors"
    save_file(output_tensors, str(output_file))
    print(f"Saved to {output_file}")
    
    # Create quant_description.json
    quant_desc = {
        "quant_type": "W8A8",
        "converted_from": "ResQ",
        "has_rotation_matrices": True,
    }
    with open(output_path / "quant_description.json", "w") as f:
        json.dump(quant_desc, f, indent=2)
    
    print("Conversion complete!")
    print("\nNote: ResQ rotation matrices (Uc, Pd, Hd) are preserved.")
    print("You need a custom model to apply online rotations during inference.")


def main():
    parser = argparse.ArgumentParser(description="Convert ResQ to W8A8")
    parser.add_argument("input_path", help="Path to ResQ checkpoint")
    parser.add_argument("output_path", help="Path to output W8A8 checkpoint")
    parser.add_argument("--input-scale", type=float, default=1.0, 
                        help="Input quantization scale (default: 1.0)")
    
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path, args.input_scale)


if __name__ == "__main__":
    main()
