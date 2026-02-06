#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""
Convert ResQ checkpoint to vLLM-compatible format.

This script converts ResQ checkpoint keys to match vLLM's expected format:
- resq.layer.{i}.Pd -> model.layers.{i}.mlp.down_proj.rotation_Pd
- resq.Hd -> model.layers.{i}.mlp.down_proj.rotation_Hd (replicated to all layers)
- resq.layer.{i}.Uc -> dropped (not needed for inference)

Usage:
    python convert_resq_ckpt.py --input_path /path/to/resq_ckpt --output_path /path/to/output
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def convert_key(key: str, num_layers: int, down_proj_is_resq: bool = True) -> list:
    """Convert ResQ checkpoint key to vLLM format.

    Args:
        key: The original checkpoint key
        num_layers: Number of layers in the model
        down_proj_is_resq: If True, down_proj uses RESQ and needs rotation matrices.
                          If False, down_proj uses W8A8_DYNAMIC and rotation matrices are dropped.
    """
    # Model weights - keep as is
    if key.startswith("model.layers."):
        return [key]
    if key.startswith("model.embed_tokens"):
        return [key]
    if key.startswith("model.lm_head") or key.startswith("lm_head"):
        return [key]
    if key == "model.norm.weight":
        return [key]

    # ResQ rotation matrices - only keep if down_proj uses RESQ
    if key.startswith("resq.layer.") and key.endswith(".Pd"):
        if down_proj_is_resq:
            parts = key.split(".")
            layer_idx = parts[2]
            return [f"model.layers.{layer_idx}.mlp.down_proj.rotation_Pd"]
        else:
            return []

    if key == "resq.Hd":
        if down_proj_is_resq:
            return [f"model.layers.{i}.mlp.down_proj.rotation_Hd" for i in range(num_layers)]
        else:
            return []

    # Uc matrices - not needed for inference
    if ".Uc" in key:
        return []

    print(f"WARNING: Dropping unknown key: {key}")
    return []


def get_down_proj_quant_type(quant_desc: dict) -> str:
    """Get the quantization type for down_proj layers from quant description.

    Returns 'RESQ' if down_proj uses RESQ, otherwise returns the actual type (e.g., 'W8A8_DYNAMIC').
    """
    # Check for down_proj weight type in quant description
    for key, value in quant_desc.items():
        if "down_proj.weight" in key and not key.endswith("_low") and not key.endswith("_high"):
            return value.upper()
        # Also check weight_low for RESQ format
        if "down_proj.weight_low" in key:
            return value.upper()
    return "RESQ"  # Default to RESQ if not found


def convert_quant_description(input_dir: str, output_dir: str, num_layers: int,
                              down_proj_is_resq: bool) -> None:
    """Convert quant_model_description.json to match new weight names."""
    quant_desc_filename = "quant_model_description.json"
    quant_desc_path = os.path.join(input_dir, quant_desc_filename)

    if not os.path.exists(quant_desc_path):
        print(f"WARNING: {quant_desc_filename} not found, skipping quant description conversion.")
        return

    with open(quant_desc_path, "r") as f:
        quant_desc = json.load(f)

    new_quant_desc = {}

    for key, value in quant_desc.items():
        # Keep model_quant_type
        if key == "model_quant_type":
            new_quant_desc[key] = value
            continue

        # Skip resq.* entries (Pd, Uc, Hd)
        if key.startswith("resq."):
            continue

        # Keep all other entries
        new_quant_desc[key] = value

    # Only add rotation_Pd and rotation_Hd entries if down_proj uses RESQ
    if down_proj_is_resq:
        for i in range(num_layers):
            new_quant_desc[f"model.layers.{i}.mlp.down_proj.rotation_Pd"] = "FLOAT"
            new_quant_desc[f"model.layers.{i}.mlp.down_proj.rotation_Hd"] = "FLOAT"
        print(f"Updated {quant_desc_filename} with rotation matrices entries.")
    else:
        print(f"down_proj uses W8A8_DYNAMIC, skipping rotation matrices entries.")

    with open(os.path.join(output_dir, quant_desc_filename), "w") as f:
        json.dump(new_quant_desc, f, indent=2)


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    """Convert ResQ checkpoint to vLLM format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_filename = "model.safetensors.index.json"
    quant_desc_filename = "quant_model_description.json"

    # Read quant description to determine down_proj quantization type
    quant_desc_path = os.path.join(input_dir, quant_desc_filename)
    down_proj_is_resq = True  # Default to RESQ
    if os.path.exists(quant_desc_path):
        with open(quant_desc_path, "r") as f:
            quant_desc = json.load(f)
        down_proj_type = get_down_proj_quant_type(quant_desc)
        down_proj_is_resq = (down_proj_type == "RESQ")
        print(f"down_proj quantization type: {down_proj_type}, is_resq: {down_proj_is_resq}")
    else:
        print(f"WARNING: {quant_desc_filename} not found, assuming down_proj uses RESQ.")

    for filename in os.listdir(input_dir):
        # Skip files that will be converted separately
        if filename.endswith(".safetensors") or filename in [index_filename, quant_desc_filename]:
            continue
        src_path = os.path.join(input_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy(src_path, os.path.join(output_dir, filename))

    index_path = os.path.join(input_dir, index_filename)
    has_index = os.path.exists(index_path)
    if has_index:
        with open(index_path, "r") as f:
            index_data = json.load(f)

    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]

    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        num_layers = config.get("num_hidden_layers", 32)

    print(f"Detected {num_layers} layers.")

    new_weight_map = {}

    for st_file in tqdm(safetensor_files, desc="Converting safetensors"):
        path = os.path.join(input_dir, st_file)
        state_dict = load_file(path)
        new_state_dict = {}

        for key, tensor in state_dict.items():
            new_keys = convert_key(key, num_layers, down_proj_is_resq)

            # Fix dtype mismatch: weight_offset is int8 in ckpt but
            # AscendW8A8DynamicLinearMethod expects bfloat16
            if key.endswith(".weight_offset") and tensor.dtype == torch.int8:
                tensor = tensor.to(torch.bfloat16)

            if len(new_keys) > 1:
                for nk in new_keys:
                    new_state_dict[nk] = tensor.clone()
                    new_weight_map[nk] = st_file
            elif len(new_keys) == 1:
                nk = new_keys[0]
                new_state_dict[nk] = tensor
                new_weight_map[nk] = st_file

        if new_state_dict:
            save_file(new_state_dict, os.path.join(output_dir, st_file))

    if has_index:
        index_data["weight_map"] = new_weight_map
        with open(os.path.join(output_dir, index_filename), "w") as f:
            json.dump(index_data, f, indent=2)
        print(f"Updated {index_filename} with {len(new_weight_map)} entries.")

    # Convert quant_model_description.json
    convert_quant_description(input_dir, output_dir, num_layers, down_proj_is_resq)

    print(f"Conversion complete. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ResQ checkpoint to vLLM-compatible format"
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    convert_checkpoint(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
