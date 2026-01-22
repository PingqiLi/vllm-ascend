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


def convert_key(key: str, num_layers: int) -> list:
    """Convert ResQ checkpoint key to vLLM format."""
    # Model weights - keep as is
    if key.startswith("model.layers."):
        return [key]
    if key.startswith("model.embed_tokens"):
        return [key]
    if key.startswith("model.lm_head") or key.startswith("lm_head"):
        return [key]
    if key == "model.norm.weight":
        return [key]

    # ResQ rotation matrices
    if key.startswith("resq.layer.") and key.endswith(".Pd"):
        parts = key.split(".")
        layer_idx = parts[2]
        return [f"model.layers.{layer_idx}.mlp.down_proj.rotation_Pd"]

    if key == "resq.Hd":
        return [f"model.layers.{i}.mlp.down_proj.rotation_Hd" for i in range(num_layers)]

    # Uc matrices - not needed for inference
    if ".Uc" in key:
        return []

    print(f"WARNING: Dropping unknown key: {key}")
    return []


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    """Convert ResQ checkpoint to vLLM format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_filename = "model.safetensors.index.json"
    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors") and filename != index_filename:
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
            new_keys = convert_key(key, num_layers)
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
