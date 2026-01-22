import argparse
import os
import json
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import shutil

def convert_key(key, num_layers):
    # Mapping rules
    # 1. model.layers.*, embed_tokens, lm_head -> keep as is
    # Note: lm_head can be either "lm_head" or "model.lm_head" depending on checkpoint format
    if key.startswith("model.layers.") or key.startswith("model.embed_tokens") or key.startswith("model.lm_head"):
        return [key]
    if key.startswith("lm_head"):
        return [key]
    
    # 2. resq.layer.{i}.Pd -> model.layers.{i}.mlp.down_proj.rotation_Pd
    if key.startswith("resq.layer.") and key.endswith(".Pd"):
        # Format: resq.layer.0.Pd
        parts = key.split(".")
        layer_idx = parts[2]
        return [f"model.layers.{layer_idx}.mlp.down_proj.rotation_Pd"]
    
    # 3. resq.Hd -> Replicate to all model.layers.{i}.mlp.down_proj.rotation_Hd
    if key == "resq.Hd":
        return [f"model.layers.{i}.mlp.down_proj.rotation_Hd" for i in range(num_layers)]
        
    # 4. resq.layer.{i}.Uc -> Ignore
    if ".Uc" in key:
        return []

    # 5. model.norm.weight -> properties of model
    if key == "model.norm.weight":
        return [key]

    # Unknown key - will be dropped
    print(f"WARNING: Dropping unknown key: {key}")
    return []

def convert_checkpoint(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy non-safetensors files (config, etc.) EXCEPT index file (we'll update it)
    index_filename = "model.safetensors.index.json"
    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors") and filename != index_filename:
            src_path = os.path.join(input_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, os.path.join(output_dir, filename))

    # Load index file if exists (for sharded models)
    index_path = os.path.join(input_dir, index_filename)
    weight_map = {}
    has_index = os.path.exists(index_path)
    if has_index:
        with open(index_path, "r") as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]

    # Read config for num_layers
    with open(os.path.join(input_dir, "config.json"), "r") as f:
        config = json.load(f)
        num_layers = config.get("num_hidden_layers", 32)

    print(f"Detected {num_layers} layers.")

    # Track new weight_map entries
    new_weight_map = {}

    for st_file in tqdm(safetensor_files, desc="Converting safetensors"):
        path = os.path.join(input_dir, st_file)
        state_dict = load_file(path)
        new_state_dict = {}

        for key, tensor in state_dict.items():
            new_keys = convert_key(key, num_layers)
            if len(new_keys) > 1:
                # Must clone to avoid safetensors shared memory error
                for nk in new_keys:
                    new_state_dict[nk] = tensor.clone()
                    new_weight_map[nk] = st_file
            elif len(new_keys) == 1:
                nk = new_keys[0]
                new_state_dict[nk] = tensor
                new_weight_map[nk] = st_file

        # Save to output
        if new_state_dict:
            save_file(new_state_dict, os.path.join(output_dir, st_file))

    # Update and save index file
    if has_index:
        index_data["weight_map"] = new_weight_map
        with open(os.path.join(output_dir, index_filename), "w") as f:
            json.dump(index_data, f, indent=2)
        print(f"Updated {index_filename} with {len(new_weight_map)} entries.")

    print(f"Conversion complete. Output saved to {output_dir}")
    print("Please make sure to also copy/create quant_model_description.json in the output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output checkpoint directory")
    args = parser.parse_args()
    
    convert_checkpoint(args.input_path, args.output_path)
