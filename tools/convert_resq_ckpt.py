import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import shutil

def convert_key(key, num_layers):
    # Mapping rules
    # 1. model.layers.* -> keep as is
    if key.startswith("model.layers.") or key.startswith("model.embed_tokens") or key.startswith("model.lm_head"):
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

    return []

def convert_checkpoint(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Copy non-safetensors files (config, etc.)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            src_path = os.path.join(input_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, os.path.join(output_dir, filename))
    
    # Read index if exists (for sharded models)
    # Assuming standard safetensors layout
    
    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    
    # We need to know num_layers to replicate Hd. 
    # Let's verify from config or infer from keys.
    # For now, let's infer max layer index from keys in the first pass or just assume Qwen3-32B (usualy 64 layers? or 40?)
    # Better to read config.json
    import json
    with open(os.path.join(input_dir, "config.json"), "r") as f:
        config = json.load(f)
        num_layers = config.get("num_hidden_layers", 32) # Default fallback
    
    print(f"Detected {num_layers} layers.")

    # Global map for Hd if it is in one file but needed in others? 
    # Safetensors usually splits by layers. Hd is likely in the first or last file.
    # If we replicate Hd, we write it to the file where it is found (or distribute?).
    # Simpler: Write it to the file where we found it, but with multiple keys? 
    # Yes, safetensors supports multiple keys.
    
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
            else:
                for nk in new_keys:
                    new_state_dict[nk] = tensor
        
        # Save to output
        if new_state_dict:
            save_file(new_state_dict, os.path.join(output_dir, st_file))
            
    print(f"Conversion complete. Output saved to {output_dir}")
    print("Please make sure to also copy/create quant_model_description.json in the output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output checkpoint directory")
    args = parser.parse_args()
    
    convert_checkpoint(args.input_path, args.output_path)
