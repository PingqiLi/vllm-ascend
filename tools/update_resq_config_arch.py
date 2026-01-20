import json
import argparse
import os

def update_config(ckpt_path):
    config_path = os.path.join(ckpt_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    current_arch = config.get("architectures", ["Unknown"])[0]
    print(f"Current architecture: {current_arch}")
    
    target_arch = "Qwen3ResQTrueQuantForCausalLM"
    
    if current_arch == target_arch:
        print("Architecture represents ResQ TrueQuant already. No change needed.")
        return

    print(f"Updating architecture to: {target_arch}")
    config["architectures"] = [target_arch]
    
    # Also ensure auto_map is not interfering (though vLLM ignores it mostly, good for HF)
    # config["auto_map"] = {
    #     "AutoModelForCausalLM": "modeling_qwen3_resq_truequant.Qwen3ResQTrueQuantForCausalLM"
    # }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Successfully updated {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update config.json architectures for ResQ")
    parser.add_argument("ckpt_path", help="Path to the checkpoint directory")
    args = parser.parse_args()
    
    update_config(args.ckpt_path)
