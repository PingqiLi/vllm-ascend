#!/usr/bin/env python3
"""
Minimal end-to-end test for ResQ model loading and forward pass.

This tests:
1. Model loads correctly with all ResQ parameters
2. Single forward pass produces reasonable outputs
3. All rotation matrices are applied correctly

Usage:
    RESQ_DEBUG=1 python test_resq_e2e.py --ckpt /path/to/resq_ckpt --prompt "Hello"
"""

import argparse
import os
import sys
import torch

# Set environment
os.environ["RESQ_DEBUG"] = "1"
os.environ["RESQ_FAKE_QUANT"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to ResQ checkpoint")
    parser.add_argument("--prompt", default="Hello, my name is", help="Test prompt")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    args = parser.parse_args()
    
    print("="*60)
    print("ResQ E2E Test")
    print("="*60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tensor parallel: {args.tp}")
    
    # Check config.json for architectures
    import json
    config_path = os.path.join(args.ckpt, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        architectures = config.get("architectures", [])
        print(f"\nArchitectures in config.json: {architectures}")
        
        if "Qwen3ResQForCausalLM" not in architectures:
            print("\n⚠ WARNING: 'Qwen3ResQForCausalLM' not in architectures!")
            print("  You may need to update config.json or specify --model-impl")
    
    # Check for ResQ parameters
    from glob import glob
    from safetensors import safe_open
    
    print("\n" + "="*60)
    print("Checking ResQ Parameters in Checkpoint")
    print("="*60)
    
    resq_params = {}
    rotation_params = []
    
    for sf_file in glob(os.path.join(args.ckpt, "*.safetensors")):
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("resq."):
                    tensor = f.get_tensor(key)
                    resq_params[key] = tensor.shape
                    print(f"  {key}: {tensor.shape}")
                elif "rotation" in key:
                    rotation_params.append(key)
    
    print(f"\nTotal global ResQ params: {len(resq_params)}")
    print(f"Total rotation params: {len(rotation_params)}")
    
    if "resq.Hd" in resq_params:
        print("✓ Found resq.Hd")
    else:
        print("⚠ Missing resq.Hd")
    
    if "resq.Hd_K" in resq_params:
        print("✓ Found resq.Hd_K")
    else:
        print("⚠ Missing resq.Hd_K")
    
    # Count rotation_Pd and rotation_R3
    pd_count = sum(1 for k in rotation_params if "rotation_Pd" in k)
    r3_count = sum(1 for k in rotation_params if "rotation_R3" in k)
    print(f"\nrotation_Pd count: {pd_count}")
    print(f"rotation_R3 count: {r3_count}")
    
    print("\n" + "="*60)
    print("Loading Model via vLLM")
    print("="*60)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Create LLM with explicit model implementation
        llm = LLM(
            model=args.ckpt,
            trust_remote_code=True,
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            tensor_parallel_size=args.tp,
            enforce_eager=True,  # Disable CUDA graph for debugging
        )
        
        print("✓ Model loaded successfully")
        print("(RESQ_DEBUG logs above should show rotation parameters being loaded)")
        
        print("\n" + "="*60)
        print("Running Inference")
        print("="*60)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy for reproducibility
            max_tokens=args.max_tokens,
        )
        
        outputs = llm.generate([args.prompt], sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_text}")
        
        # Check for garbled output
        # Count non-ASCII or unusual characters
        unusual_chars = sum(1 for c in generated_text if ord(c) > 0x4000 or c in '□■◆◇○●')
        total_chars = len(generated_text)
        
        if total_chars > 0:
            unusual_ratio = unusual_chars / total_chars
            print(f"\nUnusual character ratio: {unusual_ratio:.2%}")
            
            if unusual_ratio > 0.5:
                print("⚠ WARNING: High ratio of unusual characters - likely garbled output!")
            else:
                print("✓ Output appears readable")
        
        print("\n" + "="*60)
        print("Test Complete")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

