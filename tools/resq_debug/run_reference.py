#!/usr/bin/env python3
"""
Save reference bf16 activations from CKPT_O for comparison with quantized CKPT_A.

Uses HuggingFace model with device_map="auto" to distribute across NPUs/GPUs,
avoiding OOM for large models.

Usage:
    python -m tools.resq_debug.run_reference \
        --model ${CKPT_O} \
        --prompt "The quick brown fox jumps over" \
        --diag-layers 63 \
        --diag-dir /tmp/resq_ref_acts

    # Then compare with quantized activations:
    python -m tools.resq_debug.compare_reference \
        --ref-dir /tmp/resq_ref_acts \
        --online-dir /tmp/resq_online_acts \
        --layers 63
"""

import argparse
import os
from typing import Set, Union

import torch


def parse_layers(layers_str: str, num_layers: int) -> list:
    """Parse layer specification into sorted list of indices."""
    if layers_str.strip().lower() == 'all':
        return list(range(num_layers))
    return sorted(int(x) for x in layers_str.split(',') if x.strip().isdigit())


def main():
    parser = argparse.ArgumentParser(
        description="Save reference bf16 activations from original model"
    )
    parser.add_argument("--model", required=True,
                        help="Path to CKPT_O (original bf16 model)")
    parser.add_argument("--prompt", default="The quick brown fox jumps over",
                        help="Input prompt (should match run_online.py)")
    parser.add_argument("--diag-layers", default="63",
                        help="Layers to save (comma-separated or 'all')")
    parser.add_argument("--diag-dir", default="/tmp/resq_ref_acts",
                        help="Output directory for activations")
    args = parser.parse_args()

    os.makedirs(args.diag_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model with device_map='auto': {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    layer_indices = parse_layers(args.diag_layers, num_layers)
    print(f"Model has {num_layers} layers, hooking layers: {layer_indices}")

    # Save input_ids (same format as run_online.py)
    input_ids = tokenizer.encode(args.prompt)
    torch.save({
        'prompt': args.prompt,
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
    }, os.path.join(args.diag_dir, 'input_ids.pt'))
    print(f"Saved input_ids ({len(input_ids)} tokens)")

    # Register forward hooks on each projection
    activations = {}
    hooks = []

    PROJ_MAP = {
        'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
    }

    def make_hook(key):
        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            activations[key] = {
                'prefix': key,
                'input': inp.detach().cpu(),
                'output': out.detach().cpu(),
            }
        return hook

    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for block_name, proj_names in PROJ_MAP.items():
            block = getattr(layer, block_name)
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is not None:
                    key = f"model.layers.{layer_idx}.{block_name}.{proj_name}"
                    h = proj.register_forward_hook(make_hook(key))
                    hooks.append(h)

    # Run forward pass
    inputs = tokenizer(args.prompt, return_tensors="pt")
    # With device_map="auto", just move to model's first device
    input_ids_tensor = inputs["input_ids"].to(model.device)

    print(f"Running forward pass with prompt: {args.prompt!r}")
    with torch.no_grad():
        outputs = model(input_ids_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save activations (one file per projection, same naming as run_online.py)
    for key, data in sorted(activations.items()):
        fname = key.replace('.', '_') + '.pt'
        torch.save(data, os.path.join(args.diag_dir, fname))
        print(f"  {key}: input={list(data['input'].shape)} output={list(data['output'].shape)}")

    # Print generated text for sanity check
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Next token: {tokenizer.decode(next_token)!r}")
    print(f"\nSaved {len(activations)} activation files to {args.diag_dir}")


if __name__ == "__main__":
    main()
