#!/usr/bin/env python3
"""
Run vllm-ascend online inference to capture multi-layer activations.

Initializes vLLM's LLM class with the converted ResQ checkpoint,
runs a prompt, and the diagnostic code in resq_linear.py / w8a8_dynamic.py
automatically saves activations for configured layers to RESQ_DIAG_DIR.

Usage:
    python -m tools.resq_debug.run_online \
        --model /path/to/converted_ckpt \
        --prompt "Hello, how are you?" \
        --max-tokens 32 \
        --diag-layers 0,31,63

    # Then compare (with MLP chain check):
    python -m tools.resq_debug.compare_online \
        --ckpt-a /path/to/converted_ckpt \
        --diag-dir /tmp/resq_online_acts \
        --check-chain
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run vllm-ascend inference to capture activations"
    )
    parser.add_argument("--model", required=True,
                        help="Path to converted ResQ checkpoint")
    parser.add_argument("--prompt", default="The quick brown fox jumps over",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max tokens to generate")
    parser.add_argument("--diag-dir", default="/tmp/resq_online_acts",
                        help="Directory to save activations")
    parser.add_argument("--diag-layers", default="0",
                        help="Layers to save activations for (comma-separated or 'all')")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model context length")
    args = parser.parse_args()

    # Set env vars before importing vllm (so they're visible to resq_linear/w8a8_dynamic)
    os.environ['RESQ_DIAG_DIR'] = args.diag_dir
    os.environ['RESQ_DIAG_LAYERS'] = args.diag_layers

    from vllm import LLM, SamplingParams

    print(f"Activations will be saved to: {args.diag_dir}")
    print(f"Diagnostic layers: {args.diag_layers}")
    print(f"Loading model: {args.model}")
    print(f"Prompt: {args.prompt!r}")

    llm = LLM(
        model=args.model,
        quantization="ascend",
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
    )

    # Save tokenized input ids for embedding chain verification
    import torch
    tokenizer = llm.get_tokenizer()
    input_ids = tokenizer.encode(args.prompt)
    torch.save({
        'prompt': args.prompt,
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
    }, os.path.join(args.diag_dir, 'input_ids.pt'))
    print(f"Saved input_ids ({len(input_ids)} tokens) to {args.diag_dir}/input_ids.pt")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        print(f"\n{'=' * 70}")
        print(f"Prompt: {output.prompt!r}")
        print(f"Output: {output.outputs[0].text!r}")
        print(f"{'=' * 70}")

    # List saved activation files
    if os.path.exists(args.diag_dir):
        files = sorted(os.listdir(args.diag_dir))
        if files:
            print(f"\nSaved {len(files)} activation files to {args.diag_dir}:")
            for f in files:
                print(f"  {f}")
        else:
            print(f"\nWarning: no activation files saved to {args.diag_dir}")
    else:
        print(f"\nWarning: {args.diag_dir} does not exist")


if __name__ == "__main__":
    main()
