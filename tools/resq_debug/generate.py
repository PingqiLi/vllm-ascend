#!/usr/bin/env python3
"""
Quick end-to-end generation test for ResQ model.

Usage:
    python -m tools.resq_debug.generate --ckpt-a /path/to/ckpt_a --prompt "你好" --max-tokens 32
"""

import argparse
import torch
from transformers import AutoTokenizer


def generate(
    ckpt_a: str,
    prompt: str = "你好，请介绍一下你自己。",
    max_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
):
    """Generate text using ResQ model"""
    from .modeling_qwen3_resq import Qwen3ResQForCausalLM
    
    print(f"Loading model from {ckpt_a}...")
    model = Qwen3ResQForCausalLM.from_resq_checkpoint(ckpt_a, device=device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_a, trust_remote_code=True)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")
    print("-" * 50)
    
    # Generate
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Forward
            logits = model(generated_ids)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"[EOS at step {step + 1}]")
                break
            
            # Print incremental
            new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(new_text, end="", flush=True)
    
    print("\n" + "-" * 50)
    
    # Full output
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nFull output:\n{output_text}")
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Quick E2E test for ResQ model")
    parser.add_argument("--ckpt-a", required=True, help="Path to CKPT_A (ResQ checkpoint)")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己。", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/npu)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    
    args = parser.parse_args()
    
    if args.greedy:
        args.temperature = 0.0
        args.top_p = 1.0
    
    # Handle greedy case
    if args.temperature == 0.0:
        args.temperature = 1.0  # Will use argmax below
    
    generate(
        ckpt_a=args.ckpt_a,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == "__main__":
    main()
