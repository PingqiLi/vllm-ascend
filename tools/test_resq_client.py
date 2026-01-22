
import argparse
import sys
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Test ResQ Client (Offline V2)")
    parser.add_argument("--model", type=str, required=True, help="Path to ResQ model")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens to generate")
    parser.add_argument("--chat", action="store_true", help="Apply chat template")
    args = parser.parse_args()

    print(f"Initializing vLLM (V2) with model: {args.model}")
    
    # Standard vLLM loading for v2 (supports quantization="resq" via config)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    
    prompt = args.prompt
    if args.chat:
        print("Applying chat template...")
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Chat Prompt: {prompt!r}")

    print(f"Promoting: {prompt!r}")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        print(f"Generated text: {generated_text!r}")
        print(f"Generated Value (First Token ID): {token_ids[0] if token_ids else 'None'}")

if __name__ == "__main__":
    main()
