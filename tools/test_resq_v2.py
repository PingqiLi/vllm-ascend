import os
import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Test ResQ Model with vLLM Offline Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ResQ model")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max model length")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--chat", action="store_true", help="Apply chat template")
    
    args = parser.parse_args()

    # Clear old log file
    log_file_env = os.environ.get("RESQ_LOG_FILE")
    if log_file_env:
        root, ext = os.path.splitext(log_file_env)
        log_file = f"{root}_resqv2{ext}"
        if os.path.exists(log_file):
            print(f"Removing old log file: {log_file}")
            os.remove(log_file)

    print(f"Initializing vLLM (V2) with model: {args.model}")
    
    # Initialize the LLM engine
    # Note: We hardcode quantization="ascend" and enforce_eager=True as these are required for ResQ on Ascend
    llm = LLM(
        model=args.model,
        quantization="ascend",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
    )

    print(f"Promoting: {args.prompt}")
    
    prompt = args.prompt
    if args.chat:
        print("Applying chat template...")
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        try:
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             print(f"Chat Prompt: {prompt!r}")
        except Exception as e:
             print(f"WARN: Failed to apply chat template: {e}")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    
    outputs = llm.generate([prompt], sampling_params)
    
    print("=" * 50)
    for output in outputs:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        print(f"Generated text: {generated_text!r}")
        print(f"Generated Value (First Token ID): {token_ids[0] if token_ids else 'None'}")
    print("=" * 50)

if __name__ == "__main__":
    main()
