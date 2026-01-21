import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Test ResQ Model with vLLM Offline Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the ResQ checkpoint")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens to generate (set to 1 to check prefill/first token only)")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max model length")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()

    print(f"Initializing vLLM with model: {args.model}")
    
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

    # Create sampling parameters
    # Temperature 0 for deterministic output (greedy decoding)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=args.max_tokens
    )

    prompts = [args.prompt]
    
    print(f"Generating for prompt: {prompts[0]}")
    outputs = llm.generate(prompts, sampling_params)

    print("=" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
    print("=" * 50)

if __name__ == "__main__":
    main()
