
import argparse
import os
import torch
import torch_npu
from vllm import LLM, SamplingParams
from vllm_ascend.models.qwen3_resq_truequant import Qwen3ResQForCausalLM

# Mock vLLM's internal registration to allow custom model type
from vllm.model_executor.models import ModelRegistry
ModelRegistry.register_model("Qwen3ResQForCausalLM", Qwen3ResQForCausalLM)

def main():
    parser = argparse.ArgumentParser(description="Test ResQ Client (Offline)")
    parser.add_argument("--model", type=str, required=True, help="Path to ResQ model")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens to generate")
    parser.add_argument("--chat", action="store_true", help="Apply chat template")
    args = parser.parse_args()

    # Clear old log file
    log_file_env = os.environ.get("RESQ_LOG_FILE")
    if log_file_env:
        root, ext = os.path.splitext(log_file_env)
        log_file = f"{root}_resqv1{ext}"
        if os.path.exists(log_file):
            print(f"Removing old log file: {log_file}")
            os.remove(log_file)

    print(f"Initializing vLLM with model: {args.model}")
    
    # Force quantization config to None (handled internally by truequant model)
    # But vLLM might complain if config.json says "resq" and we don't handle it.
    # The TrueQuant model assumes "Qwen3ResQForCausalLM" as architecture_class in config.json?
    # Or we force it.
    
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,
        tensor_parallel_size=1,
        # We don't pass quantization="resq" here because verify_resq_checkpoint.py style
        # loading is embedded in the model class itself.
        # However, vLLM factory needs to know to instantiate Qwen3ResQForCausalLM.
        # This usually requires modifying config.json or using --model-type if exposed.
        # Since we registered it above, let's see if vLLM picks it up if we override architectures?
        # Actually simplest is to ensure the model path's config.json has "architectures": ["Qwen3ResQForCausalLM"]
    )

    print(f"Promoting: {args.prompt}")
    
    prompt = args.prompt
    if args.chat:
        print("Applying chat template...")
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        # Ensure we have a chat template. If not, this might fail or do nothing.
        try:
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             print(f"Chat Prompt: {prompt!r}")
        except Exception as e:
             print(f"WARN: Failed to apply chat template: {e}")

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
