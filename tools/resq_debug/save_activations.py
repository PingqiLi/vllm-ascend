"""
Save activations from model forward pass

Stage 1: Run model and save activations to file
Stage 2: Compare activations from two saved files

Usage:
    # Stage 1a: Save original model activations (on CPU machine)
    python -m tools.resq_debug.save_activations \
        --mode original \
        --model /path/to/qwen3-bf16 \
        --output orig_acts.pt \
        --prompt "Hello, how are you?" \
        --device cpu
    
    # Stage 1b: Save ResQ model activations (on NPU machine)
    python -m tools.resq_debug.save_activations \
        --mode resq \
        --model /path/to/qwen3-bf16 \
        --ckpt-a checkpoint_A.pt \
        --ckpt-b transforms_B.pt \
        --output resq_acts.pt \
        --prompt "Hello, how are you?" \
        --device npu
    
    # Stage 2: Compare saved activations
    python -m tools.resq_debug.save_activations \
        --mode compare \
        --orig-file orig_acts.pt \
        --resq-file resq_acts.pt
"""

import argparse
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ComparisonResult:
    """Result of comparing two tensors"""
    name: str
    corr: float
    rel_err: float
    scale: float
    passed: bool
    
    def __str__(self):
        status = "✓" if self.passed else "✗"
        warn = " ⚠" if not self.passed else ""
        return f"{status} {self.name}: corr={self.corr:.4f}, rel_err={self.rel_err:.4f}{warn}, scale={self.scale:.4f}"


def compute_comparison(orig: torch.Tensor, resq: torch.Tensor, 
                       name: str, threshold: float = 0.1) -> ComparisonResult:
    """Compare two tensors"""
    orig_flat = orig.float().flatten()
    resq_flat = resq.float().flatten()
    
    orig_norm = orig_flat.norm().item()
    resq_norm = resq_flat.norm().item()
    
    # Correlation
    orig_centered = orig_flat - orig_flat.mean()
    resq_centered = resq_flat - resq_flat.mean()
    corr = F.cosine_similarity(orig_centered.unsqueeze(0), resq_centered.unsqueeze(0)).item()
    
    # Relative error
    diff_norm = (orig_flat - resq_flat).norm().item()
    rel_err = diff_norm / (orig_norm + 1e-8)
    
    # Scale ratio
    scale = resq_norm / (orig_norm + 1e-8)
    
    passed = rel_err < threshold and 0.9 < scale < 1.1
    
    return ComparisonResult(name=name, corr=corr, rel_err=rel_err, scale=scale, passed=passed)


class OriginalModelRunner:
    """Run original model and collect activations"""
    
    def __init__(self, model_path: str, device: str = "cpu", dtype=torch.bfloat16):
        print(f"Loading original model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
    
    @torch.no_grad()
    def run(self, prompt: str, layers: List[int], max_new_tokens: int = 20) -> Dict:
        """Run forward and collect activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        activations = {}
        handles = []
        
        backbone = self.model.model
        
        # Embed
        def embed_hook(m, inp, out):
            activations['embed'] = out.cpu().clone()
        handles.append(backbone.embed_tokens.register_forward_hook(embed_hook))
        
        # Layer hooks
        for i in layers:
            if i >= len(backbone.layers):
                continue
            layer = backbone.layers[i]
            
            def make_layer_hooks(idx, layer):
                hooks = []
                
                # Input LN
                def input_ln_hook(m, inp, out):
                    if f'layer_{idx}' not in activations:
                        activations[f'layer_{idx}'] = {}
                    activations[f'layer_{idx}']['input_ln'] = out.cpu().clone()
                hooks.append(layer.input_layernorm.register_forward_hook(input_ln_hook))
                
                # Post-attn LN
                def post_attn_ln_hook(m, inp, out):
                    activations[f'layer_{idx}']['post_attn_ln'] = out.cpu().clone()
                hooks.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln_hook))
                
                # Attention projections with input
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(attn, proj_name)
                    def make_hook(name, save_input=False):
                        def hook(m, inp, out):
                            key = f'layer_{idx}_attn'
                            if key not in activations:
                                activations[key] = {}
                            if save_input:
                                activations[key][f'{name}_input'] = inp[0].cpu().clone()
                            activations[key][name] = out.cpu().clone()
                        return hook
                    # Save input only for q_proj (shared with k/v)
                    hooks.append(proj.register_forward_hook(make_hook(proj_name, proj_name == 'q_proj')))
                
                # MLP projections with input
                mlp = layer.mlp
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(mlp, proj_name)
                    def make_hook(name, save_input=False):
                        def hook(m, inp, out):
                            key = f'layer_{idx}_mlp'
                            if key not in activations:
                                activations[key] = {}
                            if save_input:
                                activations[key][f'{name}_input'] = inp[0].cpu().clone()
                            activations[key][name] = out.cpu().clone()
                        return hook
                    hooks.append(proj.register_forward_hook(make_hook(proj_name, proj_name == 'gate_proj')))
                
                # Layer output
                def layer_output_hook(m, inp, out):
                    if isinstance(out, tuple):
                        activations[f'layer_{idx}']['output'] = out[0].cpu().clone()
                    else:
                        activations[f'layer_{idx}']['output'] = out.cpu().clone()
                hooks.append(layer.register_forward_hook(layer_output_hook))
                
                return hooks
            
            handles.extend(make_layer_hooks(i, layer))
        
        # Final norm
        def final_norm_hook(m, inp, out):
            activations['final_norm'] = out.cpu().clone()
        handles.append(backbone.norm.register_forward_hook(final_norm_hook))
        
        # LM head
        def lm_head_hook(m, inp, out):
            activations['logits'] = out.cpu().clone()
        handles.append(self.model.lm_head.register_forward_hook(lm_head_hook))
        
        # Forward
        output = self.model(input_ids)
        
        # Cleanup
        for h in handles:
            h.remove()
        
        # Generation
        generated_tokens = []
        current_ids = input_ids
        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(current_ids)
                logits = out.logits if hasattr(out, 'logits') else out
                next_token = logits[0, -1].argmax().item()
            generated_tokens.append(next_token)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
            if next_token == self.tokenizer.eos_token_id:
                break
        
        activations['generated_tokens'] = generated_tokens
        activations['generated_text'] = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        activations['prompt'] = prompt
        activations['layers'] = layers
        
        return activations


class ResQModelRunner:
    """Run ResQ model and collect activations"""
    
    def __init__(self, model_path: str, ckpt_a: str, ckpt_b: str, 
                 device: str = "npu", dtype=torch.bfloat16):
        from .modeling_qwen3_resq import Qwen3ResQForCausalLM
        
        print(f"Loading ResQ model...")
        self.model = Qwen3ResQForCausalLM.from_resq_checkpoint(
            model_path, ckpt_a, ckpt_b, device, dtype
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
    
    @torch.no_grad()
    def run(self, prompt: str, layers: List[int], max_new_tokens: int = 20) -> Dict:
        """Run forward and collect activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Forward with activation saving
        _ = self.model(input_ids, save_activations=True, save_layers=layers)
        
        # Move activations to CPU
        def to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            return obj
        
        activations = to_cpu(self.model.activations)
        
        # Generation
        generated_tokens = []
        current_ids = input_ids
        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(current_ids)
                next_token = logits[0, -1].argmax().item()
            generated_tokens.append(next_token)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
            if next_token == self.tokenizer.eos_token_id:
                break
        
        activations['generated_tokens'] = generated_tokens
        activations['generated_text'] = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        activations['prompt'] = prompt
        activations['layers'] = layers
        
        return activations


def compare_activations(orig_file: str, resq_file: str, summary_only: bool = False):
    """Compare saved activations from two files"""
    print(f"Loading original activations from {orig_file}...")
    orig = torch.load(orig_file, map_location='cpu')
    
    print(f"Loading ResQ activations from {resq_file}...")
    resq = torch.load(resq_file, map_location='cpu')
    
    # Auto-detect layers from saved data
    orig_layers = orig.get('layers', [])
    resq_layers = resq.get('layers', [])
    
    # Find common layers
    if orig_layers and resq_layers:
        layers = sorted(set(orig_layers) & set(resq_layers))
    else:
        # Fallback: detect from keys
        orig_layer_keys = [k for k in orig.keys() if k.startswith('layer_') and '_attn' not in k and '_mlp' not in k]
        resq_layer_keys = [k for k in resq.keys() if k.startswith('layer_') and '_attn' not in k and '_mlp' not in k]
        orig_layer_nums = sorted(set(int(k.split('_')[1]) for k in orig_layer_keys))
        resq_layer_nums = sorted(set(int(k.split('_')[1]) for k in resq_layer_keys))
        layers = sorted(set(orig_layer_nums) & set(resq_layer_nums))
    
    print(f"\nPrompt: '{orig.get('prompt', 'unknown')}'")
    print(f"Comparing {len(layers)} layers: {layers[0]}...{layers[-1]}" if len(layers) > 5 else f"Comparing layers: {layers}")
    print("=" * 60)
    
    results = []
    layer_results = {}  # For per-layer summary
    
    # Embed
    if 'embed' in orig and 'embed' in resq:
        result = compute_comparison(orig['embed'], resq['embed'], 'embed')
        results.append(result)
        if not summary_only:
            print(f"{result}")
    
    # Layers
    for layer_idx in layers:
        layer_passed = 0
        layer_failed = 0
        
        if not summary_only:
            print(f"\n--- Layer {layer_idx} ---")
        
        orig_layer = orig.get(f'layer_{layer_idx}', {})
        orig_attn = orig.get(f'layer_{layer_idx}_attn', {})
        orig_mlp = orig.get(f'layer_{layer_idx}_mlp', {})
        
        resq_layer_data = resq.get(f'layer_{layer_idx}', {})
        resq_layer = resq_layer_data.get('layer', {}) if isinstance(resq_layer_data, dict) else {}
        resq_attn = resq_layer_data.get('attn', {}) if isinstance(resq_layer_data, dict) else {}
        resq_mlp = resq_layer_data.get('mlp', {}) if isinstance(resq_layer_data, dict) else {}
        
        # Layer-level
        for key in ['input_ln', 'post_attn_ln', 'output']:
            if key in orig_layer and key in resq_layer:
                result = compute_comparison(orig_layer[key], resq_layer[key], key)
                results.append(result)
                if result.passed:
                    layer_passed += 1
                else:
                    layer_failed += 1
                if not summary_only:
                    print(f"  {result}")
        
        # Attention
        if 'q_proj_input' in orig_attn and 'qkv_input' in resq_attn:
            result = compute_comparison(orig_attn['q_proj_input'], resq_attn['qkv_input'], 'attn.qkv_input')
            results.append(result)
            if result.passed:
                layer_passed += 1
            else:
                layer_failed += 1
            if not summary_only:
                print(f"  {result}")
        
        for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if key in orig_attn and key in resq_attn:
                result = compute_comparison(orig_attn[key], resq_attn[key], f'attn.{key}')
                results.append(result)
                if result.passed:
                    layer_passed += 1
                else:
                    layer_failed += 1
                if not summary_only:
                    print(f"  {result}")
        
        # MLP
        if 'gate_proj_input' in orig_mlp and 'gate_up_input' in resq_mlp:
            result = compute_comparison(orig_mlp['gate_proj_input'], resq_mlp['gate_up_input'], 'mlp.gate_up_input')
            results.append(result)
            if result.passed:
                layer_passed += 1
            else:
                layer_failed += 1
            if not summary_only:
                print(f"  {result}")
        
        for orig_key, resq_key in [('gate_proj', 'gate'), ('up_proj', 'up'), ('down_proj', 'down')]:
            if orig_key in orig_mlp and resq_key in resq_mlp:
                result = compute_comparison(orig_mlp[orig_key], resq_mlp[resq_key], f'mlp.{resq_key}')
                results.append(result)
                if result.passed:
                    layer_passed += 1
                else:
                    layer_failed += 1
                if not summary_only:
                    print(f"  {result}")
        
        layer_results[layer_idx] = {'passed': layer_passed, 'failed': layer_failed}
    
    # Logits
    if 'logits' in orig and 'logits' in resq:
        if not summary_only:
            print(f"\n--- Logits ---")
        result = compute_comparison(orig['logits'], resq['logits'], 'logits')
        results.append(result)
        if not summary_only:
            print(f"  {result}")
    
    # Generation comparison
    print(f"\n--- Generation ---")
    orig_text = orig.get('generated_text', '')
    resq_text = resq.get('generated_text', '')
    orig_tokens = orig.get('generated_tokens', [])
    resq_tokens = resq.get('generated_tokens', [])
    
    print(f"  Original: '{orig_text}'")
    print(f"  ResQ:     '{resq_text}'")
    
    token_match_rate = 0
    if orig_tokens and resq_tokens:
        matches = sum(o == r for o, r in zip(orig_tokens, resq_tokens))
        total = max(len(orig_tokens), len(resq_tokens))
        token_match_rate = matches / total
        print(f"  Token match: {matches}/{total} ({100*token_match_rate:.1f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total_passed = sum(1 for r in results if r.passed)
    total_failed = sum(1 for r in results if not r.passed)
    print(f"Total checks: {total_passed} passed, {total_failed} failed")
    
    # Per-layer summary
    if summary_only and layer_results:
        print(f"\nPer-layer summary:")
        failed_layers = [idx for idx, lr in layer_results.items() if lr['failed'] > 0]
        passed_layers = [idx for idx, lr in layer_results.items() if lr['failed'] == 0]
        print(f"  Layers all passed: {len(passed_layers)}")
        if failed_layers:
            print(f"  Layers with failures: {failed_layers}")
    
    if total_failed > 0:
        print("\nFailed checks (first 20):")
        count = 0
        for r in results:
            if not r.passed:
                print(f"  {r}")
                count += 1
                if count >= 20:
                    remaining = total_failed - 20
                    if remaining > 0:
                        print(f"  ... and {remaining} more failures")
                    break
    
    # Final verdict
    print("\n" + "=" * 60)
    if total_failed == 0 and token_match_rate >= 0.9:
        print("✓ PASS: All activation checks passed, generation matches")
    elif total_failed == 0:
        print(f"⚠ PARTIAL: Activations OK but generation only {token_match_rate*100:.0f}% match")
    else:
        print(f"✗ FAIL: {total_failed} activation checks failed")


def main():
    parser = argparse.ArgumentParser(description="Save and compare model activations")
    parser.add_argument("--mode", required=True, choices=["original", "resq", "compare"],
                       help="Mode: original (save orig acts), resq (save resq acts), compare (compare saved)")
    
    # Model loading args
    parser.add_argument("--model", help="Path to model")
    parser.add_argument("--ckpt-a", help="ResQ checkpoint A path")
    parser.add_argument("--ckpt-b", help="ResQ checkpoint B path")
    
    # Run args
    parser.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--layers", default="all", help="Layers to check: 'all' or comma-separated indices like '0,15,31'")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--device", default="cpu", help="Device")
    
    # File args
    parser.add_argument("--output", help="Output file for saving activations")
    parser.add_argument("--orig-file", help="Original activations file (for compare mode)")
    parser.add_argument("--resq-file", help="ResQ activations file (for compare mode)")
    
    # Compare args
    parser.add_argument("--summary-only", action="store_true", help="Only show summary, hide per-layer details")
    
    args = parser.parse_args()
    
    # Parse layers - 'all' means None (will be resolved after model loading)
    if args.layers.lower() == 'all':
        layers = None  # Will be set to all layers after model loads
    else:
        layers = [int(x) for x in args.layers.split(",")]
    
    if args.mode == "original":
        if not args.model or not args.output:
            parser.error("--model and --output required for original mode")
        
        runner = OriginalModelRunner(args.model, args.device)
        
        # If layers is None (all), get all layer indices
        if layers is None:
            num_layers = len(runner.model.model.layers)
            layers = list(range(num_layers))
            print(f"Saving activations for all {num_layers} layers")
        
        activations = runner.run(args.prompt, layers, args.max_tokens)
        
        print(f"\nSaving activations to {args.output}...")
        torch.save(activations, args.output)
        print(f"Saved {len(layers)} layers")
        print(f"Generated text: '{activations['generated_text']}'")
        print("Done!")
        
    elif args.mode == "resq":
        if not args.model or not args.ckpt_a or not args.ckpt_b or not args.output:
            parser.error("--model, --ckpt-a, --ckpt-b, and --output required for resq mode")
        
        runner = ResQModelRunner(args.model, args.ckpt_a, args.ckpt_b, args.device)
        
        # If layers is None (all), get all layer indices
        if layers is None:
            num_layers = len(runner.model.layers)
            layers = list(range(num_layers))
            print(f"Saving activations for all {num_layers} layers")
        
        activations = runner.run(args.prompt, layers, args.max_tokens)
        
        print(f"\nSaving activations to {args.output}...")
        torch.save(activations, args.output)
        print(f"Saved {len(layers)} layers")
        print(f"Generated text: '{activations['generated_text']}'")
        print("Done!")
        
    elif args.mode == "compare":
        if not args.orig_file or not args.resq_file:
            parser.error("--orig-file and --resq-file required for compare mode")
        
        compare_activations(args.orig_file, args.resq_file, args.summary_only)


if __name__ == "__main__":
    main()
