"""
Compare original Qwen3 and ResQ model forward passes

This script runs both models with the same input and compares intermediate
activations to identify where the discrepancy occurs.

Usage:
    python -m tools.resq_debug.compare_forward \
        --original /path/to/qwen3-bf16 \
        --ckpt-a /path/to/checkpoint_A.pt \
        --ckpt-b /path/to/transforms_B.pt \
        --prompt "Hello, world!" \
        --layers 0,1,63
"""

import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modeling_qwen3_resq import Qwen3ResQForCausalLM


@dataclass
class ComparisonResult:
    """Result of comparing two tensors"""
    name: str
    corr: float
    rel_err: float
    scale: float
    orig_norm: float
    resq_norm: float
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
    
    return ComparisonResult(
        name=name,
        corr=corr,
        rel_err=rel_err,
        scale=scale,
        orig_norm=orig_norm,
        resq_norm=resq_norm,
        passed=passed,
    )


class OriginalModelWrapper:
    """Wrapper to capture activations from original model using hooks"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
    
    @torch.no_grad()
    def forward_with_activations(
        self, 
        input_ids: torch.Tensor,
        layers: List[int],
    ) -> torch.Tensor:
        """Run forward and capture activations"""
        self.activations = {}
        handles = []
        
        backbone = self.model.model
        
        # Embed hook
        def embed_hook(m, inp, out):
            self.activations['embed'] = out.clone()
        handles.append(backbone.embed_tokens.register_forward_hook(embed_hook))
        
        # Layer hooks
        for i in layers:
            if i >= len(backbone.layers):
                continue
            layer = backbone.layers[i]
            
            def make_hooks(idx):
                hooks = []
                
                def input_ln_hook(m, inp, out):
                    if f'layer_{idx}' not in self.activations:
                        self.activations[f'layer_{idx}'] = {}
                    self.activations[f'layer_{idx}']['input_ln'] = out.clone()
                hooks.append(layer.input_layernorm.register_forward_hook(input_ln_hook))
                
                def post_attn_ln_hook(m, inp, out):
                    self.activations[f'layer_{idx}']['post_attn_ln'] = out.clone()
                hooks.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln_hook))
                
                # Attention
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(attn, proj_name)
                    def make_proj_hook(name):
                        def hook(m, inp, out):
                            if f'layer_{idx}_attn' not in self.activations:
                                self.activations[f'layer_{idx}_attn'] = {}
                            self.activations[f'layer_{idx}_attn'][name] = out.clone()
                        return hook
                    hooks.append(proj.register_forward_hook(make_proj_hook(proj_name)))
                
                # MLP
                mlp = layer.mlp
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(mlp, proj_name)
                    def make_mlp_hook(name):
                        def hook(m, inp, out):
                            if f'layer_{idx}_mlp' not in self.activations:
                                self.activations[f'layer_{idx}_mlp'] = {}
                            self.activations[f'layer_{idx}_mlp'][name] = out.clone()
                        return hook
                    hooks.append(proj.register_forward_hook(make_mlp_hook(proj_name)))
                
                # Layer output
                def layer_hook(m, inp, out):
                    if isinstance(out, tuple):
                        self.activations[f'layer_{idx}']['output'] = out[0].clone()
                    else:
                        self.activations[f'layer_{idx}']['output'] = out.clone()
                hooks.append(layer.register_forward_hook(layer_hook))
                
                return hooks
            
            handles.extend(make_hooks(i))
        
        # Final norm
        def final_norm_hook(m, inp, out):
            self.activations['final_norm'] = out.clone()
        handles.append(backbone.norm.register_forward_hook(final_norm_hook))
        
        # LM head
        def lm_head_hook(m, inp, out):
            self.activations['logits'] = out.clone()
        handles.append(self.model.lm_head.register_forward_hook(lm_head_hook))
        
        # Forward
        output = self.model(input_ids)
        
        # Cleanup
        for h in handles:
            h.remove()
        
        return output.logits if hasattr(output, 'logits') else output


def compare_models(
    orig_model,
    resq_model: Qwen3ResQForCausalLM,
    tokenizer,
    prompt: str,
    layers: List[int],
    device: str = "cpu",
) -> Dict:
    """Compare original and ResQ models"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    print(f"Input: '{prompt}' -> {input_ids.tolist()}")
    print("=" * 60)
    
    # Run original model
    orig_wrapper = OriginalModelWrapper(orig_model)
    with torch.no_grad():
        orig_logits = orig_wrapper.forward_with_activations(input_ids, layers)
    orig_acts = orig_wrapper.activations
    
    # Run ResQ model
    with torch.no_grad():
        resq_logits = resq_model(input_ids, save_activations=True, save_layers=layers)
    resq_acts = resq_model.activations
    
    results = {}
    
    # Compare embed
    if 'embed' in orig_acts and 'embed' in resq_acts:
        result = compute_comparison(orig_acts['embed'], resq_acts['embed'], 'embed')
        results['embed'] = result
        print(f"{result}")
    
    # Compare each layer
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")
        layer_results = {}
        
        orig_layer = orig_acts.get(f'layer_{layer_idx}', {})
        resq_layer_data = resq_acts.get(f'layer_{layer_idx}', {})
        resq_layer = resq_layer_data.get('layer', {})
        resq_attn = resq_layer_data.get('attn', {})
        resq_mlp = resq_layer_data.get('mlp', {})
        orig_attn = orig_acts.get(f'layer_{layer_idx}_attn', {})
        orig_mlp = orig_acts.get(f'layer_{layer_idx}_mlp', {})
        
        # Layer-level
        for key in ['input_ln', 'post_attn_ln', 'output']:
            if key in orig_layer and key in resq_layer:
                result = compute_comparison(orig_layer[key], resq_layer[key], key)
                layer_results[key] = result
                print(f"  {result}")
        
        # Attention
        for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if key in orig_attn and key in resq_attn:
                result = compute_comparison(orig_attn[key], resq_attn[key], f'attn.{key}')
                layer_results[f'attn.{key}'] = result
                print(f"  {result}")
        
        # MLP
        for key in ['gate', 'up', 'down']:
            orig_key = key if key != 'gate' else 'gate'
            mlp_key = key + '_proj' if key in ['gate', 'up'] else key
            if mlp_key in orig_mlp and key in resq_mlp:
                result = compute_comparison(orig_mlp[mlp_key], resq_mlp[key], f'mlp.{key}')
                layer_results[f'mlp.{key}'] = result
                print(f"  {result}")
        
        results[f'layer_{layer_idx}'] = layer_results
    
    # Compare logits
    print(f"\n--- Logits ---")
    if 'logits' in orig_acts and 'logits' in resq_acts:
        result = compute_comparison(orig_acts['logits'], resq_acts['logits'], 'logits')
        results['logits'] = result
        print(f"  {result}")
    
    # Top tokens comparison
    print(f"\n--- Top Predicted Tokens ---")
    orig_top = orig_logits[0, -1].topk(5)
    resq_top = resq_logits[0, -1].topk(5)
    
    print(f"  Original: {[tokenizer.decode([t]) for t in orig_top.indices.tolist()]}")
    print(f"  ResQ:     {[tokenizer.decode([t]) for t in resq_top.indices.tolist()]}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3 original and ResQ models")
    parser.add_argument("--original", required=True, help="Path to original Qwen3 model")
    parser.add_argument("--ckpt-a", required=True, help="Path to ResQ checkpoint A")
    parser.add_argument("--ckpt-b", required=True, help="Path to ResQ checkpoint B (transforms)")
    parser.add_argument("--prompt", default="Hello", help="Test prompt")
    parser.add_argument("--layers", default="0", help="Comma-separated layer indices to compare")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    layers = [int(x) for x in args.layers.split(",")]
    
    print(f"Loading original model from {args.original}...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.original, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=True)
    
    print(f"Loading ResQ model...")
    resq_model = Qwen3ResQForCausalLM.from_resq_checkpoint(
        args.original,
        args.ckpt_a,
        args.ckpt_b,
        args.device,
    ).eval()
    
    print(f"\n" + "=" * 60)
    print(f"Comparing with prompt: '{args.prompt}'")
    print(f"Layers: {layers}")
    print("=" * 60 + "\n")
    
    results = compare_models(
        orig_model, resq_model, tokenizer, args.prompt, layers, args.device
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_pass = 0
    total_fail = 0
    for key, val in results.items():
        if isinstance(val, ComparisonResult):
            if val.passed:
                total_pass += 1
            else:
                total_fail += 1
                print(f"  FAIL {key}: {val}")
        elif isinstance(val, dict):
            for name, result in val.items():
                if result.passed:
                    total_pass += 1
                else:
                    total_fail += 1
    
    print(f"\nTotal: {total_pass} passed, {total_fail} failed")


if __name__ == "__main__":
    main()
