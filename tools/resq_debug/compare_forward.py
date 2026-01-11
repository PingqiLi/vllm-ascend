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
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modeling_qwen3_resq import Qwen3ResQForCausalLM, Qwen3ResQConfig


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
        return f"{status} {self.name}: corr={self.corr:.4f}, rel_err={self.rel_err:.4f}, scale={self.scale:.4f}"


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


class ModelComparator:
    """Compare original and ResQ model outputs"""
    
    def __init__(
        self,
        orig_model: torch.nn.Module,
        resq_model: Qwen3ResQForCausalLM,
        tokenizer,
        device: str = "cpu",
    ):
        self.orig_model = orig_model.to(device).eval()
        self.resq_model = resq_model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Get Ua from ResQ checkpoint for space conversion
        self.Ua = self._extract_Ua()
    
    def _extract_Ua(self) -> Optional[torch.Tensor]:
        """Extract Ua matrix from ResQ model (for converting between spaces)"""
        # Ua is implicitly stored in the weight fusion
        # For now, we'll compare in the same space directly
        return None
    
    @torch.no_grad()
    def compare_logits(self, prompt: str) -> ComparisonResult:
        """Compare final logits"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Original model
        orig_out = self.orig_model(input_ids)
        orig_logits = orig_out.logits if hasattr(orig_out, 'logits') else orig_out
        
        # ResQ model
        resq_logits = self.resq_model(input_ids)
        
        return compute_comparison(orig_logits, resq_logits, "logits")
    
    @torch.no_grad()
    def compare_layers(
        self, 
        prompt: str, 
        layers: List[int],
        verbose: bool = True,
    ) -> Dict[int, Dict[str, ComparisonResult]]:
        """Compare intermediate activations for specified layers"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        results = {}
        
        # Run ResQ model with activation saving
        _ = self.resq_model(input_ids, save_activations=True, save_layers=layers)
        resq_acts = self.resq_model.activations
        
        # Run original model with hooks to capture activations
        orig_acts = self._capture_original_activations(input_ids, layers)
        
        # Compare
        for layer_idx in layers:
            layer_results = {}
            resq_layer = resq_acts.get(f'layer_{layer_idx}', {})
            orig_layer = orig_acts.get(f'layer_{layer_idx}', {})
            
            if verbose:
                print(f"\n--- Layer {layer_idx} ---")
            
            # Compare each activation
            for key in ['input', 'input_ln', 'post_attn', 'post_attn_ln', 'output']:
                if key in resq_layer and key in orig_layer:
                    result = compute_comparison(orig_layer[key], resq_layer[key], key)
                    layer_results[key] = result
                    if verbose:
                        print(f"  {result}")
            
            # Compare attention sub-activations
            resq_attn = self.resq_model.model.layers[layer_idx].self_attn.activations
            orig_attn = orig_acts.get(f'layer_{layer_idx}_attn', {})
            for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if key in resq_attn and key in orig_attn:
                    result = compute_comparison(orig_attn[key], resq_attn[key], f"attn.{key}")
                    layer_results[f'attn.{key}'] = result
                    if verbose:
                        print(f"  {result}")
            
            # Compare MLP sub-activations
            resq_mlp = self.resq_model.model.layers[layer_idx].mlp.activations
            orig_mlp = orig_acts.get(f'layer_{layer_idx}_mlp', {})
            for key in ['gate', 'up', 'mlp_hidden', 'down']:
                if key in resq_mlp and key in orig_mlp:
                    result = compute_comparison(orig_mlp[key], resq_mlp[key], f"mlp.{key}")
                    layer_results[f'mlp.{key}'] = result
                    if verbose:
                        print(f"  {result}")
            
            results[layer_idx] = layer_results
        
        # Compare logits
        if verbose:
            print("\n--- Logits ---")
        logits_result = compute_comparison(
            orig_acts.get('logits', torch.zeros(1)),
            resq_acts.get('logits', torch.zeros(1)),
            "logits"
        )
        results['logits'] = {'logits': logits_result}
        if verbose:
            print(f"  {logits_result}")
        
        return results
    
    def _capture_original_activations(
        self, 
        input_ids: torch.Tensor,
        layers: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Capture activations from original model using hooks"""
        activations = {}
        handles = []
        
        # Get model backbone
        model = self.orig_model.model if hasattr(self.orig_model, 'model') else self.orig_model
        
        # Hook for embeddings
        def embed_hook(module, input, output):
            activations['embed'] = output.clone()
        handles.append(model.embed_tokens.register_forward_hook(embed_hook))
        
        # Hooks for each layer
        for i in layers:
            if i >= len(model.layers):
                continue
            layer = model.layers[i]
            
            # Layer input/output
            def make_layer_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        activations[f'layer_{idx}'] = {
                            'output': output[0].clone()
                        }
                    else:
                        activations[f'layer_{idx}'] = {
                            'output': output.clone()
                        }
                return hook
            handles.append(layer.register_forward_hook(make_layer_hook(i)))
            
            # Input LayerNorm
            def make_ln_hook(idx, key):
                def hook(module, input, output):
                    if f'layer_{idx}' not in activations:
                        activations[f'layer_{idx}'] = {}
                    activations[f'layer_{idx}'][key] = output.clone()
                return hook
            handles.append(layer.input_layernorm.register_forward_hook(make_ln_hook(i, 'input_ln')))
            handles.append(layer.post_attention_layernorm.register_forward_hook(make_ln_hook(i, 'post_attn_ln')))
            
            # Attention projections
            attn = layer.self_attn
            def make_attn_hook(idx, proj_name):
                def hook(module, input, output):
                    if f'layer_{idx}_attn' not in activations:
                        activations[f'layer_{idx}_attn'] = {}
                    activations[f'layer_{idx}_attn'][proj_name] = output.clone()
                return hook
            handles.append(attn.q_proj.register_forward_hook(make_attn_hook(i, 'q_proj')))
            handles.append(attn.k_proj.register_forward_hook(make_attn_hook(i, 'k_proj')))
            handles.append(attn.v_proj.register_forward_hook(make_attn_hook(i, 'v_proj')))
            handles.append(attn.o_proj.register_forward_hook(make_attn_hook(i, 'o_proj')))
            
            # MLP projections
            mlp = layer.mlp
            def make_mlp_hook(idx, proj_name):
                def hook(module, input, output):
                    if f'layer_{idx}_mlp' not in activations:
                        activations[f'layer_{idx}_mlp'] = {}
                    activations[f'layer_{idx}_mlp'][proj_name] = output.clone()
                return hook
            handles.append(mlp.gate_proj.register_forward_hook(make_mlp_hook(i, 'gate')))
            handles.append(mlp.up_proj.register_forward_hook(make_mlp_hook(i, 'up')))
            handles.append(mlp.down_proj.register_forward_hook(make_mlp_hook(i, 'down')))
        
        # Final norm
        def final_norm_hook(module, input, output):
            activations['final_norm'] = output.clone()
        handles.append(model.norm.register_forward_hook(final_norm_hook))
        
        # LM head
        if hasattr(self.orig_model, 'lm_head'):
            def lm_head_hook(module, input, output):
                activations['logits'] = output.clone()
            handles.append(self.orig_model.lm_head.register_forward_hook(lm_head_hook))
        
        # Run forward
        with torch.no_grad():
            _ = self.orig_model(input_ids)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations


def compare_models(
    orig_model: torch.nn.Module,
    resq_model: Qwen3ResQForCausalLM,
    tokenizer,
    prompt: str,
    layers: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict:
    """Main comparison function"""
    comparator = ModelComparator(orig_model, resq_model, tokenizer, device)
    
    if layers is None:
        layers = [0]
    
    results = comparator.compare_layers(prompt, layers, verbose=True)
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
    )
    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=True)
    
    print(f"Loading ResQ model from checkpoints...")
    # Create config from original model
    orig_config = orig_model.config
    resq_config = Qwen3ResQConfig(
        hidden_size=orig_config.hidden_size,
        intermediate_size=orig_config.intermediate_size,
        num_attention_heads=orig_config.num_attention_heads,
        num_key_value_heads=orig_config.num_key_value_heads,
        num_hidden_layers=orig_config.num_hidden_layers,
        head_dim=getattr(orig_config, 'head_dim', orig_config.hidden_size // orig_config.num_attention_heads),
        vocab_size=orig_config.vocab_size,
        rms_norm_eps=orig_config.rms_norm_eps,
    )
    
    resq_model = Qwen3ResQForCausalLM.from_resq_checkpoint(
        args.ckpt_a, args.ckpt_b, resq_config, args.device
    )
    
    print(f"\nComparing with prompt: '{args.prompt}'")
    print(f"Layers to compare: {layers}")
    print("=" * 60)
    
    results = compare_models(
        orig_model, resq_model, tokenizer, args.prompt, layers, args.device
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_pass = 0
    total_fail = 0
    for layer_key, layer_results in results.items():
        for name, result in layer_results.items():
            if result.passed:
                total_pass += 1
            else:
                total_fail += 1
                print(f"  FAIL {layer_key}.{name}: rel_err={result.rel_err:.4f}, scale={result.scale:.4f}")
    
    print(f"\nTotal: {total_pass} passed, {total_fail} failed")


if __name__ == "__main__":
    main()
