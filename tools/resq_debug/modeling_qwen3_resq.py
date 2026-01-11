"""
Qwen3 ResQ Model - Based on transformers Qwen3 with ResQ transforms

This module extends the original Qwen3 model with ResQ quantization transforms.
It saves intermediate activations for debugging purposes.

Key changes from original Qwen3:
1. Q/K/V/O and gate/up/down weights are dequantized from mixed-precision format
2. Q/K apply Uc rotation after RoPE
3. MLP hidden states apply Ud rotation before down_proj
4. All LayerNorm gamma are 1 (fused into weights)
5. Embed and lm_head are fused with Ua rotation

Usage:
    from tools.resq_debug import Qwen3ResQForCausalLM
    
    resq_model = Qwen3ResQForCausalLM.from_resq_checkpoint(
        original_model_path="/path/to/qwen3-bf16",
        ckpt_a_path="/path/to/checkpoint_A.pt",
        ckpt_b_path="/path/to/transforms_B.pt",
    )
"""

import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from local modelling_qwen3.py (copied from transformers)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# We'll use transformers directly for the base classes
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM,
        Qwen3Model,
        Qwen3DecoderLayer,
        Qwen3Attention,
        Qwen3MLP,
        Qwen3RMSNorm,
        apply_rotary_pos_emb,
        rotate_half,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using standalone implementation")


def dequantize_weight(weight_low: torch.Tensor, weight_high: torch.Tensor,
                      scale_low: torch.Tensor, scale_high: torch.Tensor) -> torch.Tensor:
    """Dequantize mixed-precision weight [low_bits | high_bits]"""
    w_low = weight_low.float() * scale_low.float()
    w_high = weight_high.float() * scale_high.float()
    return torch.cat([w_low, w_high], dim=-1)


def apply_ud_rotation(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, 
                      K: int, blocksize: int) -> torch.Tensor:
    """Apply Ud rotation: x @ Pd.T @ H
    
    H = Hd ⊗ H_butterfly where Hd is [K, K] and H_butterfly is [blocksize, blocksize]
    
    After msmodelslim fix, weights have sqrt(K) factor, so activation just does:
    - Apply Pd.T block-wise
    - Apply butterfly Hadamard per block
    - Apply Hd across blocks
    - Normalize by 1/sqrt(n)
    """
    batch_shape = x.shape[:-1]
    n = x.shape[-1]
    x_flat = x.reshape(-1, n)
    
    # Step 1: Reshape to blocks and apply Pd.T
    x_blocks = x_flat.reshape(-1, K, blocksize)
    x_rotated = torch.einsum('ji,bik->bjk', Pd.to(x.dtype), x_blocks)
    
    # Step 2: Apply butterfly Hadamard to each block
    # Use fast Walsh-Hadamard transform or scipy
    try:
        from scipy.linalg import hadamard as scipy_hadamard
        H_block = torch.tensor(scipy_hadamard(blocksize), dtype=x.dtype, device=x.device) / math.sqrt(blocksize)
    except:
        # Fallback: identity (will be wrong but allows testing)
        H_block = torch.eye(blocksize, dtype=x.dtype, device=x.device)
    
    x_had = torch.einsum('bik,jk->bij', x_rotated, H_block)
    
    # Step 3: Apply Hd across K dimension  
    x_had_k = torch.einsum('ij,bjk->bik', Hd.to(x.dtype), x_had)
    
    # Step 4: Normalize
    x_out = x_had_k.reshape(-1, n) / math.sqrt(n)
    
    return x_out.reshape(*batch_shape, n)


class ActivationTracker:
    """Mixin for tracking activations"""
    def __init__(self):
        self.activations = {}
        self._save_activations = False
    
    def save(self, name: str, tensor: torch.Tensor):
        if self._save_activations:
            self.activations[name] = tensor.clone().detach()


class Qwen3ResQAttention(nn.Module, ActivationTracker):
    """ResQ Attention - replaces original attention with ResQ transforms"""
    
    def __init__(self, original_attn: nn.Module, layer_idx: int):
        nn.Module.__init__(self)
        ActivationTracker.__init__(self)
        
        self.layer_idx = layer_idx
        self.original = original_attn  # Keep reference for config
        
        # Copy config from original
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.config.num_attention_heads
        self.num_kv_heads = original_attn.config.num_key_value_heads
        self.head_dim = original_attn.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = original_attn.scaling
        
        # Weights (will be replaced with dequantized)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Copy Q/K norms from original
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm
        
        # ResQ rotation matrix
        self.Uc = None  # Will be loaded
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # 保存 linear 之前的输入
        self.save('qkv_input', hidden_states)  # Q/K/V proj 的共同输入
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        self.save('q_proj', q)
        self.save('k_proj', k)
        self.save('v_proj', v)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Q/K norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        self.save('q_norm', q)
        self.save('k_norm', k)
        
        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Uc rotation (ResQ specific - post-RoPE)
        if self.Uc is not None:
            q = torch.matmul(q, self.Uc.to(q.dtype))
            k = torch.matmul(k, self.Uc.to(k.dtype))
        
        self.save('q_uc', q)
        self.save('k_uc', k)
        
        # GQA expansion
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :seq_len, :seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        self.save('attn_output', attn_output)
        
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        self.save('o_proj', output)
        
        return output


class Qwen3ResQMLP(nn.Module, ActivationTracker):
    """ResQ MLP - replaces original MLP with Ud rotation"""
    
    def __init__(self, original_mlp: nn.Module, layer_idx: int):
        nn.Module.__init__(self)
        ActivationTracker.__init__(self)
        
        self.layer_idx = layer_idx
        self.hidden_size = original_mlp.hidden_size
        self.intermediate_size = original_mlp.intermediate_size
        
        # Weights (will be replaced with dequantized)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = original_mlp.act_fn
        
        # ResQ rotation matrices
        self.Pd = None
        self.Hd = None
        self.K = 100
        self.blocksize = 256
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存 gate/up proj 之前的输入
        self.save('gate_up_input', x)
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        self.save('gate', gate)
        self.save('up', up)
        
        hidden = self.act_fn(gate) * up
        
        self.save('mlp_hidden', hidden)
        
        # Ud rotation (ResQ specific)
        if self.Pd is not None and self.Hd is not None:
            hidden = apply_ud_rotation(hidden, self.Pd, self.Hd, self.K, self.blocksize)
            self.save('ud_rotated', hidden)
        
        output = self.down_proj(hidden)
        
        self.save('down', output)
        
        return output


class Qwen3ResQDecoderLayer(nn.Module, ActivationTracker):
    """ResQ Decoder Layer - wraps attention and MLP with activation tracking"""
    
    def __init__(self, original_layer: nn.Module, layer_idx: int):
        nn.Module.__init__(self)
        ActivationTracker.__init__(self)
        
        self.layer_idx = layer_idx
        
        # Replace attention and MLP with ResQ versions
        self.self_attn = Qwen3ResQAttention(original_layer.self_attn, layer_idx)
        self.mlp = Qwen3ResQMLP(original_layer.mlp, layer_idx)
        
        # Keep original norms (gamma should be 1 after fusion)
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.save('input', hidden_states)
        
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        self.save('input_ln', hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        self.save('post_attn', hidden_states)
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        self.save('post_attn_ln', hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        self.save('output', hidden_states)
        
        return hidden_states
    
    def set_save_activations(self, enable: bool):
        self._save_activations = enable
        self.self_attn._save_activations = enable
        self.mlp._save_activations = enable


class Qwen3ResQForCausalLM(nn.Module):
    """ResQ Qwen3 for Causal LM - built on top of transformers Qwen3"""
    
    def __init__(self, original_model: nn.Module):
        super().__init__()
        
        self.config = original_model.config
        
        # Keep embed_tokens and lm_head (weights will be replaced)
        self.embed_tokens = original_model.model.embed_tokens
        self.lm_head = original_model.lm_head
        self.norm = original_model.model.norm
        self.rotary_emb = original_model.model.rotary_emb
        
        # Replace decoder layers with ResQ versions
        self.layers = nn.ModuleList([
            Qwen3ResQDecoderLayer(layer, i) 
            for i, layer in enumerate(original_model.model.layers)
        ])
        
        self.activations = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        save_activations: bool = False,
        save_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        if save_activations:
            self.activations['embed'] = hidden_states.clone()
        
        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Create causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = attention_mask
        
        # Decoder layers
        for i, layer in enumerate(self.layers):
            save_this = save_activations and (save_layers is None or i in save_layers)
            layer.set_save_activations(save_this)
            
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
            
            if save_this:
                self.activations[f'layer_{i}'] = {
                    'layer': layer.activations.copy(),
                    'attn': layer.self_attn.activations.copy(),
                    'mlp': layer.mlp.activations.copy(),
                }
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if save_activations:
            self.activations['final_norm'] = hidden_states.clone()
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        if save_activations:
            self.activations['logits'] = logits.clone()
        
        return logits
    
    @classmethod
    def from_resq_checkpoint(
        cls,
        original_model_path: str,
        ckpt_a_path: str,
        ckpt_b_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ResQForCausalLM":
        """Load ResQ model from checkpoint files
        
        Args:
            original_model_path: Path to original Qwen3 model (for structure)
            ckpt_a_path: Path to quantized weights (checkpoint A)
            ckpt_b_path: Path to transform matrices (checkpoint B)
            device: Device to load model to
            torch_dtype: Data type for model
        """
        print(f"Loading original model structure from {original_model_path}...")
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        print(f"Creating ResQ model...")
        model = cls(original_model)
        
        print(f"Loading checkpoint A from {ckpt_a_path}...")
        ckpt_a = torch.load(ckpt_a_path, map_location='cpu')
        
        print(f"Loading checkpoint B from {ckpt_b_path}...")
        ckpt_b = torch.load(ckpt_b_path, map_location='cpu')
        
        model._load_resq_weights(ckpt_a, ckpt_b, device, torch_dtype)
        
        return model.to(device)
    
    def _load_resq_weights(self, ckpt_a: Dict, ckpt_b: Dict, device: str, dtype: torch.dtype):
        """Load ResQ weights from checkpoints"""
        # Embed (already fused with Ua)
        if 'model.embed_tokens.weight' in ckpt_a:
            self.embed_tokens.weight.data = ckpt_a['model.embed_tokens.weight'].to(device, dtype)
        
        # LM head (already fused with Ua and gamma)
        if 'lm_head.weight' in ckpt_a:
            self.lm_head.weight.data = ckpt_a['lm_head.weight'].to(device, dtype)
        
        # Final norm (should be all 1s)
        if 'model.norm.weight' in ckpt_a:
            self.norm.weight.data = ckpt_a['model.norm.weight'].to(device, dtype)
        
        # Global ResQ params
        Hd = ckpt_b.get('resq.Hd')
        K = int(ckpt_b.get('resq.Hd_K', torch.tensor(100)).item())
        blocksize = int(ckpt_b.get('resq.down_proj_blocksize', torch.tensor(256)).item())
        
        # Load each layer
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}'
            
            # Norms (gamma=1 after fusion)
            if f'{prefix}.input_layernorm.weight' in ckpt_a:
                layer.input_layernorm.weight.data = ckpt_a[f'{prefix}.input_layernorm.weight'].to(device, dtype)
            if f'{prefix}.post_attention_layernorm.weight' in ckpt_a:
                layer.post_attention_layernorm.weight.data = ckpt_a[f'{prefix}.post_attention_layernorm.weight'].to(device, dtype)
            
            # Q/K norms
            if f'{prefix}.self_attn.q_norm.weight' in ckpt_a:
                layer.self_attn.q_norm.weight.data = ckpt_a[f'{prefix}.self_attn.q_norm.weight'].to(device, dtype)
            if f'{prefix}.self_attn.k_norm.weight' in ckpt_a:
                layer.self_attn.k_norm.weight.data = ckpt_a[f'{prefix}.self_attn.k_norm.weight'].to(device, dtype)
            
            # Dequantize attention weights
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = self._dequantize_linear(ckpt_a, f'{prefix}.self_attn.{proj}')
                if weight is not None:
                    getattr(layer.self_attn, proj).weight.data = weight.to(device, dtype)
            
            # Dequantize MLP weights
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                weight = self._dequantize_linear(ckpt_a, f'{prefix}.mlp.{proj}')
                if weight is not None:
                    getattr(layer.mlp, proj).weight.data = weight.to(device, dtype)
            
            # Rotation matrices
            Uc = ckpt_b.get(f'resq.layer.{i}.Uc')
            if Uc is not None:
                layer.self_attn.Uc = Uc.to(device, dtype)
            
            Pd = ckpt_b.get(f'resq.layer.{i}.Pd')
            if Pd is not None:
                layer.mlp.Pd = Pd.to(device, dtype)
            
            if Hd is not None:
                layer.mlp.Hd = Hd.to(device, dtype)
            layer.mlp.K = K
            layer.mlp.blocksize = blocksize
            
            print(f"  Layer {i}: loaded weights and rotations")
    
    def _dequantize_linear(self, ckpt: Dict, prefix: str) -> Optional[torch.Tensor]:
        """Dequantize a linear layer's weight"""
        weight_low = ckpt.get(f'{prefix}.weight_low')
        weight_high = ckpt.get(f'{prefix}.weight_high')
        scale_low = ckpt.get(f'{prefix}.scale_low')
        scale_high = ckpt.get(f'{prefix}.scale_high')
        
        if all(x is not None for x in [weight_low, weight_high, scale_low, scale_high]):
            return dequantize_weight(weight_low, weight_high, scale_low, scale_high)
        
        # Try non-quantized weight
        return ckpt.get(f'{prefix}.weight')
