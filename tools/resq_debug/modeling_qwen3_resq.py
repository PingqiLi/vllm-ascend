"""
Qwen3 ResQ Model - Transformers-style implementation for debugging

This module provides a clean, transformers-compatible ResQ model that can be
directly compared with the original Qwen3 model. It saves intermediate activations
for debugging purposes.

Usage:
    from tools.resq_debug import Qwen3ResQForCausalLM, compare_models
    
    # Load original model
    orig_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-XXB")
    
    # Load ResQ model from checkpoints
    resq_model = Qwen3ResQForCausalLM.from_resq_checkpoint(
        ckpt_a_path="/path/to/checkpoint_A.pt",
        ckpt_b_path="/path/to/transforms_B.pt",
        config=orig_model.config,
    )
    
    # Compare
    compare_models(orig_model, resq_model, tokenizer, "Hello")
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3ResQConfig:
    """Configuration for ResQ model"""
    hidden_size: int = 5120
    intermediate_size: int = 27648
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_hidden_layers: int = 64
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    
    # ResQ specific
    high_fraction: float = 0.125
    resq_K: int = 100
    resq_blocksize: int = 256


def dequantize_weight(weight_low: torch.Tensor, weight_high: torch.Tensor,
                      scale_low: torch.Tensor, scale_high: torch.Tensor) -> torch.Tensor:
    """Dequantize mixed-precision weight"""
    w_low = weight_low.float() * scale_low.float()
    w_high = weight_high.float() * scale_high.float()
    return torch.cat([w_low, w_high], dim=-1)


class RMSNorm(nn.Module):
    """RMSNorm with gamma parameter"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class Qwen3ResQRotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # [bs, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_ud_rotation(x: torch.Tensor, Pd: torch.Tensor, Hd: torch.Tensor, 
                      K: int, blocksize: int) -> torch.Tensor:
    """Apply Ud rotation: x @ Pd.T @ H
    
    The Hadamard transform H = Hd ⊗ H_butterfly where:
    - Hd is [K, K] block diagonal matrix
    - H_butterfly is blocksize-dim Hadamard
    
    For activation: just apply the transform (weights already have sqrt(K) factor)
    """
    batch_shape = x.shape[:-1]
    n = x.shape[-1]
    x_flat = x.reshape(-1, n)
    
    # Step 1: Reshape to blocks and apply Pd.T to each block
    x_blocks = x_flat.reshape(-1, K, blocksize)
    x_rotated = torch.matmul(Pd.T.to(x.dtype), x_blocks)  # [batch, K, blocksize]
    
    # Step 2: Apply butterfly Hadamard to each block
    from scipy.linalg import hadamard as scipy_hadamard
    import numpy as np
    H_block = torch.tensor(scipy_hadamard(blocksize), dtype=x.dtype, device=x.device) / math.sqrt(blocksize)
    x_had = torch.matmul(x_rotated, H_block.T)  # [batch, K, blocksize]
    
    # Step 3: Apply Hd across K dimension
    x_had_k = torch.matmul(Hd.to(x.dtype), x_had)  # [batch, K, blocksize]
    
    # Normalize
    x_out = x_had_k.reshape(-1, n) / math.sqrt(n)
    
    return x_out.reshape(*batch_shape, n)


class Qwen3ResQAttention(nn.Module):
    """ResQ Attention with rotation matrices"""
    
    def __init__(self, config: Qwen3ResQConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # Quantized weights (will be loaded)
        self.q_proj_weight = None  # Dequantized
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.o_proj_weight = None
        
        # Rotation matrices
        self.Uc = None  # [head_dim, head_dim] post-RoPE rotation
        
        # Q/K norms
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Activation hooks
        self.activations = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        save_activations: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q/K/V projections (already in rotated space, LN gamma fused)
        q = F.linear(hidden_states, self.q_proj_weight)
        k = F.linear(hidden_states, self.k_proj_weight)
        v = F.linear(hidden_states, self.v_proj_weight)
        
        if save_activations:
            self.activations['q_proj'] = q.clone()
            self.activations['k_proj'] = k.clone()
            self.activations['v_proj'] = v.clone()
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Q/K norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if save_activations:
            self.activations['q_norm'] = q.clone()
            self.activations['k_norm'] = k.clone()
        
        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Uc rotation (post-RoPE)
        if self.Uc is not None:
            q = torch.matmul(q, self.Uc.to(q.dtype))
            k = torch.matmul(k, self.Uc.to(k.dtype))
        
        if save_activations:
            self.activations['q_uc'] = q.clone()
            self.activations['k_uc'] = k.clone()
        
        # GQA: repeat k, v for num_kv_groups
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        if save_activations:
            self.activations['attn_output'] = attn_output.clone()
        
        # Reshape and O projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = F.linear(attn_output, self.o_proj_weight)
        
        if save_activations:
            self.activations['o_proj'] = output.clone()
        
        return output


class Qwen3ResQMLP(nn.Module):
    """ResQ MLP with Ud rotation"""
    
    def __init__(self, config: Qwen3ResQConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Quantized weights (will be loaded)
        self.gate_proj_weight = None  # Dequantized
        self.up_proj_weight = None
        self.down_proj_weight = None
        
        # Rotation matrices
        self.Pd = None  # [blocksize, blocksize]
        self.Hd = None  # [K, K] shared across layers
        self.K = config.resq_K
        self.blocksize = config.resq_blocksize
        
        # Activation hooks
        self.activations = {}
    
    def forward(self, x: torch.Tensor, save_activations: bool = False) -> torch.Tensor:
        gate = F.linear(x, self.gate_proj_weight)
        up = F.linear(x, self.up_proj_weight)
        
        if save_activations:
            self.activations['gate'] = gate.clone()
            self.activations['up'] = up.clone()
        
        hidden = F.silu(gate) * up
        
        if save_activations:
            self.activations['mlp_hidden'] = hidden.clone()
        
        # Apply Ud rotation before down_proj
        if self.Pd is not None and self.Hd is not None:
            hidden = apply_ud_rotation(hidden, self.Pd, self.Hd, self.K, self.blocksize)
            if save_activations:
                self.activations['ud_rotated'] = hidden.clone()
        
        output = F.linear(hidden, self.down_proj_weight)
        
        if save_activations:
            self.activations['down'] = output.clone()
        
        return output


class Qwen3ResQDecoderLayer(nn.Module):
    """ResQ Decoder Layer"""
    
    def __init__(self, config: Qwen3ResQConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3ResQAttention(config, layer_idx)
        self.mlp = Qwen3ResQMLP(config, layer_idx)
        
        # LayerNorms (gamma should be 1 after fusion)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        self.activations = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        save_activations: bool = False,
    ) -> torch.Tensor:
        if save_activations:
            self.activations['input'] = hidden_states.clone()
        
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if save_activations:
            self.activations['input_ln'] = hidden_states.clone()
        
        hidden_states = self.self_attn(
            hidden_states, position_ids, cos, sin, attention_mask, save_activations
        )
        hidden_states = residual + hidden_states
        
        if save_activations:
            self.activations['post_attn'] = hidden_states.clone()
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if save_activations:
            self.activations['post_attn_ln'] = hidden_states.clone()
        
        hidden_states = self.mlp(hidden_states, save_activations)
        hidden_states = residual + hidden_states
        
        if save_activations:
            self.activations['output'] = hidden_states.clone()
        
        return hidden_states


class Qwen3ResQModel(nn.Module):
    """ResQ Qwen3 Model (backbone)"""
    
    def __init__(self, config: Qwen3ResQConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3ResQDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        self.rotary_emb = Qwen3ResQRotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
        )
        
        self.activations = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        save_activations: bool = False,
        save_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        hidden_states = self.embed_tokens(input_ids)
        
        if save_activations:
            self.activations['embed'] = hidden_states.clone()
        
        # RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        # Layers
        for i, layer in enumerate(self.layers):
            save_this_layer = save_activations and (save_layers is None or i in save_layers)
            hidden_states = layer(
                hidden_states, position_ids, cos, sin, attention_mask, save_this_layer
            )
            if save_this_layer:
                self.activations[f'layer_{i}'] = layer.activations.copy()
        
        hidden_states = self.norm(hidden_states)
        
        if save_activations:
            self.activations['final_norm'] = hidden_states.clone()
        
        return hidden_states


class Qwen3ResQForCausalLM(nn.Module):
    """ResQ Qwen3 for Causal LM"""
    
    def __init__(self, config: Qwen3ResQConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3ResQModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.activations = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        save_activations: bool = False,
        save_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, position_ids, attention_mask, save_activations, save_layers
        )
        
        if save_activations:
            self.activations = self.model.activations.copy()
            self.activations['hidden_states'] = hidden_states.clone()
        
        logits = self.lm_head(hidden_states)
        
        if save_activations:
            self.activations['logits'] = logits.clone()
        
        return logits
    
    @classmethod
    def from_resq_checkpoint(
        cls,
        ckpt_a_path: str,
        ckpt_b_path: str,
        config: Optional[Qwen3ResQConfig] = None,
        device: str = "cpu",
    ) -> "Qwen3ResQForCausalLM":
        """Load ResQ model from checkpoint files
        
        Args:
            ckpt_a_path: Path to quantized weights (checkpoint A)
            ckpt_b_path: Path to transform matrices (checkpoint B)
            config: Model configuration (optional, will infer from weights)
            device: Device to load model to
        """
        print(f"Loading checkpoint A from {ckpt_a_path}...")
        ckpt_a = torch.load(ckpt_a_path, map_location='cpu')
        print(f"Loading checkpoint B from {ckpt_b_path}...")
        ckpt_b = torch.load(ckpt_b_path, map_location='cpu')
        
        # Infer config if not provided
        if config is None:
            # Try to infer from weights
            embed_weight = ckpt_a.get('model.embed_tokens.weight')
            if embed_weight is not None:
                vocab_size, hidden_size = embed_weight.shape
                # Count layers
                num_layers = len([k for k in ckpt_a.keys() if 'layers.' in k and '.q_proj.' in k]) // 4
                config = Qwen3ResQConfig(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    num_hidden_layers=num_layers,
                )
        
        model = cls(config)
        model._load_weights(ckpt_a, ckpt_b, device)
        return model.to(device)
    
    def _load_weights(self, ckpt_a: Dict, ckpt_b: Dict, device: str):
        """Load weights from checkpoints"""
        # Embed
        if 'model.embed_tokens.weight' in ckpt_a:
            self.model.embed_tokens.weight.data = ckpt_a['model.embed_tokens.weight'].to(device)
        
        # LM head
        if 'lm_head.weight' in ckpt_a:
            self.lm_head.weight.data = ckpt_a['lm_head.weight'].to(device)
        
        # Final norm
        if 'model.norm.weight' in ckpt_a:
            self.model.norm.weight.data = ckpt_a['model.norm.weight'].to(device)
        
        # Get Hd (shared)
        Hd = ckpt_b.get('resq.Hd')
        K = int(ckpt_b.get('resq.Hd_K', torch.tensor(100)).item())
        blocksize = int(ckpt_b.get('resq.down_proj_blocksize', torch.tensor(256)).item())
        
        # Load each layer
        for i, layer in enumerate(self.model.layers):
            prefix = f'model.layers.{i}'
            
            # Load norms
            if f'{prefix}.input_layernorm.weight' in ckpt_a:
                layer.input_layernorm.weight.data = ckpt_a[f'{prefix}.input_layernorm.weight'].to(device)
            if f'{prefix}.post_attention_layernorm.weight' in ckpt_a:
                layer.post_attention_layernorm.weight.data = ckpt_a[f'{prefix}.post_attention_layernorm.weight'].to(device)
            
            # Load Q/K norms
            if f'{prefix}.self_attn.q_norm.weight' in ckpt_a:
                layer.self_attn.q_norm.weight.data = ckpt_a[f'{prefix}.self_attn.q_norm.weight'].to(device)
            if f'{prefix}.self_attn.k_norm.weight' in ckpt_a:
                layer.self_attn.k_norm.weight.data = ckpt_a[f'{prefix}.self_attn.k_norm.weight'].to(device)
            
            # Dequantize and load linear weights
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = self._dequantize_linear(ckpt_a, f'{prefix}.self_attn.{proj}')
                if weight is not None:
                    setattr(layer.self_attn, f'{proj}_weight', weight.to(device))
            
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                weight = self._dequantize_linear(ckpt_a, f'{prefix}.mlp.{proj}')
                if weight is not None:
                    setattr(layer.mlp, f'{proj}_weight', weight.to(device))
            
            # Load rotation matrices
            Uc = ckpt_b.get(f'resq.layer.{i}.Uc')
            if Uc is not None:
                layer.self_attn.Uc = Uc.to(device)
            
            Pd = ckpt_b.get(f'resq.layer.{i}.Pd')
            if Pd is not None:
                layer.mlp.Pd = Pd.to(device)
            
            if Hd is not None:
                layer.mlp.Hd = Hd.to(device)
            layer.mlp.K = K
            layer.mlp.blocksize = blocksize
    
    def _dequantize_linear(self, ckpt: Dict, prefix: str) -> Optional[torch.Tensor]:
        """Dequantize a linear layer's weight"""
        weight_low = ckpt.get(f'{prefix}.weight_low')
        weight_high = ckpt.get(f'{prefix}.weight_high')
        scale_low = ckpt.get(f'{prefix}.scale_low')
        scale_high = ckpt.get(f'{prefix}.scale_high')
        
        if all(x is not None for x in [weight_low, weight_high, scale_low, scale_high]):
            return dequantize_weight(weight_low, weight_high, scale_low, scale_high)
        
        # Try non-quantized weight
        weight = ckpt.get(f'{prefix}.weight')
        return weight
