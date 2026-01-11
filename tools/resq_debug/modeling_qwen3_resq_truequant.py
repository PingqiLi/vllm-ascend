"""
Qwen3 ResQ True Quantization Model (Debug Version)

真量化实现：forward 时做 int matmul，不在 load 时 dequantize。
用于验证量化计算的正确性。

支持:
- CPU: int32 matmul
- NPU: npu_quant_matmul 或 float16 matmul fallback

Reference: Quantization/ResQ-w4a4-dev/reference_op_impl.py

和 modeling_qwen3_resq.py 的区别：
- modeling_qwen3_resq.py: load 时 dequant 成 bf16，forward 做 bf16 matmul (伪量化)
- 本文件: load 时保持 int8，forward 做量化 matmul 再 dequant (真量化)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import Qwen3Config
from pathlib import Path
from safetensors import safe_open

# Import unified quantized matmul operations (supports CPU and NPU)
from .quant_ops import resq_quant_matmul


# ============================================================================
# RMSNorm
# ============================================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(input_dtype)


# ============================================================================
# Rotary Embedding
# ============================================================================

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_cache(seq_len)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ============================================================================
# Hadamard Transform
# ============================================================================

def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm."""
    n = u.shape[-1]
    original_shape = u.shape
    x = u.reshape(-1, n).clone()
    
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    
    return x.view(original_shape)


# ============================================================================
# ResQ Mixed-Precision Linear (True Quantization)
# ============================================================================

class ResQTrueQuantLinear(nn.Module):
    """
    真量化 Linear：存 int8 权重，forward 时做量化 matmul
    """
    def __init__(self, in_features: int, out_features: int, high_fraction: float = 0.125):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_fraction = high_fraction
        
        self.in_high = int(in_features * high_fraction)
        self.in_low = in_features - self.in_high
        
        # int8 weights
        self.register_buffer('weight_low', torch.empty(out_features, self.in_low, dtype=torch.int8))
        self.register_buffer('weight_high', torch.empty(out_features, self.in_high, dtype=torch.int8))
        # scales
        self.register_buffer('scale_low', torch.empty(out_features, dtype=torch.float32))
        self.register_buffer('scale_high', torch.empty(out_features, dtype=torch.float32))
        # activation scales (per-token, computed dynamically)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features).float()
        M = x_2d.shape[0]
        
        # Split input
        x_low = x_2d[:, :self.in_low]
        x_high = x_2d[:, self.in_low:]
        
        # Dynamic per-token quantization
        x_low_abs_max = x_low.abs().amax(dim=-1).clamp(min=1e-10)
        lxScale = (x_low_abs_max / 7.0).to(torch.float32)
        
        x_high_abs_max = x_high.abs().amax(dim=-1).clamp(min=1e-10)
        rxScale = (x_high_abs_max / 127.0).to(torch.float32)
        
        # Call quantized matmul (supports only NPU)
        output = resq_quant_matmul(
            x_2d, self.weight_low, self.weight_high,
            self.scale_low, self.scale_high,
            lxScale, rxScale,
            out_dtype=x.dtype
        )
        
        output_shape = list(original_shape[:-1]) + [self.out_features]
        return output.view(output_shape)


# ============================================================================
# Attention
# ============================================================================

class Qwen3ResQTrueQuantAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # True quantized projections
        self.q_proj = ResQTrueQuantLinear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = ResQTrueQuantLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = ResQTrueQuantLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = ResQTrueQuantLinear(self.num_heads * self.head_dim, self.hidden_size)
        
        # QK norms
        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)
        
        # ResQ: Uc rotation
        self.register_buffer('Uc', torch.empty(0))
        self.register_buffer('o_proj_column_order', torch.empty(0, dtype=torch.long))
    
    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # ResQ: Uc rotation after RoPE
        if self.Uc.numel() > 0:
            Uc = self.Uc.to(device=q.device, dtype=q.dtype)
            q = torch.matmul(q, Uc)
            k = torch.matmul(k, Uc)
        
        # GQA: expand KV
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        
        # ResQ: reorder columns before o_proj
        if self.o_proj_column_order.numel() > 0:
            attn_output = attn_output[..., self.o_proj_column_order]
        
        return self.o_proj(attn_output)


# ============================================================================
# MLP
# ============================================================================

class Qwen3ResQTrueQuantMLP(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = ResQTrueQuantLinear(self.hidden_size, self.intermediate_size)
        self.up_proj = ResQTrueQuantLinear(self.hidden_size, self.intermediate_size)
        self.down_proj = ResQTrueQuantLinear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.SiLU()
        
        # ResQ: Pd rotation
        self.register_buffer('Pd', torch.empty(0))
        self.Hd: Optional[torch.Tensor] = None
        self.K: int = 1
        self.blocksize: int = 256
    
    def _apply_ud_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Ud = block_diag(Pd).T @ H rotation before down_proj"""
        if self.Pd.numel() == 0 or self.Hd is None:
            return x
        
        original_shape = x.shape
        dtype = x.dtype
        device = x.device
        x = x.float()
        
        # Move rotation matrices to same device as x
        Pd = self.Pd.to(device).float()
        Hd = self.Hd.to(device).float()
        
        # 1. Apply block_diag(Pd).T
        blocksize = Pd.shape[0]
        num_blocks = x.shape[-1] // blocksize
        x_blocked = x.reshape(*original_shape[:-1], num_blocks, blocksize)
        x_blocked = torch.matmul(x_blocked, Pd.T)
        x = x_blocked.reshape(original_shape)
        
        # 2. Apply Hadamard: H = Hd ⊗ H_butterfly
        K = Hd.shape[0]
        x_blocked = x.reshape(*original_shape[:-1], K, self.blocksize)
        # Block Hadamard on Hd dimension
        x_blocked = torch.matmul(Hd, x_blocked)
        # Butterfly Hadamard on blocksize dimension
        x_blocked = hadamard_transform(x_blocked) / math.sqrt(self.blocksize)
        x = x_blocked.reshape(original_shape)
        
        return x.to(dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        intermediate = self.act_fn(gate) * up
        
        # ResQ: apply Ud rotation before down_proj
        if self.Pd.numel() > 0 and self.Hd is not None:
            intermediate = self._apply_ud_rotation(intermediate)
        
        return self.down_proj(intermediate)


# ============================================================================
# Decoder Layer
# ============================================================================

class Qwen3ResQTrueQuantDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ResQTrueQuantAttention(config, layer_idx)
        self.mlp = Qwen3ResQTrueQuantMLP(config, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# Full Model
# ============================================================================

class Qwen3ResQTrueQuantForCausalLM(nn.Module):
    """真量化 Qwen3 模型"""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3ResQTrueQuantDecoderLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = Qwen3RotaryEmbedding(head_dim, config.max_position_embeddings, config.rope_theta)
        
        # Global ResQ params
        self.Hd: Optional[torch.Tensor] = None
        self.K: int = 1
        self.blocksize: int = 256
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        hidden_states = self.embed_tokens(input_ids)
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask[:, None, None, :].float()) * float('-inf')
            causal_mask = causal_mask + padding_mask
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @classmethod
    def from_resq_checkpoint(
        cls,
        ckpt_a_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ResQTrueQuantForCausalLM":
        """从 ResQ checkpoint 加载真量化模型 (只需 CKPT_A)"""
        
        config = Qwen3Config.from_pretrained(ckpt_a_path)
        model = cls(config)
        
        ckpt_a = cls._load_safetensors_dir(ckpt_a_path)
        model._load_weights(ckpt_a, device, dtype)
        model = model.to(device).to(dtype)
        
        return model
    
    @staticmethod
    def _load_safetensors_dir(path: str) -> Dict:
        """Load all safetensors from a directory"""
        path = Path(path)
        result = {}
        
        if path.is_file() and path.suffix == '.pt':
            return torch.load(path, map_location='cpu')
        
        if path.is_dir():
            for f in path.glob("*.safetensors"):
                with safe_open(f, framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        result[key] = sf.get_tensor(key)
            return result
        
        raise ValueError(f"Unknown format: {path}")
    
    def _load_weights(self, ckpt_a: Dict, device: str, dtype: torch.dtype):
        """Load weights - keep quantized weights as int8 (只需 CKPT_A)"""
        
        # Global ResQ params
        if 'resq.Hd' in ckpt_a:
            self.Hd = ckpt_a['resq.Hd']
        if 'resq.Hd_K' in ckpt_a:
            self.K = int(ckpt_a['resq.Hd_K'].item())
        if 'resq.down_proj_blocksize' in ckpt_a:
            self.blocksize = int(ckpt_a['resq.down_proj_blocksize'].item())
        
        # Embed
        if 'model.embed_tokens.weight' in ckpt_a:
            self.embed_tokens.weight.data = ckpt_a['model.embed_tokens.weight'].to(dtype)
        
        # LM head
        for key in ['model.lm_head.weight', 'lm_head.weight']:
            if key in ckpt_a:
                self.lm_head.weight.data = ckpt_a[key].to(dtype)
                break
        
        # Final norm
        if 'model.norm.weight' in ckpt_a:
            self.norm.weight.data = ckpt_a['model.norm.weight'].to(dtype)
        
        # Layers
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}'
            
            # Norms
            for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                key = f'{prefix}.{norm_name}.weight'
                if key in ckpt_a:
                    getattr(layer, norm_name).weight.data = ckpt_a[key].to(dtype)
            
            # QK norms
            for qk in ['q_norm', 'k_norm']:
                key = f'{prefix}.self_attn.{qk}.weight'
                if key in ckpt_a:
                    getattr(layer.self_attn, qk).weight.data = ckpt_a[key].to(dtype)
            
            # Load quantized weights for projections
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                self._load_quant_linear(ckpt_a, f'{prefix}.self_attn.{proj}', 
                                        getattr(layer.self_attn, proj))
            
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                self._load_quant_linear(ckpt_a, f'{prefix}.mlp.{proj}', 
                                        getattr(layer.mlp, proj))
            
            # Uc rotation (在线应用于 Q/K)
            uc_key = f'resq.layer.{i}.Uc'
            if uc_key in ckpt_a:
                layer.self_attn.Uc = ckpt_a[uc_key].to(dtype)
            else:
                print(f"  [WARN] {uc_key} not found in CKPT_A")
            
            # Pd rotation (在线应用于 down_proj 输入)
            pd_key = f'resq.layer.{i}.Pd'
            if pd_key in ckpt_a:
                layer.mlp.Pd = ckpt_a[pd_key].to(dtype)
            else:
                print(f"  [WARN] {pd_key} not found in CKPT_A")
            
            # Set shared Hadamard params
            layer.mlp.Hd = self.Hd
            layer.mlp.K = self.K
            layer.mlp.blocksize = self.blocksize
            
            # o_proj column order
            self._setup_o_proj_column_order(layer.self_attn)
    
    def _load_quant_linear(self, ckpt: Dict, prefix: str, linear: ResQTrueQuantLinear):
        """Load quantized weights into ResQTrueQuantLinear"""
        
        low_key = f'{prefix}.weight_low'
        high_key = f'{prefix}.weight_high'
        scale_low_key = f'{prefix}.scale_low'
        scale_high_key = f'{prefix}.scale_high'
        
        if low_key in ckpt and high_key in ckpt:
            # Load int8 weights directly
            linear.weight_low.data = ckpt[low_key].to(torch.int8)
            linear.weight_high.data = ckpt[high_key].to(torch.int8)
            
            # Load scales
            if scale_low_key in ckpt:
                scale = ckpt[scale_low_key].float()
                if scale.dim() == 2:
                    scale = scale.squeeze()
                linear.scale_low.data = scale
            
            if scale_high_key in ckpt:
                scale = ckpt[scale_high_key].float()
                if scale.dim() == 2:
                    scale = scale.squeeze()
                linear.scale_high.data = scale
    
    def _setup_o_proj_column_order(self, attn):
        """Setup o_proj column reordering based on high_fraction"""
        high_fraction = 0.125
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        in_dim = num_heads * head_dim
        
        high_per_head = int(head_dim * high_fraction)
        low_per_head = head_dim - high_per_head
        
        low_indices = []
        high_indices = []
        for h in range(num_heads):
            base = h * head_dim
            low_indices.extend(range(base, base + low_per_head))
            high_indices.extend(range(base + low_per_head, base + head_dim))
        
        column_order = torch.tensor(low_indices + high_indices, dtype=torch.long)
        attn.o_proj_column_order = column_order
