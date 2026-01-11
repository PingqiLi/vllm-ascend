"""
Qwen3 ResQ Model for Debug/Comparison

参考:
- vllm_ascend/models/qwen3_resq_truequant.py (真量化实现)
- project-resq/fake_quant/eval_utils/rotation_utils.py (原始算法)

用法:
    model = Qwen3ResQForCausalLM.from_resq_checkpoint(ckpt_a, ckpt_b)
    logits = model(input_ids)
"""

import math
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


# ============================================================================
# Hadamard Transform (from project-resq hadamard_utils.py)
# ============================================================================

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform using butterfly algorithm (unnormalized)."""
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    
    original_shape = x.shape
    u = x.reshape(-1, n).clone()
    
    h = 1
    while h < n:
        u = u.view(-1, n // (2 * h), 2, h)
        a, b = u[:, :, 0, :], u[:, :, 1, :]
        u = torch.stack([a + b, a - b], dim=2)
        u = u.view(-1, n)
        h *= 2
    
    return u.view(original_shape)


def matmul_hadU(x: torch.Tensor, hadK: torch.Tensor, K: int) -> torch.Tensor:
    """
    project-resq 的 Hadamard 变换: x @ (hadK ⊗ H_butterfly) / sqrt(n)
    
    Args:
        x: [..., n] where n = K * blocksize
        hadK: [K, K] 矩阵
        K: block 数量
    """
    n = x.shape[-1]
    blocksize = n // K
    original_shape = x.shape
    dtype = x.dtype
    
    # Reshape to [..., K, blocksize]
    x = x.float().reshape(-1, K, blocksize)
    
    # 1. Apply butterfly Hadamard to each block (blocksize dimension)
    x = hadamard_transform(x)
    
    # 2. Apply hadK across K dimension
    hadK_f32 = hadK.to(x.device, torch.float32)
    x = torch.einsum('ij,bjk->bik', hadK_f32, x)
    
    # 3. Normalize
    x = x / math.sqrt(n)
    
    return x.reshape(original_shape).to(dtype)


# ============================================================================
# Weight Dequantization (for CPU debug, true quant uses NPU kernels)
# ============================================================================

def dequantize_weight(
    weight_low: torch.Tensor, 
    weight_high: torch.Tensor,
    scale_low: torch.Tensor, 
    scale_high: torch.Tensor,
) -> torch.Tensor:
    """反量化混合精度权重 [low_part | high_part]
    
    weight_low/high: [out_features, partial_in_features] int8
    scale_low/high: [out_features, 1] or [out_features] float
    """
    # Ensure scales are broadcastable
    s_low = scale_low.float()
    s_high = scale_high.float()
    if s_low.dim() == 1:
        s_low = s_low.unsqueeze(-1)
    if s_high.dim() == 1:
        s_high = s_high.unsqueeze(-1)
    
    w_low = weight_low.float() * s_low
    w_high = weight_high.float() * s_high
    
    return torch.cat([w_low, w_high], dim=-1)


# ============================================================================
# ResQ Attention
# ============================================================================

class Qwen3ResQAttention(nn.Module):
    """
    ResQ Attention with Uc rotation after RoPE.
    
    参考 project-resq QKRotationWrapper:
        q = torch.matmul(q, self.k_rotation.to(q))
        k = torch.matmul(k, self.k_rotation.to(k))
    """
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK Norm
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # ResQ: Uc rotation matrix [head_dim, head_dim]
        self.register_buffer('Uc', torch.empty(0))
        
        # ResQ: o_proj column reorder (rearrange_o_proj in project-resq)
        self.register_buffer('o_proj_column_order', torch.empty(0, dtype=torch.long))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        
        # Q/K/V projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [bsz, num_heads, seq_len, head_dim]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # ResQ: Uc rotation after RoPE (project-resq QKRotationWrapper)
        if self.Uc.numel() > 0:
            q = torch.matmul(q, self.Uc.to(q.dtype))
            k = torch.matmul(k, self.Uc.to(k.dtype))

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :seq_len, :seq_len]
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        # ResQ: Reorder columns before o_proj (rearrange_o_proj in project-resq)
        if self.o_proj_column_order.numel() > 0:
            attn_output = attn_output[..., self.o_proj_column_order]
        
        return self.o_proj(attn_output)


# ============================================================================
# ResQ MLP
# ============================================================================

class Qwen3ResQMLP(nn.Module):
    """
    ResQ MLP with Ud rotation before down_proj.
    
    参考 project-resq:
    - 权重融合: W_new = matmul_hadU(W, K * inv(hadK).T, K)
    - 推理时激活: x' = matmul_hadU(x, hadK, K)
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
        # ResQ: Ud = Pd.T @ H where H = Hd ⊗ H_butterfly
        self.register_buffer('Pd', torch.empty(0))
        self.Hd: Optional[torch.Tensor] = None
        self.K: int = 1
        self.blocksize: int = 256
    
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = self.act_fn(gate) * up
        
        # ResQ: Apply Ud rotation before down_proj
        if self.Pd.numel() > 0 and self.Hd is not None:
            intermediate = self._apply_ud_rotation(intermediate, debug=debug)
        
        out = self.down_proj(intermediate)
        return out
    
    def _apply_ud_rotation(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Apply Ud = block_diag(Pd) @ H.T / sqrt(n)
        
        推导:
        - 权重融合: W_fused = Ua.T @ W @ block_diag(Pd) @ H.T / sqrt(n)
        - 推理: y = (x @ Ud) @ W_fused.T = x @ W.T @ Ua
        - 所以: Ud @ sqrt(n) * H @ block_diag(Pd.T) = I
        - 解得: Ud = block_diag(Pd) @ H.T / sqrt(n)
        
        操作顺序: 1. x @ block_diag(Pd)  2. x @ H.T  3. / sqrt(n)
        """
        original_shape = x.shape
        n = x.shape[-1]
        dtype = x.dtype
        
        # Reshape to [..., K, blocksize]
        x = x.float().reshape(-1, self.K, self.blocksize)
        
        # Step 1: x @ block_diag(Pd) (per-block multiplication)
        Pd_f32 = self.Pd.to(x.device, torch.float32)
        x = torch.matmul(x, Pd_f32)  # [batch, K, blocksize] @ [blocksize, blocksize]
        
        # Step 2: x @ H.T where H = Hd ⊗ H_butterfly
        # H.T = Hd.T ⊗ H_butterfly.T = Hd ⊗ H_butterfly (symmetric for real Hadamard)
        # For x @ (Hd ⊗ H_butterfly):
        #   2a. x @ H_butterfly on blocksize dim
        #   2b. x @ Hd on K dim
        
        # 2a. Apply H_butterfly (symmetric, so H = H.T)
        x = hadamard_transform(x)  # acts on last dim (blocksize)
        
        # 2b. Apply Hd on K dimension: x @ Hd
        # x shape: [batch, K, blocksize]
        # Transpose to [batch, blocksize, K], multiply by Hd, transpose back
        Hd_f32 = self.Hd.to(x.device, torch.float32)
        x = x.transpose(-1, -2)  # [batch, blocksize, K]
        x = torch.matmul(x, Hd_f32)  # [batch, blocksize, K] @ [K, K]
        x = x.transpose(-1, -2)  # [batch, K, blocksize]
        
        # Step 3: Normalize
        x = x / math.sqrt(n)
        
        return x.reshape(original_shape).to(dtype)


# ============================================================================
# Decoder Layer
# ============================================================================

class Qwen3ResQDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3ResQAttention(config, layer_idx)
        self.mlp = Qwen3ResQMLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.debug = False  # Set to True for layer 0 debugging
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        debug = self.debug and self.layer_idx == 0  # Only debug layer 0
        
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, debug=debug)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# Full Model
# ============================================================================

class Qwen3ResQForCausalLM(nn.Module):
    """
    Qwen3 ResQ for debug/comparison.
    
    在 CPU 上运行，权重反量化为 bf16，但前向逻辑与真量化版本一致。
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3ResQDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # RoPE
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        
        # ResQ global params
        self.Hd: Optional[torch.Tensor] = None
        self.K: int = 1
        self.blocksize: int = 256
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        
        # RoPE
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Combine with attention_mask if provided
        if attention_mask is not None:
            # attention_mask: [bsz, seq_len] -> [bsz, 1, 1, seq_len]
            # Note: 0 * -inf = nan in IEEE float, so use torch.where
            padding_mask = attention_mask[:, None, None, :].float()
            padding_mask = torch.where(padding_mask == 0, float('-inf'), 0.0)
            causal_mask = causal_mask + padding_mask
        
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
        
        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @classmethod
    def from_resq_checkpoint(
        cls,
        ckpt_a_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ResQForCausalLM":
        """
        从 ResQ checkpoint 加载模型
        
        Args:
            ckpt_a_path: CKPT_A 目录 (含 config.json、量化权重、Uc/Pd)
        """
        config = Qwen3Config.from_pretrained(ckpt_a_path)
        model = cls(config)
        ckpt_a = cls._load_checkpoint(ckpt_a_path)
        model._load_weights(ckpt_a, device, dtype)
        return model.to(device, dtype)
    
    @staticmethod
    def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
        """Load from .pt or safetensors directory"""
        from pathlib import Path
        path = Path(path)
        
        if path.suffix == '.pt':
            return torch.load(path, map_location='cpu')
        
        if path.is_dir():
            from safetensors import safe_open
            result = {}
            for sf in sorted(path.glob("*.safetensors")):
                with safe_open(sf, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        result[key] = f.get_tensor(key)
            return result
        
        raise ValueError(f"Unknown format: {path}")
    
    def _load_weights(self, ckpt_a: Dict, device: str, dtype: torch.dtype):
        """Load ResQ weights with dequantization (只需 CKPT_A)"""
        
        # Global ResQ params
        self.Hd = ckpt_a.get('resq.Hd')
        if 'resq.Hd_K' in ckpt_a:
            self.K = int(ckpt_a['resq.Hd_K'].item())
        if 'resq.down_proj_blocksize' in ckpt_a:
            self.blocksize = int(ckpt_a['resq.down_proj_blocksize'].item())
        
        # Embed
        if 'model.embed_tokens.weight' in ckpt_a:
            w = ckpt_a['model.embed_tokens.weight']
            print(f"[load] embed_tokens: {w.shape}, norm={w.float().norm():.2f}")
            self.embed_tokens.weight.data = w.to(dtype)
        else:
            print("[WARN] model.embed_tokens.weight not found!")
        
        # LM head
        if 'model.lm_head.weight' in ckpt_a:
            self.lm_head.weight.data = ckpt_a['model.lm_head.weight'].to(dtype)
            print(f"[load] lm_head: {self.lm_head.weight.shape}")
        elif 'lm_head.weight' in ckpt_a:
            self.lm_head.weight.data = ckpt_a['lm_head.weight'].to(dtype)
            print(f"[load] lm_head: {self.lm_head.weight.shape}")
        else:
            print("[WARN] lm_head.weight not found!")
        
        # Final norm
        if 'model.norm.weight' in ckpt_a:
            w = ckpt_a['model.norm.weight']
            print(f"[load] norm: {w.shape}, values={w[:5]}")
            self.norm.weight.data = w.to(dtype)
        else:
            print("[WARN] model.norm.weight not found!")
        
        # Layers
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}'
            
            # Norms
            if f'{prefix}.input_layernorm.weight' in ckpt_a:
                w = ckpt_a[f'{prefix}.input_layernorm.weight']
                if i == 0:
                    print(f"[load] L0 input_layernorm: {w.shape}, all_ones={torch.allclose(w.float(), torch.ones_like(w.float()))}, sample={w[:3]}")
                layer.input_layernorm.weight.data = w.to(dtype)
            if f'{prefix}.post_attention_layernorm.weight' in ckpt_a:
                layer.post_attention_layernorm.weight.data = ckpt_a[f'{prefix}.post_attention_layernorm.weight'].to(dtype)
            
            # QK Norm
            if f'{prefix}.self_attn.q_norm.weight' in ckpt_a:
                layer.self_attn.q_norm.weight.data = ckpt_a[f'{prefix}.self_attn.q_norm.weight'].to(dtype)
            if f'{prefix}.self_attn.k_norm.weight' in ckpt_a:
                layer.self_attn.k_norm.weight.data = ckpt_a[f'{prefix}.self_attn.k_norm.weight'].to(dtype)
            
            # Attention projections (dequantize)
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                w = self._dequantize(ckpt_a, f'{prefix}.self_attn.{proj}', dtype)
                if w is not None:
                    target = getattr(layer.self_attn, proj)
                    if w.shape != target.weight.shape:
                        print(f"  [ERROR] {prefix}.self_attn.{proj}: shape mismatch! got {w.shape}, expected {target.weight.shape}")
                    else:
                        target.weight.data = w
                else:
                    print(f"  [ERROR] {prefix}.self_attn.{proj}: weight is None, using random init!")
            
            # MLP projections (dequantize)
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                w = self._dequantize(ckpt_a, f'{prefix}.mlp.{proj}', dtype)
                if w is not None:
                    target = getattr(layer.mlp, proj)
                    if w.shape != target.weight.shape:
                        print(f"  [ERROR] {prefix}.mlp.{proj}: shape mismatch! got {w.shape}, expected {target.weight.shape}")
                    else:
                        target.weight.data = w
                else:
                    print(f"  [ERROR] {prefix}.mlp.{proj}: weight is None, using random init!")
            
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
            
            # o_proj column reorder
            self._setup_o_proj_column_order(layer.self_attn)
            
            # Debug is disabled by default for performance
            # To enable: model.layers[0].debug = True
    
    def _dequantize(self, ckpt: Dict, prefix: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Dequantize a linear layer's weight"""
        w_low = ckpt.get(f'{prefix}.weight_low')
        w_high = ckpt.get(f'{prefix}.weight_high')
        s_low = ckpt.get(f'{prefix}.scale_low')
        s_high = ckpt.get(f'{prefix}.scale_high')
        
        if all(x is not None for x in [w_low, w_high, s_low, s_high]):
            w = dequantize_weight(w_low, w_high, s_low, s_high).to(dtype)
            return w
        
        # Check what's missing
        missing = []
        if w_low is None: missing.append('weight_low')
        if w_high is None: missing.append('weight_high')
        if s_low is None: missing.append('scale_low')
        if s_high is None: missing.append('scale_high')
        
        # Non-quantized weight
        if f'{prefix}.weight' in ckpt:
            w = ckpt[f'{prefix}.weight'].to(dtype)
            print(f"  [direct] {prefix}: {w.shape}")
            return w
        
        print(f"  [WARN] {prefix}: missing {missing}, no weight loaded!")
        return None
    
    def _setup_o_proj_column_order(self, attn: Qwen3ResQAttention):
        """Setup o_proj column reorder (rearrange_o_proj in project-resq)"""
        head_dim = attn.head_dim
        num_heads = attn.num_heads
        in_dim = num_heads * head_dim # For Qwen3, 64 * 128 = 8192
        high_fraction = 0.125
        
        # msmodelslim uses in_dim * high_fraction for rearrange
        high_bits_length = int(in_dim * high_fraction)
        high_per_head = high_bits_length // num_heads
        low_per_head = head_dim - high_per_head
        
        # Build column order exactly matching msmodelslim:
        # chunk_starts = [0, 128, 256, ...]
        # high_precision_columns = [112, 113, ..., 127] (head内最后16维)
        # columns_to_end = 所有heads的high列
        # remaining_columns = 所有非high列
        # new_column_order = [remaining_columns | columns_to_end]
        
        chunk_starts = torch.arange(0, in_dim, head_dim)
        high_precision_columns = torch.arange(head_dim - high_per_head, head_dim)
        columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()
        
        all_columns = torch.arange(in_dim)
        mask = torch.ones(in_dim, dtype=torch.bool)
        mask[columns_to_end] = False
        remaining_columns = all_columns[mask]
        
        new_column_order = torch.cat([remaining_columns, columns_to_end])
        
        # Debug: Print first few indices
        if attn.layer_idx == 0:
            print(f"  [o_proj_column_order] in_dim={in_dim}, high_per_head={high_per_head}")
            print(f"  [o_proj_column_order] remaining[:5]={remaining_columns[:5].tolist()}, "
                  f"columns_to_end[:5]={columns_to_end[:5].tolist()}")
        
        attn.o_proj_column_order = new_column_order
