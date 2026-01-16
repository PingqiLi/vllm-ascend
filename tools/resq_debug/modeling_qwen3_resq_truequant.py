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
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
)
from pathlib import Path
from safetensors import safe_open

import torch_npu

# ============================================================================
# utils
# ============================================================================

def pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """
    x: int8 tensor, shape (E, K, N)，值域 ∈ [-8, 7]
    return: int8 tensor, shape (E, K, N/2)，每个元素打包两个有符号 int4
    """
    assert x.dtype == torch.int8
    K, N = x.shape
    assert N % 2 == 0
    
    # 转成无符号补码 [0, 15]
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]   # 偶数 -> 低 4 位
    high = x_unsigned[..., 1::2]  # 奇数 -> 高 4 位

    out = (low | (high << 4)).to(torch.int8)
    return out


# ============================================================================
# Rotary Embedding
# ============================================================================



# ============================================================================
# Hadamard Transform
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
        x_low = x_2d[:, :self.in_low].to(torch.float16).npu()
        x_high = x_2d[:, self.in_low:].to(torch.float16).npu()
        

        # NPU implement
        x_low_abs_max, lxScale = torch_npu.npu_dynamic_quant(x_low, dst_type=torch.quint4x2)
        x_high_abs_max, rxScale = torch_npu.npu_dynamic_quant(x_high, dst_type=torch.int8)
        
        weight_low = pack_int4_to_int8_signed(self.weight_low)
        weight_low = weight_low.view(torch.int32).transpose(-1, -2).npu()
        # weight_low = torch_npu.npu_format_cast(weight_low.npu(), 29).view(torch.int32)
        output_low = torch_npu.npu_quant_matmul(x_low_abs_max, weight_low, self.scale_low.to(torch.float).npu(), 
                                                pertoken_scale=lxScale, output_dtype=torch.float16)

        weight_nz_high = torch_npu.npu_format_cast(self.weight_high.npu(), 29).transpose(-1,-2)
        output_high = torch_npu.npu_quant_matmul(x_high_abs_max, weight_nz_high, self.scale_high.to(torch.float).npu(), 
                                                pertoken_scale=rxScale, output_dtype=torch.float16)
        import numpy as np

        def CPU_MM_golden(x_low_abs_max, weight_low, scale_low, lxScale, islow):
            def unpack_int32_to_int4_signed(x: torch.Tensor) -> torch.Tensor:
                """
                x: int32 tensor, shape (K, N/8)
                return: int8 tensor, shape ( K, N)，存储 signed int4 ∈ [-8, 7]
                """
                assert x.dtype == torch.int32
                K, N_ = x.shape  # N_ = N/8

                # 取出 8 个 4bit
                out = torch.stack([(x >> (4 * i)) & 0xF for i in range(8)], dim=-1)  # (E,K,M,8)

                # 转成有符号 int4
                out = out.to(torch.int8)
                out = torch.where(out >= 8, out - 16, out)  # [-8,7]

                out = out.reshape(K, N_ * 8).to(torch.int8)  # (E, K, N)
                return out
            if islow:
                weight_low = unpack_int32_to_int4_signed(weight_low).numpy().astype(np.float64)
                x_low_abs_max = unpack_int32_to_int4_signed(x_low_abs_max).numpy().astype(np.float64)
            else:
                weight_low = (weight_low).numpy().astype(np.float64)
                x_low_abs_max = (x_low_abs_max).numpy().astype(np.float64)
            mm = np.matmul(x_low_abs_max, weight_low.T).astype(float)
            mm = mm * scale_low.astype(float).reshape(1, -1)
            mm = mm * lxScale.reshape(-1,1)
            return mm
        mm_low = CPU_MM_golden(x_low_abs_max.cpu(), weight_low.T.cpu(), self.scale_low.to(torch.float32).cpu().numpy(), lxScale.cpu().numpy(), True)
        mm_high = CPU_MM_golden(x_high_abs_max.cpu(), self.weight_high.cpu(), self.scale_high.to(torch.float32).cpu().numpy(), rxScale.cpu().numpy(), False)
        print(f"low mm error:{torch.mean(output_low.cpu() - mm_low)}")
        print(f"high mm error:{torch.mean(output_high.cpu() - mm_high)}")
        diff = (output_low.cpu() - mm_low).abs().max()
        print(f"Low-bit NPU vs Golden Max Diff: {diff}")
        diff = (output_high.cpu() - mm_high).abs().max()
        print(f"High-bit NPU vs Golden Max Diff: {diff}")
        return torch.add(output_low, output_high)


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
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
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
        
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        # ResQ: reorder columns before o_proj (rearrange_o_proj in project-resq)
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
        
        # 2a. Apply H_butterfly (symmetric, so H = H.T)
        x = hadamard_transform(x)  # acts on last dim (blocksize)
        
        # 2b. Apply Hd on K dimension: x @ Hd.T (Inverse of fusion)
        Hd_f32 = self.Hd.to(x.device, torch.float32)
        x = x.transpose(-1, -2)  # [batch, blocksize, K]
        # We need x_new @ H = x, so x_new = x @ H.T
        x = torch.matmul(x, Hd_f32.t())  # [batch, blocksize, K] @ [K, K]
        x = x.transpose(-1, -2)  # [batch, K, blocksize]
        
        # Step 3: Normalize
        x = x * self.K / math.sqrt(n)
        
        return x.reshape(original_shape).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
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
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP
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
        
        # RoPE
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        
        # Global ResQ params
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
            if f'{prefix}.input_layernorm.weight' in ckpt_a:
                layer.input_layernorm.weight.data = ckpt_a[f'{prefix}.input_layernorm.weight'].to(dtype)
            if f'{prefix}.post_attention_layernorm.weight' in ckpt_a:
                layer.post_attention_layernorm.weight.data = ckpt_a[f'{prefix}.post_attention_layernorm.weight'].to(dtype)
            
            # QK norms
            if f'{prefix}.self_attn.q_norm.weight' in ckpt_a:
                layer.self_attn.q_norm.weight.data = ckpt_a[f'{prefix}.self_attn.q_norm.weight'].to(dtype)
            if f'{prefix}.self_attn.k_norm.weight' in ckpt_a:
                layer.self_attn.k_norm.weight.data = ckpt_a[f'{prefix}.self_attn.k_norm.weight'].to(dtype)
            
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
        """Setup o_proj column reorder (rearrange_o_proj in project-resq)"""
        head_dim = attn.head_dim
        num_heads = attn.num_heads
        in_dim = num_heads * head_dim
        high_fraction = 0.125
        
        high_bits_length = int(in_dim * high_fraction)
        high_per_head = high_bits_length // num_heads
        
        chunk_starts = torch.arange(0, in_dim, head_dim)
        high_precision_columns = torch.arange(head_dim - high_per_head, head_dim)
        columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()
        
        all_columns = torch.arange(in_dim)
        mask = torch.ones(in_dim, dtype=torch.bool)
        mask[columns_to_end] = False
        remaining_columns = all_columns[mask]
        
        new_column_order = torch.cat([remaining_columns, columns_to_end])
        attn.o_proj_column_order = new_column_order
