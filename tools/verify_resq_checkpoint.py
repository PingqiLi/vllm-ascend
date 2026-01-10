#!/usr/bin/env python3
"""
ResQ Checkpoint 验证工具（精简版）

验证 ResQ 量化权重的正确性：
1. 权重融合验证：W_A = f(W_O, Ua, Ub, Uc, Ud, LayerNorm_gamma)
2. 激活值验证：验证中间激活值满足预期的旋转关系

重要：LayerNorm 融合
===================
msmodelslim 在量化前会将 LayerNorm 的 gamma 融合到线性层权重中：
- Q/K/V weights: W_quantized = quantize((W_O * input_ln_gamma) @ Ua)
- gate/up weights: W_quantized = quantize((W_O * post_attn_ln_gamma) @ Ua)
然后将 LayerNorm 的 gamma 设置为 1。

这意味着验证时的 expected 值应该是 (W_O * gamma) @ Ua，而不是 W_O @ Ua。

指标说明：
- corr: 相关系数，衡量方向相似度（尺度无关），高 corr 不代表数值正确！
- rel_err: 相对 L2 误差 = ||restored - expected|| / ||expected||，最重要的指标
- scale_ratio: ||restored|| / ||expected||，快速定位 scale 问题

判定标准：
- 通过: corr >= 0.98, rel_err <= 0.1, scale_ratio in [0.9, 1.1]
- ⚠ 标记表示该指标未通过阈值

用法:
    python tools/verify_resq_checkpoint.py \
        --original /path/to/bf16_model \
        --ckpt-a /path/to/resq_quantized \
        --ckpt-b /path/to/resq_matrices \
        --prompt "你好"
"""

import argparse
import torch
import math
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Tuple

# ============================================================================
# 工具函数
# ============================================================================

from vllm_ascend.models.qwen3_resq_truequant import is_pow2, hadamard_transform, apply_ud_rotation

def load_safetensors(path: str) -> dict:
    """Load all tensors from safetensors files"""
    result = {}
    for f in Path(path).glob("*.safetensors"):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                result[key] = st.get_tensor(key)
    return result

def dequantize(weight_low, weight_high, scale_low, scale_high, debug=False):
    """Dequantize ResQ mixed-precision weights.
    
    msmodelslim symmetric quantization:
    - 4-bit: values in [-8, 7], stored directly as int8 (NO packing!)
    - 8-bit: values in [-128, 127], stored as signed int8
    - dequant: float_val = int_val * scale
    
    Reference: msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/quant_modules.py
    """
    # int4 直接存储为 int8，不需要解包
    w_low = weight_low.float()  # 已经是 [-8, 7]
    # int8 是 signed
    w_high = weight_high.float()  # 已经是 [-128, 127]
    
    if debug:
        print(f"    [dequantize DEBUG]")
        print(f"      weight_low: shape={weight_low.shape}, dtype={weight_low.dtype}, range=[{weight_low.min()}, {weight_low.max()}]")
        print(f"      weight_high: shape={weight_high.shape}, dtype={weight_high.dtype}, range=[{weight_high.min()}, {weight_high.max()}]")
        print(f"      scale_low: shape={scale_low.shape}, dtype={scale_low.dtype}, range=[{scale_low.float().min():.6f}, {scale_low.float().max():.6f}]")
        print(f"      scale_high: shape={scale_high.shape}, dtype={scale_high.dtype}, range=[{scale_high.float().min():.6f}, {scale_high.float().max():.6f}]")
    
    # dequant: W_float = W_int * scale
    w_low_deq = w_low * scale_low.float()
    w_high_deq = w_high * scale_high.float()
    
    if debug:
        print(f"      w_low_deq: ||norm||={w_low_deq.norm():.4f}")
        print(f"      w_high_deq: ||norm||={w_high_deq.norm():.4f}")
        result = torch.cat([w_low_deq, w_high_deq], dim=1)
        print(f"      final: ||norm||={result.norm():.4f}")
        return result
    
    return torch.cat([w_low_deq, w_high_deq], dim=1)


def verify_dequantize_logic(ckpt_a: dict, layer_idx: int = 0):
    """
    验证反量化逻辑是否正确。
    
    对于给定的量化权重，检查：
    1. Scale 值是否合理
    2. 反量化后的权重 norm 是否合理
    3. 与预期的融合权重比较
    """
    print("\n" + "=" * 60)
    print("反量化逻辑诊断")
    print("=" * 60)
    
    prefix = f'model.layers.{layer_idx}.self_attn.q_proj'
    
    weight_low = ckpt_a.get(f'{prefix}.weight_low')
    weight_high = ckpt_a.get(f'{prefix}.weight_high')
    scale_low = ckpt_a.get(f'{prefix}.scale_low')
    scale_high = ckpt_a.get(f'{prefix}.scale_high')
    
    if weight_low is None:
        print(f"  无法找到 {prefix} 的量化权重")
        return
    
    print(f"\n[Layer {layer_idx} Q_proj 反量化诊断]")
    print(f"  weight_low: shape={weight_low.shape}, dtype={weight_low.dtype}")
    print(f"    int 值范围: [{weight_low.min().item()}, {weight_low.max().item()}]")
    print(f"    预期范围 (4-bit signed): [-8, 7]")
    
    print(f"  weight_high: shape={weight_high.shape}, dtype={weight_high.dtype}")
    print(f"    int 值范围: [{weight_high.min().item()}, {weight_high.max().item()}]")
    print(f"    预期范围 (8-bit signed): [-128, 127]")
    
    print(f"  scale_low: shape={scale_low.shape}, dtype={scale_low.dtype}")
    print(f"    scale 值范围: [{scale_low.float().min().item():.6f}, {scale_low.float().max().item():.6f}]")
    print(f"    scale 平均值: {scale_low.float().mean().item():.6f}")
    
    print(f"  scale_high: shape={scale_high.shape}, dtype={scale_high.dtype}")
    print(f"    scale 值范围: [{scale_high.float().min().item():.6f}, {scale_high.float().max().item():.6f}]")
    print(f"    scale 平均值: {scale_high.float().mean().item():.6f}")
    
    # 反量化
    w_deq = dequantize(weight_low, weight_high, scale_low, scale_high)
    print(f"\n  反量化结果:")
    print(f"    shape: {w_deq.shape}")
    print(f"    ||norm||: {w_deq.norm().item():.4f}")
    print(f"    值范围: [{w_deq.min().item():.6f}, {w_deq.max().item():.6f}]")
    
    # 估算预期 norm
    # 对于典型的 LLM，权重 norm 应该在几十到几百之间
    # 如果 scale 很小，可能是量化时出了问题
    out_features, in_features = w_deq.shape
    expected_norm_per_element = 0.02  # 典型值
    expected_total_norm = expected_norm_per_element * (out_features * in_features) ** 0.5
    print(f"\n  预期 norm (估算): ~{expected_total_norm:.1f}")
    print(f"  实际 norm: {w_deq.norm().item():.4f}")
    
    if w_deq.norm().item() < expected_total_norm * 0.1:
        print(f"  ⚠ 警告: 反量化后的权重 norm 异常小！可能存在 scale 问题")
    
    print("=" * 60)

def corrcoef(a, b):
    """Compute correlation coefficient between two tensors"""
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

def compute_metrics(restored, expected, eps=1e-8):
    """
    Compute comprehensive metrics for comparing tensors.
    
    Args:
        restored: The tensor restored from quantized/rotated space
        expected: The expected tensor (original/ground truth)
        eps: Small epsilon for numerical stability
        
    Returns:
        Dict with:
        - corr: Correlation coefficient (direction similarity)
        - rel_err: Relative L2 error ||restored - expected|| / ||expected||
        - scale_ratio: ||restored|| / ||expected|| (quick scale check)
        - max_abs_err: Maximum absolute error
        - mean_abs_err: Mean absolute error
        
    Note on metrics:
        - corr: Measures direction/shape similarity, IGNORES scale!
        - rel_err: The most important metric, measures actual numerical accuracy
        - scale_ratio: Should be ~1.0, quick indicator of scale problems
    """
    restored_f = restored.flatten().float()
    expected_f = expected.flatten().float()
    
    # Correlation (direction similarity, scale-invariant)
    corr = torch.corrcoef(torch.stack([restored_f, expected_f]))[0, 1].item()
    
    # Relative L2 error (the key metric for correctness)
    diff_norm = (restored_f - expected_f).norm().item()
    expected_norm = expected_f.norm().item()
    rel_err = diff_norm / (expected_norm + eps)
    
    # Scale ratio (quick scale check)
    restored_norm = restored_f.norm().item()
    scale_ratio = restored_norm / (expected_norm + eps)
    
    # Absolute errors
    abs_diff = (restored_f - expected_f).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    
    return {
        'corr': corr,
        'rel_err': rel_err,
        'scale_ratio': scale_ratio,
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'restored_norm': restored_norm,
        'expected_norm': expected_norm,
    }

def format_metrics(metrics, name="", thresholds=None):
    """
    Format metrics with pass/fail indicators.
    
    Args:
        metrics: Dict from compute_metrics
        name: Name of the check
        thresholds: Dict with 'corr_min', 'rel_err_max', 'scale_ratio_range'
                   Defaults: corr>0.98, rel_err<0.1, scale_ratio in [0.9, 1.1]
    
    Returns:
        Formatted string with status
    """
    if thresholds is None:
        thresholds = {
            'corr_min': 0.98,
            'rel_err_max': 0.1,  # 10% relative error
            'scale_ratio_range': (0.9, 1.1),
        }
    
    corr = metrics['corr']
    rel_err = metrics['rel_err']
    scale_ratio = metrics['scale_ratio']
    
    # Determine status
    corr_ok = corr >= thresholds['corr_min']
    rel_err_ok = rel_err <= thresholds['rel_err_max']
    scale_ok = thresholds['scale_ratio_range'][0] <= scale_ratio <= thresholds['scale_ratio_range'][1]
    
    all_ok = corr_ok and rel_err_ok and scale_ok
    status = "✓" if all_ok else "✗"
    
    # Build message
    corr_status = "" if corr_ok else " ⚠"
    rel_err_status = "" if rel_err_ok else " ⚠"
    scale_status = "" if scale_ok else " ⚠"
    
    msg = (f"{status} {name}: "
           f"corr={corr:.4f}{corr_status}, "
           f"rel_err={rel_err:.4f}{rel_err_status}, "
           f"scale={scale_ratio:.4f}{scale_status}")
    
    return msg, all_ok

# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    def __init__(self, original_model, ckpt_a: dict, ckpt_b: dict):
        self.original = original_model
        self.ckpt_a = ckpt_a
        self.ckpt_b = ckpt_b
        self.config = original_model.config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.intermediate_size = self.config.intermediate_size
        
        # head_dim: prefer config, otherwise infer from k_proj weight shape
        if hasattr(self.config, 'head_dim') and self.config.head_dim:
            self.head_dim = self.config.head_dim
        else:
            # For GQA models: kv_size = num_kv_heads * head_dim
            # Infer from k_proj weight shape
            k_proj_weight = None
            for key in ckpt_a:
                if 'layers.0.self_attn.k_proj.weight_low' in key:
                    k_proj_weight = ckpt_a[key]
                    break
            if k_proj_weight is not None:
                kv_size = k_proj_weight.shape[0]
                self.head_dim = kv_size // self.num_kv_heads
            else:
                # Fallback
                self.head_dim = self.hidden_size // self.num_heads
        
        print(f"[Config] hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
              f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")
    
    def get_original(self, name: str) -> Optional[torch.Tensor]:
        """Get original weight by name"""
        try:
            parts = name.split('.')
            obj = self.original
            for p in parts:
                obj = getattr(obj, p) if hasattr(obj, p) else obj[int(p)]
            return obj.weight.data if hasattr(obj, 'weight') else obj.data
        except:
            return None
    
    def get_quantized(self, prefix: str) -> Optional[torch.Tensor]:
        """Get dequantized weight from ckpt_a"""
        keys = ['weight_low', 'weight_high', 'scale_low', 'scale_high']
        tensors = [self.ckpt_a.get(f'{prefix}.{k}') for k in keys]
        if any(t is None for t in tensors):
            return None
        return dequantize(*tensors)
    
    def get_ua(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Ua = Pa @ Ra"""
        Pa = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_a')
        Ra = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_a')
        return torch.matmul(Pa.float(), Ra.float()) if Pa is not None and Ra is not None else None
    
    def get_ub(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Ub = Pb @ Rb (per-kv-head)"""
        Pb = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_b')
        Rb = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_b')
        return torch.matmul(Pb.float(), Rb.float()) if Pb is not None and Rb is not None else None
    
    def get_uc(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Uc = Pc @ Rc"""
        Pc = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_c')
        Rc = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_c')
        return torch.matmul(Pc.float(), Rc.float()) if Pc is not None and Rc is not None else None
    
    def get_pd(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get Pd matrix"""
        return self.ckpt_b.get(f'resq.layer.{layer_idx}.P_d')
    
    def get_hd(self) -> Tuple[Optional[torch.Tensor], int]:
        """Get Hd matrix and K"""
        Hd = self.ckpt_a.get('resq.Hd')
        K = Hd.shape[0] if Hd is not None else 1
        return Hd, K
    
    def get_input_layernorm_gamma(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get input_layernorm gamma (weight) from original model.
        
        msmodelslim fuses this into Q/K/V weights: W_new = W_O * gamma
        Then sets gamma to 1 in the quantized checkpoint.
        """
        try:
            layer = self.original.model.layers[layer_idx]
            return layer.input_layernorm.weight.data.float()
        except:
            return None
    
    def get_post_attention_layernorm_gamma(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get post_attention_layernorm gamma (weight) from original model.
        
        msmodelslim fuses this into gate/up weights: W_new = W_O * gamma
        Then sets gamma to 1 in the quantized checkpoint.
        """
        try:
            layer = self.original.model.layers[layer_idx]
            return layer.post_attention_layernorm.weight.data.float()
        except:
            return None

# ============================================================================
# Verifier
# ============================================================================

class ResQVerifier:
    def __init__(self, mgr: CheckpointManager):
        self.mgr = mgr
        self.results = {}
    
    def verify_embed(self) -> dict:
        """Verify: embed_A = embed_O @ Ua
        
        Returns:
            Dict with metrics (corr, rel_err, scale_ratio, etc.)
        """
        embed_O = self.mgr.get_original('model.embed_tokens')
        embed_A = self.mgr.ckpt_a.get('model.embed_tokens.weight')
        Ua = self.mgr.get_ua(0)
        if any(x is None for x in [embed_O, embed_A, Ua]):
            return {'corr': float('nan'), 'rel_err': float('nan'), 'scale_ratio': float('nan')}
        
        expected = torch.matmul(embed_O.float(), Ua)
        metrics = compute_metrics(embed_A, expected)
        self.results['embed'] = metrics
        return metrics
    
    def verify_qkv(self, layer_idx: int, debug: bool = False) -> Dict[str, dict]:
        """Verify Q/K/V weight fusion with LayerNorm fusion
        
        msmodelslim fuses input_layernorm gamma into Q/K/V weights before quantization:
            W_fused = W_O * gamma  (element-wise, broadcast along input dim)
            W_quantized = quantize(W_fused @ Ua)
        
        So the expected value is: (W_O * gamma) @ Ua
        
        Returns:
            Dict mapping proj name to metrics dict
        """
        results = {}
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx)
        if Ua is None:
            return results
        
        prefix = f'model.layers.{layer_idx}.self_attn'
        
        # Get LayerNorm gamma for fusion
        ln_gamma = self.mgr.get_input_layernorm_gamma(layer_idx)
        
        # Q_A = (Q_O * gamma) @ Ua, K_A = (K_O * gamma) @ Ua
        for name in ['q_proj', 'k_proj']:
            W_O = self.mgr.get_original(f'{prefix}.{name}')
            W_A = self.mgr.get_quantized(f'{prefix}.{name}')
            if W_O is not None and W_A is not None:
                # Fuse LayerNorm gamma: W_fused = W_O * gamma (broadcast: [out, in] * [in])
                if ln_gamma is not None:
                    W_fused = W_O.float() * ln_gamma.unsqueeze(0)
                else:
                    W_fused = W_O.float()
                expected = torch.matmul(W_fused, Ua)
                results[name] = compute_metrics(W_A, expected)
                
                if debug and layer_idx == 0:
                    # Debug: print raw values
                    weight_low = self.mgr.ckpt_a.get(f'{prefix}.{name}.weight_low')
                    weight_high = self.mgr.ckpt_a.get(f'{prefix}.{name}.weight_high')
                    scale_low = self.mgr.ckpt_a.get(f'{prefix}.{name}.scale_low')
                    scale_high = self.mgr.ckpt_a.get(f'{prefix}.{name}.scale_high')
                    
                    print(f"\n  [DEBUG] {name} raw values (with LN gamma fusion):")
                    print(f"    W_O shape: {W_O.shape}, ||W_O||={W_O.float().norm():.4f}")
                    if ln_gamma is not None:
                        print(f"    ln_gamma: shape={ln_gamma.shape}, range=[{ln_gamma.min():.4f}, {ln_gamma.max():.4f}], mean={ln_gamma.mean():.4f}")
                        print(f"    W_fused (W_O * gamma): ||W_fused||={W_fused.norm():.4f}")
                    print(f"    expected ((W_O * gamma) @ Ua): ||expected||={expected.norm():.4f}")
                    if weight_low is not None:
                        print(f"    weight_low: shape={weight_low.shape}, dtype={weight_low.dtype}")
                        print(f"      int range: [{weight_low.min()}, {weight_low.max()}]")
                    if weight_high is not None:
                        print(f"    weight_high: shape={weight_high.shape}, dtype={weight_high.dtype}")
                        print(f"      int range: [{weight_high.min()}, {weight_high.max()}]")
                    if scale_low is not None:
                        print(f"    scale_low: shape={scale_low.shape}, dtype={scale_low.dtype}")
                        print(f"      scale range: [{scale_low.min():.6f}, {scale_low.max():.6f}]")
                    if scale_high is not None:
                        print(f"    scale_high: shape={scale_high.shape}, dtype={scale_high.dtype}")
                        print(f"      scale range: [{scale_high.min():.6f}, {scale_high.max():.6f}]")
                    print(f"    W_A (dequantized): ||W_A||={W_A.norm():.4f}")
                    
                    # Compare with different fusion strategies
                    W_O_only = compute_metrics(W_A, W_O.float())
                    W_O_Ua_only = compute_metrics(W_A, torch.matmul(W_O.float(), Ua))
                    print(f"    Compare W_A vs W_O (no fusion): corr={W_O_only['corr']:.4f}, scale={W_O_only['scale_ratio']:.4f}")
                    print(f"    Compare W_A vs W_O @ Ua (no gamma): corr={W_O_Ua_only['corr']:.4f}, scale={W_O_Ua_only['scale_ratio']:.4f}")
                    print(f"    Compare W_A vs (W_O * gamma) @ Ua: corr={results[name]['corr']:.4f}, scale={results[name]['scale_ratio']:.4f}")
        
        # V_A[h] = Ub[h].T @ (V_O * gamma)[h] @ Ua
        if Ub is not None:
            V_O = self.mgr.get_original(f'{prefix}.v_proj')
            V_A = self.mgr.get_quantized(f'{prefix}.v_proj')
            if V_O is not None and V_A is not None:
                # Fuse LayerNorm gamma first
                if ln_gamma is not None:
                    V_fused = V_O.float() * ln_gamma.unsqueeze(0)
                else:
                    V_fused = V_O.float()
                V_fused_reshaped = V_fused.view(self.mgr.num_kv_heads, self.mgr.head_dim, -1)
                # einsum: 'nji,nje->nie' computes Ub.T @ V_fused per-head
                V_expected = torch.einsum('nji,nje->nie', Ub.float(), V_fused_reshaped)
                V_expected = torch.matmul(V_expected, Ua)
                V_expected = V_expected.view_as(V_A)
                results['v_proj'] = compute_metrics(V_A, V_expected)
        
        self.results[f'layer{layer_idx}_qkv'] = results
        return results
    
    def verify_o_proj(self, layer_idx: int, use_rearrange_logic: bool = False) -> dict:
        """Verify O_proj weight fusion with column rearrangement
        
        Returns:
            Dict with metrics (corr, rel_err, scale_ratio, etc.)
        """
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx)
        if Ua is None or Ub is None:
            return {'corr': float('nan'), 'rel_err': float('nan'), 'scale_ratio': float('nan')}
        
        prefix = f'model.layers.{layer_idx}.self_attn.o_proj'
        O_O = self.mgr.get_original(f'model.layers.{layer_idx}.self_attn.o_proj')
        O_A = self.mgr.get_quantized(prefix)
        if O_O is None or O_A is None:
            return {'corr': float('nan'), 'rel_err': float('nan'), 'scale_ratio': float('nan')}
        
        hidden_size = self.mgr.hidden_size
        num_heads = self.mgr.num_heads
        head_dim = self.mgr.head_dim
        high_fraction = 0.125
        
        # Expand Ub for GQA
        num_q_per_kv = num_heads // self.mgr.num_kv_heads
        Ub_expanded = Ub.repeat_interleave(num_q_per_kv, dim=0)
        
        # O_expected = Ua.T @ O_O @ block_diag(Ub)
        O_O_reshaped = O_O.float().view(hidden_size, num_heads, head_dim)
        # einsum: 'enh,nhd->end' computes O_O @ Ub per-head
        O_fused = torch.einsum('enh,nhd->end', O_O_reshaped, Ub_expanded.float())
        O_expected = torch.matmul(Ua.float().T, O_fused.reshape(hidden_size, -1))
        
        # Apply column rearrangement
        if use_rearrange_logic:
            high_bits_length = int(hidden_size * high_fraction)
        else:
            high_bits_length = int(num_heads * head_dim * high_fraction)
        
        high_per_head = high_bits_length // num_heads
        low_per_head = head_dim - high_per_head
        
        column_order = []
        for h in range(num_heads):
            column_order.extend(range(h * head_dim, h * head_dim + low_per_head))
        for h in range(num_heads):
            column_order.extend(range(h * head_dim + low_per_head, (h + 1) * head_dim))
        
        O_expected_rearranged = O_expected[:, column_order]
        metrics = compute_metrics(O_A, O_expected_rearranged)
        self.results[f'layer{layer_idx}_o_proj'] = metrics
        return metrics
    
    def verify_mlp(self, layer_idx: int) -> Dict[str, dict]:
        """Verify MLP weight fusion with LayerNorm fusion
        
        msmodelslim fuses post_attention_layernorm gamma into gate/up weights:
            W_fused = W_O * gamma  (element-wise, broadcast along input dim)
            W_quantized = quantize(W_fused @ Ua)
        
        So the expected value is: (W_O * gamma) @ Ua
        
        Returns:
            Dict mapping proj name to metrics dict
        """
        results = {}
        Ua = self.mgr.get_ua(layer_idx)
        Pd = self.mgr.get_pd(layer_idx)
        Hd, K = self.mgr.get_hd()
        if Ua is None:
            return results
        
        prefix = f'model.layers.{layer_idx}.mlp'
        
        # Get LayerNorm gamma for fusion
        ln_gamma = self.mgr.get_post_attention_layernorm_gamma(layer_idx)
        
        # gate_A = (gate_O * gamma) @ Ua, up_A = (up_O * gamma) @ Ua
        for name in ['gate_proj', 'up_proj']:
            W_O = self.mgr.get_original(f'{prefix}.{name}')
            W_A = self.mgr.get_quantized(f'{prefix}.{name}')
            if W_O is not None and W_A is not None:
                # Fuse LayerNorm gamma: W_fused = W_O * gamma (broadcast: [out, in] * [in])
                if ln_gamma is not None:
                    W_fused = W_O.float() * ln_gamma.unsqueeze(0)
                else:
                    W_fused = W_O.float()
                expected = torch.matmul(W_fused, Ua)
                results[name] = compute_metrics(W_A, expected)
        
        # down_A = Ua.T @ down_O @ Ud (Ud applied online, so verify structure)
        # For now, just verify the Ua.T fusion on output
        W_O = self.mgr.get_original(f'{prefix}.down_proj')
        W_A = self.mgr.get_quantized(f'{prefix}.down_proj')
        if W_O is not None and W_A is not None and Pd is not None:
            # down_A[:, rearranged] = Ua.T @ down_O @ Ud[:, rearranged]
            # Simplified: check correlation after Ua.T fusion
            intermediate_size = self.mgr.intermediate_size
            blocksize = intermediate_size // K
            high_fraction = 0.125
            high_bits = int(intermediate_size * high_fraction)
            low_bits = intermediate_size - high_bits
            
            # Build Ud transformation for verification
            # Ud = BlockDiag(Pd.T) @ H, but H is complex
            # For now, verify that down_A @ Ua ≈ some transformation of down_O
            # This is a simplified check
            results['down_proj'] = 'structure_check_needed'
        
        self.results[f'layer{layer_idx}_mlp'] = results
        return results
    
    def verify_activations(self, tokenizer, prompt: str, layer_idx: int = 0) -> Dict[str, dict]:
        """
        完整验证 Layer 0 内每个算子的激活值。
        
        注意 LayerNorm 融合：
        - msmodelslim 将 LayerNorm gamma 融合到权重中，然后将 gamma 设为 1
        - 所以量化后的权重 W_A 已经包含了 gamma 的效果
        - 激活值验证时使用原始模型（gamma != 1），所以 Q/K/V 输出应该是一致的：
          - 原始：LN(x, gamma) @ W_O.T = (x * gamma / rms) @ W_O.T
          - ResQ：LN(x, 1) @ W_A.T = (x / rms) @ (W_O * gamma).T = 原始结果
        
        计算流程：
        1. Embed: x_orig, x_resq (在 Ua 空间)
        2. input_layernorm (gamma != 1 for orig, gamma = 1 for resq)
        3. Q/K/V projections (权重已融合 gamma)
        4. QK norm (Qwen3)
        5. RoPE
        6. Uc rotation (如果有)
        7. Attention: softmax(QK^T / sqrt(d)) @ V
        8. O_proj (包括列重排)
        9. Residual + post_attention_layernorm
        10. Gate/Up projections (权重已融合 gamma)
        11. SiLU(gate) * up
        12. Ud rotation
        13. Down projection
        14. Residual
        
        Returns:
            Dict mapping check names to metrics dict containing:
            - corr: Correlation coefficient
            - rel_err: Relative L2 error
            - scale_ratio: Norm ratio (restored/expected)
            - etc.
        """
        results = {}
        import torch.nn.functional as F
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]
        
        # 获取旋转矩阵
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx)
        Uc = self.mgr.get_uc(layer_idx)
        Pd = self.mgr.get_pd(layer_idx)
        Hd, K = self.mgr.get_hd()
        
        num_heads = self.mgr.num_heads
        num_kv_heads = self.mgr.num_kv_heads
        head_dim = self.mgr.head_dim
        num_q_per_kv = num_heads // num_kv_heads
        hidden_size = self.mgr.hidden_size
        intermediate_size = self.mgr.intermediate_size
        
        print(f"\n[Layer {layer_idx} 完整激活值验证]")
        print(f"  seq_len={seq_len}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        with torch.no_grad():
            layer = self.mgr.original.model.layers[layer_idx]
            prefix = f'model.layers.{layer_idx}'
            
            # ============================================================
            # Step 1: Embed
            # ============================================================
            embed_O = self.mgr.original.model.embed_tokens
            embed_A_weight = self.mgr.ckpt_a.get('model.embed_tokens.weight')
            
            x_orig = embed_O(input_ids).float()
            x_resq = embed_A_weight.float()[input_ids.squeeze(0)].unsqueeze(0)
            
            # x_resq = x_orig @ Ua, 验证 x_resq @ Ua.T ≈ x_orig
            x_restored = torch.matmul(x_resq, Ua.float().T)
            results['1.embed'] = compute_metrics(x_restored, x_orig)
            msg, _ = format_metrics(results['1.embed'], "1. Embed (x_resq @ Ua.T ≈ x_orig)")
            print(f"  {msg}")
            
            # ============================================================
            # Step 2: Input LayerNorm
            # ============================================================
            # 原始模型使用 gamma != 1
            ln1 = layer.input_layernorm
            x_orig_ln = ln1(x_orig.to(ln1.weight.dtype)).float()
            
            # ResQ 路径：ckpt A 中 gamma=1（因为 gamma 已融合到权重中）
            # RMSNorm with gamma=1: output = x / rms(x)
            # 手动实现 gamma=1 的 RMSNorm
            x_resq_f = x_resq.float()
            variance = x_resq_f.pow(2).mean(-1, keepdim=True)
            x_resq_ln = x_resq_f * torch.rsqrt(variance + ln1.variance_epsilon)
            
            # LayerNorm 不改变空间，仍然满足 x_resq_ln @ Ua.T ≈ x_orig_ln
            x_ln_restored = torch.matmul(x_resq_ln, Ua.float().T)
            results['2.input_ln'] = compute_metrics(x_ln_restored, x_orig_ln)
            msg, _ = format_metrics(results['2.input_ln'], "2. Input LN (x_resq_ln @ Ua.T ≈ x_orig_ln)")
            print(f"  {msg}")
            
            # ============================================================
            # Step 3: Q/K/V Projections
            # ============================================================
            attn = layer.self_attn
            attn_prefix = f'{prefix}.self_attn'
            
            # Q: Q_A = Q_O @ Ua, 所以 q_resq ≈ q_orig
            Q_O = attn.q_proj.weight.data.float()
            Q_A = self.mgr.get_quantized(f'{attn_prefix}.q_proj')
            q_orig = torch.matmul(x_orig_ln, Q_O.T)
            q_resq = torch.matmul(x_resq_ln, Q_A.float().T) if Q_A is not None else None
            if q_resq is not None:
                results['3a.q_proj'] = compute_metrics(q_resq, q_orig)
                msg, _ = format_metrics(results['3a.q_proj'], "3a. Q_proj (q_resq ≈ q_orig)")
                print(f"  {msg}")
            
            # K: K_A = K_O @ Ua, 所以 k_resq ≈ k_orig
            K_O = attn.k_proj.weight.data.float()
            K_A = self.mgr.get_quantized(f'{attn_prefix}.k_proj')
            k_orig = torch.matmul(x_orig_ln, K_O.T)
            k_resq = torch.matmul(x_resq_ln, K_A.float().T) if K_A is not None else None
            if k_resq is not None:
                results['3b.k_proj'] = compute_metrics(k_resq, k_orig)
                msg, _ = format_metrics(results['3b.k_proj'], "3b. K_proj (k_resq ≈ k_orig)")
                print(f"  {msg}")
            
            # V: V_A = Ub.T @ V_O @ Ua, 所以 v_resq = v_orig @ Ub
            V_O = attn.v_proj.weight.data.float()
            V_A = self.mgr.get_quantized(f'{attn_prefix}.v_proj')
            v_orig = torch.matmul(x_orig_ln, V_O.T)
            v_resq = torch.matmul(x_resq_ln, V_A.float().T) if V_A is not None else None
            if v_resq is not None and Ub is not None:
                # v_resq @ Ub.T ≈ v_orig
                v_resq_4d = v_resq.view(1, seq_len, num_kv_heads, head_dim)
                v_orig_4d = v_orig.view(1, seq_len, num_kv_heads, head_dim)
                v_restored = torch.einsum('bsnh,ndh->bsnd', v_resq_4d, Ub.float())
                results['3c.v_proj'] = compute_metrics(v_restored, v_orig_4d)
                msg, _ = format_metrics(results['3c.v_proj'], "3c. V_proj (v_resq @ Ub.T ≈ v_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 4: QK Norm (Qwen3 specific)
            # ============================================================
            if hasattr(attn, 'q_norm') and attn.q_norm is not None:
                q_orig_4d = q_orig.view(1, seq_len, num_heads, head_dim)
                k_orig_4d = k_orig.view(1, seq_len, num_kv_heads, head_dim)
                q_resq_4d = q_resq.view(1, seq_len, num_heads, head_dim) if q_resq is not None else None
                k_resq_4d = k_resq.view(1, seq_len, num_kv_heads, head_dim) if k_resq is not None else None
                
                # 应用 QK norm
                q_orig_normed = attn.q_norm(q_orig_4d.to(attn.q_norm.weight.dtype)).float()
                k_orig_normed = attn.k_norm(k_orig_4d.to(attn.k_norm.weight.dtype)).float()
                if q_resq_4d is not None:
                    q_resq_normed = attn.q_norm(q_resq_4d.to(attn.q_norm.weight.dtype)).float()
                    k_resq_normed = attn.k_norm(k_resq_4d.to(attn.k_norm.weight.dtype)).float()
                    results['4a.q_norm'] = compute_metrics(q_resq_normed, q_orig_normed)
                    results['4b.k_norm'] = compute_metrics(k_resq_normed, k_orig_normed)
                    msg, _ = format_metrics(results['4a.q_norm'], "4a. Q_norm")
                    print(f"  {msg}")
                    msg, _ = format_metrics(results['4b.k_norm'], "4b. K_norm")
                    print(f"  {msg}")
            else:
                q_orig_normed = q_orig.view(1, seq_len, num_heads, head_dim)
                k_orig_normed = k_orig.view(1, seq_len, num_kv_heads, head_dim)
                q_resq_normed = q_resq.view(1, seq_len, num_heads, head_dim) if q_resq is not None else None
                k_resq_normed = k_resq.view(1, seq_len, num_kv_heads, head_dim) if k_resq is not None else None
            
            # ============================================================
            # Step 5: RoPE (位置编码)
            # ============================================================
            # 简化：跳过 RoPE，因为它对两边都一样应用
            # 实际验证中可以添加完整的 RoPE 实现
            q_orig_rope = q_orig_normed
            k_orig_rope = k_orig_normed
            q_resq_rope = q_resq_normed
            k_resq_rope = k_resq_normed
            print(f"  5. RoPE: (skipped, applied identically to both)")
            
            # ============================================================
            # Step 6: Uc Rotation (如果有)
            # ============================================================
            if Uc is not None and Uc.numel() > 0:
                # Uc 应用于 Q 和 K
                # q @ Uc, k @ Uc
                q_orig_uc = torch.matmul(q_orig_rope, Uc.float())
                k_orig_uc = torch.matmul(k_orig_rope, Uc.float())
                if q_resq_rope is not None:
                    q_resq_uc = torch.matmul(q_resq_rope, Uc.float())
                    k_resq_uc = torch.matmul(k_resq_rope, Uc.float())
                    results['6.uc_rotation'] = compute_metrics(q_resq_uc, q_orig_uc)
                    msg, _ = format_metrics(results['6.uc_rotation'], "6. Uc rotation (q_resq_uc ≈ q_orig_uc)")
                    print(f"  {msg}")
            else:
                q_orig_uc = q_orig_rope
                k_orig_uc = k_orig_rope
                q_resq_uc = q_resq_rope
                k_resq_uc = k_resq_rope
                print(f"  6. Uc rotation: (no Uc matrix)")
            
            # ============================================================
            # Step 7: Attention: softmax(QK^T / sqrt(d)) @ V
            # ============================================================
            # 扩展 K, V 到 num_heads
            k_orig_expanded = k_orig_uc.repeat_interleave(num_q_per_kv, dim=2)  # [1, seq, num_heads, head_dim]
            v_orig_expanded = v_orig.view(1, seq_len, num_kv_heads, head_dim).repeat_interleave(num_q_per_kv, dim=2)
            
            # QK^T / sqrt(d)
            # [1, num_heads, seq, head_dim] @ [1, num_heads, head_dim, seq] -> [1, num_heads, seq, seq]
            q_t = q_orig_uc.transpose(1, 2)  # [1, num_heads, seq, head_dim]
            k_t = k_orig_expanded.transpose(1, 2)  # [1, num_heads, seq, head_dim]
            scores_orig = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights_orig = F.softmax(scores_orig, dim=-1)
            
            # Attention output: [1, num_heads, seq, seq] @ [1, num_heads, seq, head_dim]
            v_t = v_orig_expanded.transpose(1, 2)  # [1, num_heads, seq, head_dim]
            attn_output_orig = torch.matmul(attn_weights_orig, v_t)  # [1, num_heads, seq, head_dim]
            attn_output_orig = attn_output_orig.transpose(1, 2).reshape(1, seq_len, num_heads * head_dim)
            
            # ResQ 侧
            if q_resq_uc is not None and k_resq_uc is not None and v_resq is not None:
                k_resq_expanded = k_resq_uc.repeat_interleave(num_q_per_kv, dim=2)
                v_resq_expanded = v_resq.view(1, seq_len, num_kv_heads, head_dim).repeat_interleave(num_q_per_kv, dim=2)
                
                q_t_resq = q_resq_uc.transpose(1, 2)
                k_t_resq = k_resq_expanded.transpose(1, 2)
                scores_resq = torch.matmul(q_t_resq, k_t_resq.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights_resq = F.softmax(scores_resq, dim=-1)
                
                v_t_resq = v_resq_expanded.transpose(1, 2)
                attn_output_resq = torch.matmul(attn_weights_resq, v_t_resq)
                attn_output_resq = attn_output_resq.transpose(1, 2).reshape(1, seq_len, num_heads * head_dim)
                
                # attn_output_resq = attn_output_orig @ Ub (per-head)
                # 验证：attn_output_resq @ Ub.T ≈ attn_output_orig
                Ub_expanded = Ub.repeat_interleave(num_q_per_kv, dim=0)
                attn_resq_4d = attn_output_resq.view(1, seq_len, num_heads, head_dim)
                attn_orig_4d = attn_output_orig.view(1, seq_len, num_heads, head_dim)
                attn_restored = torch.einsum('bsnh,ndh->bsnd', attn_resq_4d, Ub_expanded.float())
                results['7.attn_output'] = compute_metrics(attn_restored, attn_orig_4d)
                msg, _ = format_metrics(results['7.attn_output'], "7. Attn output (attn_resq @ Ub.T ≈ attn_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 8: O_proj (包括列重排)
            # ============================================================
            O_O = attn.o_proj.weight.data.float()
            O_A = self.mgr.get_quantized(f'{attn_prefix}.o_proj')
            
            if O_A is not None and Ub is not None and Ua is not None:
                # 原始 O_proj
                o_orig = torch.matmul(attn_output_orig, O_O.T)
                
                # ResQ: 需要先重排 attn_output_resq
                high_fraction = 0.125
                high_per_head = int(head_dim * high_fraction)
                low_per_head = head_dim - high_per_head
                
                column_order = []
                for h in range(num_heads):
                    column_order.extend(range(h * head_dim, h * head_dim + low_per_head))
                for h in range(num_heads):
                    column_order.extend(range(h * head_dim + low_per_head, (h + 1) * head_dim))
                
                attn_resq_reordered = attn_output_resq[..., column_order]
                o_resq = torch.matmul(attn_resq_reordered, O_A.float().T)
                
                # o_resq @ Ua.T ≈ o_orig
                o_restored = torch.matmul(o_resq, Ua.float().T)
                results['8.o_proj'] = compute_metrics(o_restored, o_orig)
                msg, _ = format_metrics(results['8.o_proj'], "8. O_proj (o_resq @ Ua.T ≈ o_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 9: Residual + Post-Attention LayerNorm
            # ============================================================
            # residual = x + o
            residual_orig = x_orig + o_orig if 'o_orig' in dir() else x_orig
            residual_resq = x_resq + o_resq if 'o_resq' in dir() else x_resq
            
            # DEBUG: 打印各张量的范数和误差 (关键诊断信息)
            print(f"\n  [DEBUG] Norm analysis (critical for diagnosing scale issues):")
            print(f"    ||x_orig|| = {x_orig.norm().item():.4f}")
            print(f"    ||x_resq|| = {x_resq.norm().item():.4f}")
            print(f"    ||o_orig|| = {o_orig.norm().item():.4f}")
            print(f"    ||o_resq|| = {o_resq.norm().item():.4f}")
            x_diff = (torch.matmul(x_resq, Ua.float().T) - x_orig).norm().item()
            o_diff = (torch.matmul(o_resq, Ua.float().T) - o_orig).norm().item()
            print(f"    ||x_resq @ Ua.T - x_orig|| = {x_diff:.4f} (rel: {x_diff / (x_orig.norm().item() + 1e-8):.4f})")
            print(f"    ||o_resq @ Ua.T - o_orig|| = {o_diff:.4f} (rel: {o_diff / (o_orig.norm().item() + 1e-8):.4f})")
            print(f"    ||residual_orig|| = {residual_orig.norm().item():.4f}")
            print(f"    ||residual_resq|| = {residual_resq.norm().item():.4f}")
            
            # 验证 residual 仍在 Ua 空间
            residual_restored = torch.matmul(residual_resq, Ua.float().T)
            results['9a.residual1'] = compute_metrics(residual_restored, residual_orig)
            msg, _ = format_metrics(results['9a.residual1'], "9a. Residual1 (residual_resq @ Ua.T ≈ residual_orig)")
            print(f"  {msg}")
            
            # 原始模型使用 gamma != 1
            ln2 = layer.post_attention_layernorm
            mlp_input_orig = ln2(residual_orig.to(ln2.weight.dtype)).float()
            
            # ResQ 路径：ckpt A 中 gamma=1（因为 gamma 已融合到权重中）
            # RMSNorm with gamma=1: output = x / rms(x)
            residual_resq_f = residual_resq.float()
            variance = residual_resq_f.pow(2).mean(-1, keepdim=True)
            mlp_input_resq = residual_resq_f * torch.rsqrt(variance + ln2.variance_epsilon)
            
            mlp_input_restored = torch.matmul(mlp_input_resq, Ua.float().T)
            results['9b.post_attn_ln'] = compute_metrics(mlp_input_restored, mlp_input_orig)
            msg, _ = format_metrics(results['9b.post_attn_ln'], "9b. Post-Attn LN (mlp_input_resq @ Ua.T ≈ mlp_input_orig)")
            print(f"  {msg}")
            
            # ============================================================
            # Step 10: Gate/Up Projections
            # ============================================================
            mlp = layer.mlp
            mlp_prefix = f'{prefix}.mlp'
            
            gate_O = mlp.gate_proj.weight.data.float()
            gate_A = self.mgr.get_quantized(f'{mlp_prefix}.gate_proj')
            gate_orig = torch.matmul(mlp_input_orig, gate_O.T)
            gate_resq = torch.matmul(mlp_input_resq, gate_A.float().T) if gate_A is not None else None
            
            if gate_resq is not None:
                results['10a.gate_proj'] = compute_metrics(gate_resq, gate_orig)
                msg, _ = format_metrics(results['10a.gate_proj'], "10a. Gate_proj (gate_resq ≈ gate_orig)")
                print(f"  {msg}")
            
            up_O = mlp.up_proj.weight.data.float()
            up_A = self.mgr.get_quantized(f'{mlp_prefix}.up_proj')
            up_orig = torch.matmul(mlp_input_orig, up_O.T)
            up_resq = torch.matmul(mlp_input_resq, up_A.float().T) if up_A is not None else None
            
            if up_resq is not None:
                results['10b.up_proj'] = compute_metrics(up_resq, up_orig)
                msg, _ = format_metrics(results['10b.up_proj'], "10b. Up_proj (up_resq ≈ up_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 11: SiLU(gate) * up
            # ============================================================
            mlp_hidden_orig = F.silu(gate_orig) * up_orig
            mlp_hidden_resq = F.silu(gate_resq) * up_resq if gate_resq is not None and up_resq is not None else None
            
            if mlp_hidden_resq is not None:
                results['11.mlp_hidden'] = compute_metrics(mlp_hidden_resq, mlp_hidden_orig)
                msg, _ = format_metrics(results['11.mlp_hidden'], "11. MLP hidden (mlp_hidden_resq ≈ mlp_hidden_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 12: Ud Rotation
            # ============================================================
            if Pd is not None and mlp_hidden_resq is not None:
                blocksize = intermediate_size // K
                mlp_hidden_ud = apply_ud_rotation(mlp_hidden_resq, Pd, Hd, K, blocksize)
                # Ud 是正交变换，验证 norm 保持
                norm_ratio = mlp_hidden_ud.norm() / mlp_hidden_resq.norm()
                is_orthogonal = abs(norm_ratio.item() - 1.0) < 0.01
                results['12.ud_rotation'] = {
                    'is_orthogonal': is_orthogonal,
                    'norm_ratio': norm_ratio.item(),
                    'corr': 1.0 if is_orthogonal else 0.0,  # placeholder for consistent format
                    'rel_err': abs(norm_ratio.item() - 1.0),
                    'scale_ratio': norm_ratio.item(),
                }
                status = "✓" if is_orthogonal else "✗"
                print(f"  {status} 12. Ud rotation: norm_preserved={is_orthogonal}, ratio={norm_ratio.item():.4f}")
            else:
                mlp_hidden_ud = mlp_hidden_resq
            
            # ============================================================
            # Step 13: Down Projection
            # ============================================================
            down_O = mlp.down_proj.weight.data.float()
            down_A = self.mgr.get_quantized(f'{mlp_prefix}.down_proj')
            
            down_orig = torch.matmul(mlp_hidden_orig, down_O.T)
            if down_A is not None and mlp_hidden_ud is not None:
                down_resq = torch.matmul(mlp_hidden_ud, down_A.float().T)
                
                # down_resq @ Ua.T ≈ down_orig
                down_restored = torch.matmul(down_resq, Ua.float().T)
                results['13.down_proj'] = compute_metrics(down_restored, down_orig)
                msg, _ = format_metrics(results['13.down_proj'], "13. Down_proj (down_resq @ Ua.T ≈ down_orig)")
                print(f"  {msg}")
            
            # ============================================================
            # Step 14: Final Residual
            # ============================================================
            if 'down_resq' in dir() and 'down_orig' in dir():
                final_orig = residual_orig + down_orig
                final_resq = residual_resq + down_resq
                
                final_restored = torch.matmul(final_resq, Ua.float().T)
                results['14.layer_output'] = compute_metrics(final_restored, final_orig)
                msg, _ = format_metrics(results['14.layer_output'], "14. Layer output (final_resq @ Ua.T ≈ final_orig)")
                print(f"  {msg}")
        
        self.results[f'layer{layer_idx}_activations'] = results
        return results
    
    def verify_ud_transform(self, layer_idx: int = 0) -> float:
        """Verify Ud transformation is orthogonal"""
        Pd = self.mgr.get_pd(layer_idx)
        Hd, K = self.mgr.get_hd()
        if Pd is None:
            return float('nan')
        
        # Create test vector
        intermediate_size = self.mgr.intermediate_size
        blocksize = intermediate_size // K
        x = torch.randn(1, 1, intermediate_size)
        
        # Apply Ud
        x_ud = apply_ud_rotation(x, Pd, Hd, K, blocksize)
        
        # Ud should preserve norm (orthogonal)
        norm_ratio = x_ud.norm() / x.norm()
        results = abs(norm_ratio.item() - 1.0) < 0.01
        self.results['ud_orthogonal'] = results
        return 1.0 if results else 0.0
    
    def run_all(self, tokenizer, prompt: str, num_layers: int = 1):
        """Run all verification tests"""
        print("=" * 60)
        print("ResQ Checkpoint 验证")
        print("=" * 60)
        
        # Embed
        metrics = self.verify_embed()
        msg, _ = format_metrics(metrics, "Embed fusion")
        print(msg)
        
        # Layer-wise verification
        for i in range(num_layers):
            print(f"\n--- Layer {i} ---")
            
            # QKV
            qkv = self.verify_qkv(i, debug=(i == 0))  # Debug for layer 0
            for name, metrics in qkv.items():
                msg, _ = format_metrics(metrics, name)
                print(f"  {msg}")
            
            # O_proj
            metrics = self.verify_o_proj(i)
            msg, _ = format_metrics(metrics, "o_proj")
            print(f"  {msg}")
            
            # MLP
            mlp = self.verify_mlp(i)
            for name, val in mlp.items():
                if isinstance(val, dict) and 'corr' in val:
                    msg, _ = format_metrics(val, name)
                    print(f"  {msg}")
                else:
                    print(f"  ⚠ {name}: {val}")
        
        # Activations
        print(f"\n--- 激活值验证 (Layer 0) ---")
        acts = self.verify_activations(tokenizer, prompt, 0)
        
        # Summary of activation verification
        print(f"\n--- 激活值验证摘要 ---")
        for name, metrics in acts.items():
            if isinstance(metrics, dict) and 'corr' in metrics:
                msg, passed = format_metrics(metrics, name)
                print(f"  {msg}")
            else:
                print(f"  {name}: {metrics}")
        
        # Ud
        ud_ok = self.verify_ud_transform(0)
        status = "✓" if ud_ok > 0.5 else "✗"
        print(f"\n{status} Ud orthogonality check")
        
        print("\n" + "=" * 60)
        print("验证完成")
        print("=" * 60)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ResQ Checkpoint 验证工具')
    parser.add_argument('--original', required=True, help='原始模型路径')
    parser.add_argument('--ckpt-a', required=True, help='ResQ量化权重路径')
    parser.add_argument('--ckpt-b', required=True, help='ResQ中间矩阵路径')
    parser.add_argument('--prompt', default='你好', help='测试 prompt')
    parser.add_argument('--layers', type=int, default=2, help='验证层数')
    args = parser.parse_args()
    
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=True)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("加载 checkpoints...")
    ckpt_a = load_safetensors(args.ckpt_a)
    ckpt_b = load_safetensors(args.ckpt_b)
    
    print(f"ckpt_a keys: {len(ckpt_a)}, ckpt_b keys: {len(ckpt_b)}")
    
    # 反量化诊断
    verify_dequantize_logic(ckpt_a, layer_idx=0)
    
    mgr = CheckpointManager(original_model, ckpt_a, ckpt_b)
    verifier = ResQVerifier(mgr)
    verifier.run_all(tokenizer, args.prompt, args.layers)

if __name__ == '__main__':
    main()
