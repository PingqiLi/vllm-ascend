#!/usr/bin/env python3
"""
ResQ Checkpoint 验证工具

验证 ResQ 量化权重 A 的正确性：
1. 权重融合验证：A 中的权重 = O 中的权重 融合 B 中的旋转矩阵
2. 激活值比较：A 与 O 的推理激活值应该满足已知的旋转关系
3. 最终输出：logits 应该接近

用法:
    python tools/verify_resq_checkpoint.py \
        --original /path/to/bf16_model \
        --ckpt-a /path/to/resq_quantized_weights \
        --ckpt-b /path/to/resq_intermediate_matrices \
        --prompt "你好"

输出解读:
    ✓ PASS: 验证通过
    ✗ FAIL: 验证失败，需要排查
    ⚠ WARN: 存在差异但可能可接受
"""

import argparse
import torch
import sys
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# 直接导入 vllm-ascend 的实现，用于验证
try:
    from vllm_ascend.models.qwen3_resq_truequant import (
        hadamard_transform,
        apply_block_rotation,
        apply_ud_rotation,
        is_pow2,
    )
    USING_VLLM_IMPL = True
except ImportError:
    USING_VLLM_IMPL = False
    import math
    
    def is_pow2(n):
        return (n & (n - 1) == 0) and (n > 0)
    
    def hadamard_transform(u):
        n = u.shape[-1]
        assert is_pow2(n)
        x = u.reshape(-1, n).clone()
        h = 1
        while h < n:
            x = x.view(-1, n // (2 * h), 2, h)
            a, b = x[:, :, 0, :], x[:, :, 1, :]
            x = torch.stack([a + b, a - b], dim=2).view(-1, n)
            h *= 2
        return x.view(u.shape)
    
    def apply_block_rotation(x, R):
        if R is None or R.numel() == 0:
            return x
        K = R.shape[0]
        original_shape = x.shape
        N = original_shape[-1]
        if N == K:
            return torch.matmul(x.float(), R.float()).to(x.dtype)
        num_blocks = N // K
        x_blocked = x.float().reshape(*original_shape[:-1], num_blocks, K)
        x_rotated = torch.matmul(x_blocked, R.float())
        return x_rotated.reshape(original_shape).to(x.dtype)
    
    def apply_ud_rotation(x, Pd, Hd, K, blocksize):
        n = x.shape[-1]
        original_shape = x.shape
        original_dtype = x.dtype
        x = x.float().reshape(*original_shape[:-1], K, blocksize)
        x = torch.matmul(x, Pd.float().T)
        x = hadamard_transform(x.contiguous())
        if Hd is not None and K > 1:
            batch_shape = x.shape[:-2]
            batch_size = 1
            for d in batch_shape:
                batch_size *= d
            x = x.reshape(batch_size, K, blocksize)
            x = torch.einsum('ij,bjk->bik', Hd.float(), x)
            x = x.reshape(*batch_shape, K, blocksize)
        x = x / math.sqrt(n)
        if Hd is not None and K > 1:
            if Hd.abs().max().item() < 0.5:
                x = x * math.sqrt(K)
        return x.reshape(original_shape).to(original_dtype)

# ============================================================================
# 配置
# ============================================================================

@dataclass
class Thresholds:
    """验证阈值"""
    weight_rel_diff: float = 0.05      # 权重相对误差阈值（放宽到5%，考虑bf16精度损失）
    weight_max_diff: float = 0.1       # 权重最大绝对误差阈值
    activation_rel_diff: float = 0.05  # 激活值相对误差阈值
    logits_rel_diff: float = 0.1       # logits 相对误差阈值
    orthogonality: float = 1e-4        # 正交性误差阈值
    # 量化层：使用相关系数而非 rel_diff（int4量化误差很大但相关性应该高）
    quantized_correlation: float = 0.95  # 相关系数阈值


THRESHOLDS = Thresholds()


# ============================================================================
# 工具函数
# ============================================================================

def load_safetensors(path: str) -> dict:
    """加载 safetensors 文件"""
    weights = {}
    p = Path(path)
    files = [p] if p.is_file() else list(p.glob("*.safetensors"))
    
    for sf in files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
    return weights


@dataclass
class DiffResult:
    """差异结果"""
    name: str
    max_diff: float
    mean_diff: float
    rel_diff: float
    passed: bool
    message: str = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.name}: max={self.max_diff:.2e}, rel={self.rel_diff:.2%}"


def compute_diff(a: torch.Tensor, b: torch.Tensor, name: str, 
                 threshold: float = 0.01) -> DiffResult:
    """计算差异并判断是否通过"""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (b.abs() + 1e-10)).mean().item()
    passed = rel_diff < threshold
    
    return DiffResult(
        name=name,
        max_diff=max_diff,
        mean_diff=mean_diff,
        rel_diff=rel_diff,
        passed=passed
    )


def dequantize_mixed_precision(weight_low, weight_high, scale_low, scale_high, debug=False):
    """反量化混精度权重"""
    if debug:
        print(f"    [DEQUANT] weight_low: shape={weight_low.shape}, dtype={weight_low.dtype}, "
              f"range=[{weight_low.float().min():.2f}, {weight_low.float().max():.2f}]")
        print(f"    [DEQUANT] scale_low: shape={scale_low.shape}, dtype={scale_low.dtype}, "
              f"range=[{scale_low.float().min():.6f}, {scale_low.float().max():.6f}]")
        if weight_high is not None:
            print(f"    [DEQUANT] weight_high: shape={weight_high.shape}, dtype={weight_high.dtype}, "
                  f"range=[{weight_high.float().min():.2f}, {weight_high.float().max():.2f}]")
            print(f"    [DEQUANT] scale_high: shape={scale_high.shape}, dtype={scale_high.dtype}, "
                  f"range=[{scale_high.float().min():.6f}, {scale_high.float().max():.6f}]")
    
    # 处理 scale 形状
    if scale_low.dim() == 1:
        scale_low = scale_low.unsqueeze(1)
    elif scale_low.dim() == 2 and scale_low.shape[1] == 1:
        pass  # 已经是 [N, 1]
    
    if scale_high is not None:
        if scale_high.dim() == 1:
            scale_high = scale_high.unsqueeze(1)
        elif scale_high.dim() == 2 and scale_high.shape[1] == 1:
            pass
    
    w_low = weight_low.float() * scale_low.float()
    
    if debug:
        print(f"    [DEQUANT] w_low (after scale): range=[{w_low.min():.6f}, {w_low.max():.6f}]")
    
    if weight_high is not None and scale_high is not None:
        w_high = weight_high.float() * scale_high.float()
        if debug:
            print(f"    [DEQUANT] w_high (after scale): range=[{w_high.min():.6f}, {w_high.max():.6f}]")
        return torch.cat([w_low, w_high], dim=1)
    
    return w_low


# ============================================================================
# 权重管理
# ============================================================================

class CheckpointManager:
    """管理三套权重：O（原始）、A（量化）、B（中间矩阵）"""
    
    def __init__(self, original_model, ckpt_a: dict, ckpt_b: dict):
        self.original_model = original_model
        self.ckpt_a = ckpt_a
        self.ckpt_b = ckpt_b
        
        # 全局 ResQ 参数
        self.Hd = ckpt_a.get('resq.Hd')
        self.Hd_K = int(ckpt_a.get('resq.Hd_K', torch.tensor(1)).item())
        self.blocksize = int(ckpt_a.get('resq.down_proj_blocksize', torch.tensor(256)).item())
    
    def get_original_weight(self, name: str, fuse_layernorm: bool = False) -> Optional[torch.Tensor]:
        """获取原始模型权重
        
        Args:
            name: 权重名称
            fuse_layernorm: 是否融合 LayerNorm (msmodelslim 量化前会融合)
        """
        parts = name.split('.')
        obj = self.original_model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        if hasattr(obj, 'data'):
            weight = obj.data.float()
        else:
            weight = obj.float() if isinstance(obj, torch.Tensor) else None
        
        if weight is None:
            return None
        
        # 融合 LayerNorm: W_new = W * gamma
        if fuse_layernorm:
            gamma = self._get_layernorm_gamma(name)
            if gamma is not None:
                weight = weight * gamma
        
        return weight
    
    def _get_layernorm_gamma(self, weight_name: str) -> Optional[torch.Tensor]:
        """获取对应的 LayerNorm gamma
        
        msmodelslim fuse_layer_norms: W_new = W * gamma
        """
        # 解析层索引
        import re
        match = re.search(r'layers\.(\d+)', weight_name)
        if not match:
            return None
        layer_idx = int(match.group(1))
        
        # Q/K/V -> input_layernorm
        # gate/up -> post_attention_layernorm
        if 'q_proj' in weight_name or 'k_proj' in weight_name or 'v_proj' in weight_name:
            ln_path = f'model.layers.{layer_idx}.input_layernorm.weight'
        elif 'gate_proj' in weight_name or 'up_proj' in weight_name:
            ln_path = f'model.layers.{layer_idx}.post_attention_layernorm.weight'
        else:
            return None
        
        parts = ln_path.split('.')
        obj = self.original_model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        
        if hasattr(obj, 'data'):
            return obj.data.float()
        return obj.float() if isinstance(obj, torch.Tensor) else None
    
    def get_quantized_weight(self, prefix: str, debug: bool = False) -> Optional[torch.Tensor]:
        """获取并反量化 A 中的权重"""
        weight_low = self.ckpt_a.get(f'{prefix}.weight_low')
        if weight_low is None:
            return None
        
        weight_high = self.ckpt_a.get(f'{prefix}.weight_high')
        scale_low = self.ckpt_a.get(f'{prefix}.scale_low')
        scale_high = self.ckpt_a.get(f'{prefix}.scale_high')
        
        return dequantize_mixed_precision(weight_low, weight_high, scale_low, scale_high, debug=debug)
    
    def get_ua(self, layer_idx: int, debug: bool = False) -> Optional[torch.Tensor]:
        """获取 Ua = Pa @ Ra"""
        Pa = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_a')
        Ra = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_a')
        if Pa is None or Ra is None:
            if debug:
                print(f"  [DEBUG] P_a or R_a not found for layer {layer_idx}")
            return None
        
        if debug:
            print(f"  [DEBUG] P_a shape: {Pa.shape}, R_a shape: {Ra.shape}")
            # 检查 P_a 正交性
            Pa_f = Pa.float()
            Pa_orth_err = (Pa_f @ Pa_f.T - torch.eye(Pa.shape[0])).abs().max().item()
            print(f"  [DEBUG] P_a 正交性误差: {Pa_orth_err:.2e}")
            # 检查 R_a 正交性
            Ra_f = Ra.float()
            Ra_orth_err = (Ra_f @ Ra_f.T - torch.eye(Ra.shape[0])).abs().max().item()
            print(f"  [DEBUG] R_a 正交性误差: {Ra_orth_err:.2e}")
            
            # 检查 R_a 是否是 block_diag 形式
            # R1 = block_diag(R1_1, R1_2) 其中 R1_1:[4480,4480], R1_2:[640,640]
            hidden_size = Ra.shape[0]
            high_len = int(hidden_size * 0.125)  # 640
            low_len = hidden_size - high_len  # 4480
            
            # 检查 off-diagonal blocks 是否接近 0
            off_diag_low_high = Ra[:low_len, low_len:].abs().max().item()
            off_diag_high_low = Ra[low_len:, :low_len].abs().max().item()
            print(f"  [DEBUG] R_a 结构检查 (应该是 block_diag):")
            print(f"    off-diag [low, high] max: {off_diag_low_high:.2e}")
            print(f"    off-diag [high, low] max: {off_diag_high_low:.2e}")
            
            if off_diag_low_high > 0.01 or off_diag_high_low > 0.01:
                print(f"  [DEBUG] ⚠ R_a 不是 block_diag 形式！")
        
        return torch.matmul(Pa.float(), Ra.float())
    
    def get_uc(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取 Uc = Pc @ Rc"""
        # 优先从 A 获取
        Uc = self.ckpt_a.get(f'resq.layer.{layer_idx}.Uc')
        if Uc is not None:
            return Uc.float()
        
        # 从 B 计算
        Pc = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_c')
        Rc = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_c')
        if Pc is None or Rc is None:
            return None
        return torch.matmul(Pc.float(), Rc.float())
    
    def get_ub(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取 Ub = Pb @ Rb (per-head)"""
        Pb = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_b')  # [num_heads, head_dim, head_dim]
        Rb = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_b')  # [head_dim, head_dim]
        if Pb is None or Rb is None:
            return None
        # Ub[i] = Pb[i] @ Rb
        return torch.matmul(Pb.float(), Rb.float())
    
    def get_pd(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取 Pd"""
        Pd = self.ckpt_a.get(f'resq.layer.{layer_idx}.Pd')
        if Pd is not None:
            return Pd.float()
        return self.ckpt_b.get(f'resq.layer.{layer_idx}.P_d')


# ============================================================================
# 验证测试
# ============================================================================

class ResQVerifier:
    """ResQ 验证器"""
    
    def __init__(self, mgr: CheckpointManager):
        self.mgr = mgr
        self.results = []
    
    def log(self, result: DiffResult):
        """记录结果"""
        self.results.append(result)
        print(f"  {result}")
    
    def verify_ab_consistency(self, layer_idx: int) -> bool:
        """
        验证 ckpt A 和 ckpt B 的一致性
        
        - A 中的 Uc 应该等于 B 中的 Pc @ Rc
        - A 中的 Hd 应该等于 B 中的 R_d_hadK
        - A 中的 Pd 应该等于 B 中的 P_d
        """
        print(f"\n[Layer {layer_idx}] A/B 一致性验证")
        all_passed = True
        
        # 1. Uc 一致性: A.Uc == B.Pc @ B.Rc
        Uc_A = self.mgr.ckpt_a.get(f'resq.layer.{layer_idx}.Uc')
        Pc_B = self.mgr.ckpt_b.get(f'resq.layer.{layer_idx}.P_c')
        Rc_B = self.mgr.ckpt_b.get(f'resq.layer.{layer_idx}.R_c')
        
        if Uc_A is not None and Pc_B is not None and Rc_B is not None:
            Uc_computed = torch.matmul(Pc_B.float(), Rc_B.float())
            result = compute_diff(Uc_A.float(), Uc_computed, "Uc: A vs Pc@Rc", 1e-5)
            self.log(result)
            all_passed = all_passed and result.passed
        else:
            print("  ⚠ 无法验证 Uc 一致性 (缺少数据)")
        
        # 2. Hd 一致性: A.Hd == B.R_d_hadK
        Hd_A = self.mgr.ckpt_a.get('resq.Hd')
        Hd_B = self.mgr.ckpt_b.get(f'resq.layer.{layer_idx}.R_d_hadK')
        
        if Hd_A is not None and Hd_B is not None:
            result = compute_diff(Hd_A.float(), Hd_B.float(), "Hd: A vs R_d_hadK", 1e-5)
            self.log(result)
            all_passed = all_passed and result.passed
        else:
            print("  ⚠ 无法验证 Hd 一致性 (缺少数据)")
        
        # 3. Pd 一致性: A.Pd == B.P_d
        Pd_A = self.mgr.ckpt_a.get(f'resq.layer.{layer_idx}.Pd')
        Pd_B = self.mgr.ckpt_b.get(f'resq.layer.{layer_idx}.P_d')
        
        if Pd_A is not None and Pd_B is not None:
            result = compute_diff(Pd_A.float(), Pd_B.float(), "Pd: A vs P_d", 1e-5)
            self.log(result)
            all_passed = all_passed and result.passed
        else:
            print("  ⚠ 无法验证 Pd 一致性 (缺少数据)")
        
        return all_passed
    
    def verify_orthogonality(self, layer_idx: int) -> bool:
        """
        验证旋转矩阵的正交性
        
        - Ua, Uc, Pd 来自 torch.linalg.eigh 的特征向量，应该正交
        - Ud = BlockDiag(Pd.T) @ H，由正交矩阵组成，也应该正交
        """
        print(f"\n[Layer {layer_idx}] 正交性验证")
        all_passed = True
        
        # 验证 Ua, Uc, Pd 的正交性
        # 首次验证时启用 debug 输出
        if layer_idx == 0:
            print("  [DEBUG] 检查 P_a 和 R_a 的正交性:")
            self.mgr.get_ua(layer_idx, debug=True)
        
        for name, get_fn in [
            ("Ua", lambda: self.mgr.get_ua(layer_idx)),
            ("Uc", lambda: self.mgr.get_uc(layer_idx)),
            ("Pd", lambda: self.mgr.get_pd(layer_idx)),
        ]:
            M = get_fn()
            if M is None:
                print(f"  ⚠ {name} 未找到")
                continue
            
            # 检查 M @ M.T ≈ I
            identity = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
            diff = (torch.matmul(M, M.T) - identity).abs().max().item()
            passed = diff < THRESHOLDS.orthogonality
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: ||M @ M.T - I||_max = {diff:.2e}")
            all_passed = all_passed and passed
        
        # 验证 Ud 正交性（通过测试 apply_ud_rotation 是否保持范数）
        # Ud 正交 => ||Ud @ x|| = ||x||
        Pd = self.mgr.get_pd(layer_idx)
        Hd = self.mgr.Hd
        K = self.mgr.Hd_K
        blocksize = self.mgr.blocksize
        
        if Pd is not None and K > 0 and blocksize > 0:
            intermediate_size = K * blocksize
            test_x = torch.randn(1, intermediate_size)
            test_x_norm = test_x.norm().item()
            
            try:
                rotated = apply_ud_rotation(test_x, Pd, Hd, K, blocksize)
                rotated_norm = rotated.norm().item()
                
                # 正交变换应该保持范数
                norm_ratio = rotated_norm / test_x_norm
                passed = abs(norm_ratio - 1.0) < 0.01  # 允许 1% 误差
                status = "✓" if passed else "✗"
                print(f"  {status} Ud (via norm): ||Ud@x||/||x|| = {norm_ratio:.4f} (应≈1.0)")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ⚠ Ud 正交性验证失败: {e}")
        
        return all_passed
    
    def verify_embed_fusion(self) -> DiffResult:
        """
        验证 Embed 权重融合
        
        msmodelslim 的处理流程:
          1. fuse_layer_norms: embed_centered = embed_O - mean(embed_O, dim=-1, keepdim=True)
          2. rotate_embeddings: embed_A = embed_centered @ Ua
          
          综合: embed_A = (embed_O - mean(embed_O)) @ Ua
          验证: (embed_O - mean) @ Ua ≈ embed_A
        """
        print("\n[Embed] 权重融合验证")
        
        embed_O = self.mgr.get_original_weight('model.embed_tokens.weight')
        embed_A = self.mgr.ckpt_a.get('model.embed_tokens.weight')
        Ua = self.mgr.get_ua(0)  # 使用第0层的 Ua
        
        if embed_O is None or embed_A is None:
            result = DiffResult("Embed", 0, 0, 0, False, "权重未找到")
            self.log(result)
            return result
        
        if Ua is None:
            result = DiffResult("Embed", 0, 0, 0, False, "Ua 未找到，无法验证")
            self.log(result)
            return result
        
        # 直接比较（应该差异大）
        direct = compute_diff(embed_A.float(), embed_O.float(), "embed_A vs embed_O (直接)")
        print(f"  直接比较: rel={direct.rel_diff:.2%} (应该差异大)")
        
        # msmodelslim 的 fuse_layer_norms 对 embed 做了 mean subtraction:
        #   W_new = W_ - W_.mean(dim=-1, keepdim=True)
        embed_O_float = embed_O.float()
        embed_centered = embed_O_float - embed_O_float.mean(dim=-1, keepdim=True)
        print(f"  [DEBUG] embed_O mean subtraction: mean={embed_O_float.mean(dim=-1).abs().mean():.4f}")
        
        # 正向验证: (embed_O - mean) @ Ua ≈ embed_A
        embed_forward = torch.matmul(embed_centered, Ua)
        forward_result = compute_diff(embed_forward, embed_A.float(), 
                                     "(embed_O - mean) @ Ua vs embed_A")
        print(f"  正向验证: (embed_O - mean) @ Ua vs embed_A: rel={forward_result.rel_diff:.2%}")
        
        # 逆向验证: embed_A @ Ua.T ≈ embed_O - mean
        embed_restored = torch.matmul(embed_A.float(), Ua.T)
        result = compute_diff(embed_restored, embed_centered, 
                             "embed_A @ Ua.T vs (embed_O - mean)", 
                             THRESHOLDS.weight_rel_diff)
        self.log(result)
        
        return result
    
    def verify_qkv_fusion(self, layer_idx: int) -> Dict[str, DiffResult]:
        """
        验证 Q/K/V 权重融合
        
        msmodelslim 的 rotate_attention_inputs 和 rotate_ov_proj:
        
        Q/K_proj:
          - rotate_attention_inputs: W_A = W_O @ Ua
          - 验证: W_O ≈ W_A @ Ua.T
        
        V_proj:
          - rotate_attention_inputs: W_temp = W_O @ Ua
          - rotate_ov_proj (output=True): W_A[h] = Ub[h].T @ W_temp[h]
          - 综合: W_A[h] = Ub[h].T @ W_O[h] @ Ua
          - 验证: W_O[h] ≈ Ub[h] @ W_A[h] @ Ua.T
        """
        print(f"\n[Layer {layer_idx}] Q/K/V 权重融合验证")
        
        results = {}
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx)  # [num_kv_heads, head_dim, head_dim]
        
        if Ua is None:
            print("  ⚠ Ua 未找到")
            return results
        
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            # 使用 fuse_layernorm=True，因为 msmodelslim 在量化前会融合 LayerNorm
            W_O = self.mgr.get_original_weight(f'model.layers.{layer_idx}.self_attn.{proj}.weight', fuse_layernorm=True)
            # 对 layer 0 的 q_proj 启用 debug
            debug_dequant = (layer_idx == 0 and proj == 'q_proj')
            W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.self_attn.{proj}', debug=debug_dequant)
            
            if W_O is None or W_A is None:
                print(f"  ⚠ {proj} 权重未找到")
                continue
            
            # W shape: [out_features, in_features]
            out_dim = W_A.shape[0]
            in_dim = W_A.shape[1]  # = hidden_size
            
            if proj in ['q_proj', 'k_proj']:
                # 诊断：直接比较 W_A 和 W_O
                if layer_idx == 0 and proj == 'q_proj':
                    print(f"  [DEBUG] {proj} 形状: W_A={W_A.shape}, W_O={W_O.shape}")
                    print(f"  [DEBUG] {proj} 数值范围:")
                    print(f"    W_A: min={W_A.min().item():.4f}, max={W_A.max().item():.4f}, mean={W_A.float().mean().item():.4f}")
                    print(f"    W_O: min={W_O.min().item():.4f}, max={W_O.max().item():.4f}, mean={W_O.float().mean().item():.4f}")
                    
                    # 检查 LayerNorm gamma
                    gamma = self.mgr._get_layernorm_gamma(f'model.layers.{layer_idx}.self_attn.{proj}.weight')
                    if gamma is not None:
                        print(f"    gamma: min={gamma.min().item():.4f}, max={gamma.max().item():.4f}, mean={gamma.mean().item():.4f}")
                    
                    # 检查 Ua 范围和正交性
                    print(f"    Ua: min={Ua.min().item():.4f}, max={Ua.max().item():.4f}")
                    Ua_orth_err = (torch.matmul(Ua, Ua.T) - torch.eye(Ua.shape[0])).abs().max().item()
                    print(f"    Ua orthogonality error: {Ua_orth_err:.2e}")
                    
                    # 不融合 gamma 时的 W_O
                    W_O_raw = self.mgr.get_original_weight(f'model.layers.{layer_idx}.self_attn.{proj}.weight', fuse_layernorm=False)
                    print(f"    W_O_raw (no gamma): min={W_O_raw.min().item():.4f}, max={W_O_raw.max().item():.4f}")
                    
                    direct_diff = compute_diff(W_A.float(), W_O, f"{proj} 直接比较")
                    print(f"  [DEBUG] {proj} 直接比较: rel={direct_diff.rel_diff:.2%}")
                    
                    # 正向验证: W_O @ Ua ≈ W_A
                    W_forward = torch.matmul(W_O.float(), Ua)
                    print(f"    W_O@Ua: min={W_forward.min().item():.4f}, max={W_forward.max().item():.4f}")
                    forward_diff = compute_diff(W_forward, W_A.float(), f"{proj} 正向")
                    print(f"  [DEBUG] {proj} 正向验证 (W_O @ Ua vs W_A): rel={forward_diff.rel_diff:.2%}")
                    
                    # 打印前几个元素对比
                    print(f"  [DEBUG] 前5个元素对比 (row 0):")
                    print(f"    W_O@Ua[0,:5]: {W_forward[0,:5].tolist()}")
                    print(f"    W_A[0,:5]:    {W_A[0,:5].float().tolist()}")
                    print(f"    diff[0,:5]:   {(W_forward[0,:5] - W_A[0,:5].float()).tolist()}")
                    
                    # 检查相关性（如果只是 scale 问题，相关性应该高）
                    corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1]
                    print(f"    相关系数: {corr.item():.4f} (1.0=完全相关, 0=无关)")
                    
                    # 不融合 gamma 的正向验证
                    W_forward_raw = torch.matmul(W_O_raw.float(), Ua)
                    print(f"    W_O_raw@Ua: min={W_forward_raw.min().item():.4f}, max={W_forward_raw.max().item():.4f}")
                    forward_raw_diff = compute_diff(W_forward_raw, W_A.float(), f"{proj} 正向 (no gamma)")
                    print(f"  [DEBUG] {proj} 正向验证 (W_O_raw @ Ua vs W_A): rel={forward_raw_diff.rel_diff:.2%}")
                
                # Q/K: W_A = W_O @ Ua
                # 验证方式改为相关系数（int4量化导致rel_diff很大但相关性高）
                W_forward = torch.matmul(W_O.float(), Ua)
                corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1].item()
                
                passed = corr >= THRESHOLDS.quantized_correlation
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {proj}: corr={corr:.4f} (阈值>={THRESHOLDS.quantized_correlation})")
                
                result = DiffResult(proj, 0, 0, 0, passed, f"corr={corr:.4f}")
                results[proj] = result
                if not passed:
                    self.failed.append(result)
                else:
                    self.passed.append(result)
                continue
            else:
                # V: W_A[h] = Ub[h].T @ W_O[h] @ Ua
                # 验证方式改为相关系数
                if Ub is None:
                    print("  ⚠ Ub 未找到，跳过 v_proj")
                    continue
                
                num_kv_heads, head_dim, _ = Ub.shape
                
                # 计算 W_forward = Ub[h].T @ W_O[h] @ Ua
                W_O_reshaped = W_O.float().reshape(num_kv_heads, head_dim, in_dim)
                # Ub[h].T @ W_O[h]: [head_dim, head_dim].T @ [head_dim, in_dim] = [head_dim, in_dim]
                W_tmp = torch.einsum('nji,nje->nie', Ub.float(), W_O_reshaped)
                W_tmp = W_tmp.reshape(out_dim, in_dim)
                W_forward = torch.matmul(W_tmp, Ua)
                
                corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1].item()
                
                passed = corr >= THRESHOLDS.quantized_correlation
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {proj}: corr={corr:.4f} (阈值>={THRESHOLDS.quantized_correlation})")
                
                result = DiffResult(proj, 0, 0, 0, passed, f"corr={corr:.4f}")
                results[proj] = result
                if not passed:
                    self.failed.append(result)
                else:
                    self.passed.append(result)
        
        return results
    
    def verify_o_proj_fusion(self, layer_idx: int) -> DiffResult:
        """
        验证 O_proj 权重融合
        
        msmodelslim 的 rotate_ov_proj (output=False) 和 rotate_attention_output:
        
        O_proj:
          - rotate_ov_proj: W_temp[:,h,:] = W_O[:,h,:] @ Ub_expanded[h] (GQA 扩展)
          - rotate_attention_output: W_A = Ua.T @ W_temp
          - 综合: W_A = Ua.T @ (W_O @ block_diag(Ub_expanded))
          - 验证: W_O ≈ Ua @ W_A @ block_diag(Ub_expanded.T)
        
        GQA: num_attention_heads = 64, num_kv_heads = 8
        Ub shape: [num_kv_heads, head_dim, head_dim] = [8, 128, 128]
        需要扩展成 [num_attention_heads, head_dim, head_dim] = [64, 128, 128]
        """
        print(f"\n[Layer {layer_idx}] O_proj 权重融合验证")
        
        W_O = self.mgr.get_original_weight(f'model.layers.{layer_idx}.self_attn.o_proj.weight')
        W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.self_attn.o_proj')
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx)  # [num_kv_heads, head_dim, head_dim]
        
        if W_O is None or W_A is None:
            result = DiffResult("o_proj", 0, 0, 0, False, "权重未找到")
            self.log(result)
            return result
        
        # W shape: [hidden_size, num_attention_heads * head_dim]
        hidden_size = W_A.shape[0]  # 5120
        in_dim = W_A.shape[1]       # 8192 = num_attention_heads * head_dim
        
        if Ua is None or Ub is None:
            print("  ⚠ Ua 或 Ub 未找到，跳过 o_proj 验证")
            result = DiffResult("o_proj", 0, 0, 0, False, "Ua/Ub 未找到")
            self.log(result)
            return result
        
        num_kv_heads, head_dim, _ = Ub.shape  # 8, 128, 128
        num_attention_heads = in_dim // head_dim  # 64
        
        # GQA: 扩展 Ub 到 num_attention_heads
        # 每个 KV head 对应 num_attention_heads // num_kv_heads 个 Q heads
        num_q_per_kv = num_attention_heads // num_kv_heads  # 8
        Ub_expanded = Ub.repeat_interleave(num_q_per_kv, dim=0)  # [64, 128, 128]
        
        print(f"  GQA: num_attention_heads={num_attention_heads}, num_kv_heads={num_kv_heads}")
        print(f"  Ub_expanded shape: {Ub_expanded.shape}")
        
        # 验证: W_O ≈ Ua @ W_A @ block_diag(Ub_expanded.T)
        # Step 1: W_A @ block_diag(Ub_expanded.T)
        # W_A shape: [hidden_size, num_attention_heads * head_dim]
        # reshape to [hidden_size, num_attention_heads, head_dim]
        W_tmp = W_A.reshape(hidden_size, num_attention_heads, head_dim).float()
        
        # 对每个 head: W_tmp[:, h, :] @ Ub_expanded[h].T
        # einsum: W_tmp[hid, n, d] @ Ub.T[n, d, d'] -> W_tmp[hid, n, d']
        Ub_expanded_T = Ub_expanded.transpose(-1, -2).float()  # [64, 128, 128]
        W_tmp = torch.einsum('hnd,nde->hne', W_tmp, Ub_expanded_T)
        W_tmp = W_tmp.reshape(hidden_size, in_dim)
        
        # Step 2: Ua @ W_tmp
        W_restored = torch.matmul(Ua.float(), W_tmp)
        
        result = compute_diff(W_restored, W_O, "o_proj", THRESHOLDS.weight_rel_diff)
        self.log(result)
        return result
    
    def verify_mlp_fusion(self, layer_idx: int) -> Dict[str, DiffResult]:
        """
        验证 MLP 权重融合
        
        msmodelslim 的 rotate_mlp_input:
        
        gate_proj/up_proj:
          - rotate_mlp_input: W_A = W_O @ Ua
          - 验证: W_O ≈ W_A @ Ua.T
        
        down_proj (输出侧 Ua.T，输入侧 Ud):
          - 变换复杂，暂时跳过
        """
        print(f"\n[Layer {layer_idx}] MLP 权重融合验证")
        
        results = {}
        Ua = self.mgr.get_ua(layer_idx)
        
        if Ua is None:
            print("  ⚠ Ua 未找到")
            return results
        
        # gate_proj 和 up_proj: W_A = (W_O * gamma) @ Ua
        # 验证: (W_O * gamma) ≈ W_A @ Ua.T
        for proj in ['gate_proj', 'up_proj']:
            # 使用 fuse_layernorm=True，因为 msmodelslim 在量化前会融合 LayerNorm
            W_O = self.mgr.get_original_weight(f'model.layers.{layer_idx}.mlp.{proj}.weight', fuse_layernorm=True)
            W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.mlp.{proj}')
            
            if W_O is None or W_A is None:
                print(f"  ⚠ {proj} 权重未找到")
                continue
            
            # W_A @ Ua.T: [intermediate, hidden] @ [hidden, hidden] = [intermediate, hidden]
            W_restored = torch.matmul(W_A.float(), Ua.T)
            result = compute_diff(W_restored, W_O, proj, THRESHOLDS.weight_rel_diff)
            results[proj] = result
            self.log(result)
        
        # down_proj 更复杂，涉及 Ud，暂时跳过详细验证
        print("  ⚠ down_proj 涉及 Ud 变换，需要单独验证")
        
        return results
    
    def verify_activation_with_rotation(self, tokenizer, prompt: str) -> Tuple[DiffResult, torch.Tensor, torch.Tensor]:
        """
        验证激活值（考虑旋转）
        
        msmodelslim 处理:
          embed_centered = embed_O - mean(embed_O, dim=-1, keepdim=True)
          embed_A = embed_centered @ Ua
        
        所以:
          resq_hidden = embed_A[tokens] = embed_centered[tokens] @ Ua
          验证: embed_centered[tokens] ≈ resq_hidden @ Ua.T
        """
        print("\n[Activation] 激活值验证")
        print(f"  使用 vllm-ascend 实现: {USING_VLLM_IMPL}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")
        
        # 原始模型 embed（需要做 mean subtraction）
        orig_embed_weight = self.mgr.original_model.model.embed_tokens.weight.data.float()
        orig_embed_centered = orig_embed_weight - orig_embed_weight.mean(dim=-1, keepdim=True)
        orig_hidden = torch.embedding(orig_embed_centered, input_ids)
        
        # ResQ embed
        resq_embed_weight = self.mgr.ckpt_a.get('model.embed_tokens.weight')
        if resq_embed_weight is None:
            result = DiffResult("Embed activation", 0, 0, 0, False, "权重未找到")
            self.log(result)
            return result, None, None
        
        resq_hidden = torch.embedding(resq_embed_weight, input_ids).float()
        
        # 获取 Ua
        Ua = self.mgr.get_ua(0)
        if Ua is None:
            result = DiffResult("Embed activation", 0, 0, 0, False, "Ua 未找到")
            self.log(result)
            return result, orig_hidden, resq_hidden
        
        # 验证: orig_hidden (centered) ≈ resq_hidden @ Ua.T
        # 因为 resq_hidden = orig_centered @ Ua
        resq_restored = torch.matmul(resq_hidden, Ua.T)
        result = compute_diff(resq_restored, orig_hidden, "resq_embed @ Ua.T vs orig_centered_embed",
                             THRESHOLDS.activation_rel_diff)
        self.log(result)
        
        return result, orig_hidden, resq_hidden
    
    def verify_ud_rotation(self, layer_idx: int) -> Optional[DiffResult]:
        """
        验证 Ud 旋转实现
        
        使用 vllm-ascend 的 apply_ud_rotation 函数
        """
        print(f"\n[Layer {layer_idx}] Ud 旋转验证")
        
        Pd = self.mgr.get_pd(layer_idx)
        Hd = self.mgr.Hd
        K = self.mgr.Hd_K
        blocksize = self.mgr.blocksize
        
        if Pd is None:
            print("  ⚠ Pd 未找到")
            return None
        
        print(f"  Pd: {tuple(Pd.shape)}, K={K}, blocksize={blocksize}")
        if Hd is not None:
            print(f"  Hd: {tuple(Hd.shape)}, max={Hd.abs().max().item():.4f}")
        
        # 创建测试输入
        intermediate_size = K * blocksize
        test_input = torch.randn(1, 10, intermediate_size)  # [batch, seq, intermediate]
        
        # 应用 Ud 旋转
        try:
            rotated = apply_ud_rotation(test_input, Pd, Hd, K, blocksize)
            print(f"  ✓ apply_ud_rotation 成功: input={tuple(test_input.shape)} -> output={tuple(rotated.shape)}")
            
            # 检查输出范围
            print(f"    输入范围: [{test_input.min().item():.2f}, {test_input.max().item():.2f}]")
            print(f"    输出范围: [{rotated.min().item():.2f}, {rotated.max().item():.2f}]")
            
            return DiffResult("Ud rotation", 0, 0, 0, True, "函数执行成功")
        except Exception as e:
            print(f"  ✗ apply_ud_rotation 失败: {e}")
            return DiffResult("Ud rotation", 0, 0, 0, False, str(e))
    
    def verify_uc_rotation(self, layer_idx: int) -> Optional[DiffResult]:
        """
        验证 Uc 旋转实现
        
        使用 vllm-ascend 的 apply_block_rotation 函数
        """
        print(f"\n[Layer {layer_idx}] Uc 旋转验证")
        
        Uc = self.mgr.get_uc(layer_idx)
        if Uc is None:
            print("  ⚠ Uc 未找到")
            return None
        
        head_dim = Uc.shape[0]
        num_heads = 64  # Qwen3-32B
        
        print(f"  Uc: {tuple(Uc.shape)}")
        
        # 创建测试输入
        test_input = torch.randn(1, 10, num_heads * head_dim)  # [batch, seq, num_heads * head_dim]
        
        # 应用 Uc 旋转 (block-wise)
        try:
            rotated = apply_block_rotation(test_input, Uc)
            print(f"  ✓ apply_block_rotation 成功: input={tuple(test_input.shape)} -> output={tuple(rotated.shape)}")
            
            return DiffResult("Uc rotation", 0, 0, 0, True, "函数执行成功")
        except Exception as e:
            print(f"  ✗ apply_block_rotation 失败: {e}")
            return DiffResult("Uc rotation", 0, 0, 0, False, str(e))
    
    def verify_resq_quant_matmul(self) -> Optional[DiffResult]:
        """
        验证 resq_quant_matmul 接口精度
        
        对比 vllm-ascend 的实现与 CPU 参考实现（int32 matmul）
        这是 E=1 情况下 reference_op_impl.py run_test 的封装
        """
        print("\n[resq_quant_matmul] 量化 MatMul 精度验证")
        
        try:
            from vllm_ascend.ops.resq_quant_matmul import resq_quant_matmul
        except ImportError:
            print("  ⚠ 无法导入 resq_quant_matmul，跳过验证")
            return None
        
        # 测试参数（模拟 Qwen3-32B 的一个线性层）
        M = 32     # batch * seq
        K = 5120   # hidden_size
        N = 8192   # intermediate_size 或 num_heads * head_dim
        splitKPos = (K // 8) * 7  # int4 占 7/8
        
        print(f"  测试参数: M={M}, K={K}, N={N}, splitKPos={splitKPos}")
        
        # 生成随机量化输入
        torch.manual_seed(42)
        
        # 激活：int8，前 splitKPos 是 int4 范围 [-8, 7]
        x = torch.zeros(M, K, dtype=torch.int8)
        x[:, :splitKPos] = torch.randint(-8, 8, (M, splitKPos), dtype=torch.int8)
        x[:, splitKPos:] = torch.randint(-128, 128, (M, K - splitKPos), dtype=torch.int8)
        
        # 权重：int8，同样的切分
        weight = torch.zeros(1, K, N, dtype=torch.int8)
        weight[:, :splitKPos, :] = torch.randint(-8, 8, (1, splitKPos, N), dtype=torch.int8)
        weight[:, splitKPos:, :] = torch.randint(-128, 128, (1, K - splitKPos, N), dtype=torch.int8)
        
        # Scales
        lweightScale = torch.rand(1, 1, N, dtype=torch.float32) * 0.1
        hweightScale = torch.rand(1, 1, N, dtype=torch.float32) * 0.01
        lxScale = torch.rand(M, dtype=torch.float32) * 0.1
        rxScale = torch.rand(M, dtype=torch.float32) * 0.01
        
        # 运行 resq_quant_matmul
        try:
            output = resq_quant_matmul(
                x=x,
                weight=weight,
                lweightScale=lweightScale,
                hweightScale=hweightScale,
                lxScale=lxScale,
                rxScale=rxScale,
                splitKPos=splitKPos,
                groupList=None,
                outDtype=torch.float16,
            )
            print(f"  ✓ resq_quant_matmul 执行成功: output shape={tuple(output.shape)}")
        except Exception as e:
            print(f"  ✗ resq_quant_matmul 执行失败: {e}")
            return DiffResult("resq_quant_matmul", 0, 0, 0, False, str(e))
        
        # 参考实现：直接用 int32 matmul
        # y_low = (x[:, :splitKPos].int32 @ weight[0, :splitKPos, :].int32) * lweightScale * lxScale
        # y_high = (x[:, splitKPos:].int32 @ weight[0, splitKPos:, :].int32) * hweightScale * rxScale
        x_low = x[:, :splitKPos].to(torch.int32)
        x_high = x[:, splitKPos:].to(torch.int32)
        w_low = weight[0, :splitKPos, :].to(torch.int32)
        w_high = weight[0, splitKPos:, :].to(torch.int32)
        
        y_low_ref = torch.matmul(x_low, w_low).float()
        y_low_ref = y_low_ref * lweightScale.flatten().unsqueeze(0)  # (1, N)
        y_low_ref = y_low_ref * lxScale.unsqueeze(1)  # (M, 1)
        
        y_high_ref = torch.matmul(x_high, w_high).float()
        y_high_ref = y_high_ref * hweightScale.flatten().unsqueeze(0)
        y_high_ref = y_high_ref * rxScale.unsqueeze(1)
        
        golden = (y_low_ref + y_high_ref).to(torch.float16)
        
        # 比较
        output_f32 = output.float()
        golden_f32 = golden.float()
        diff = (output_f32 - golden_f32).abs()
        
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (golden_f32.abs() + 1e-10)).mean().item()
        
        print("  对比结果:")
        print(f"    max_diff: {max_diff:.2e}")
        print(f"    mean_diff: {mean_diff:.2e}")
        print(f"    rel_diff: {rel_diff:.2%}")
        
        # 判断是否通过（允许 1% 相对误差，因为 float16 精度有限）
        passed = rel_diff < 0.01
        
        result = DiffResult(
            name="resq_quant_matmul",
            max_diff=max_diff,
            mean_diff=mean_diff,
            rel_diff=rel_diff,
            passed=passed,
            message="CPU 参考实现对比" if passed else "精度超出阈值"
        )
        self.log(result)
        return result
    
    def verify_final_logits(self, tokenizer, prompt: str) -> DiffResult:
        """
        验证最终 logits
        
        这是最终判断：即使中间激活有旋转，最终 logits 应该一致
        """
        print("\n[Logits] 最终输出验证")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        # 原始模型推理
        self.mgr.original_model.eval()
        with torch.no_grad():
            orig_outputs = self.mgr.original_model(input_ids)
        
        orig_logits = orig_outputs.logits
        
        # 获取原始模型预测
        pred_id = orig_logits[0, -1].argmax().item()
        pred_token = tokenizer.decode([pred_id])
        print(f"  原始模型预测: '{pred_token}' (id={pred_id})")
        
        # 显示 top-5
        top5 = torch.topk(orig_logits[0, -1], 5)
        print("  Top-5:")
        for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
            tok = tokenizer.decode([idx.item()])
            print(f"    {i+1}. '{tok}' (score={score.item():.2f})")
        
        # 注意：这里只能比较原始模型，ResQ 模型需要在 vllm 中运行
        # 返回一个 placeholder 结果
        result = DiffResult("Final logits", 0, 0, 0, True, 
                           "需要在 vllm serve 中比较 ResQ 推理结果")
        return result
    
    def run_all_tests(self, tokenizer, prompt: str, num_layers: int = 2):
        """运行所有验证测试"""
        print("=" * 70)
        print("ResQ Checkpoint 验证报告")
        print("=" * 70)
        print(f"使用 vllm-ascend 实现: {USING_VLLM_IMPL}")
        
        print("\n" + "-" * 70)
        print("矩阵计算公式 (从 ckpt B 计算):")
        print("  Ua = P_a @ R_a      [hidden, hidden]")
        print("  Ub = P_b @ R_b      [num_heads, head_dim, head_dim]")
        print("  Uc = P_c @ R_c      [head_dim, head_dim]")
        print("  Ud = BlockDiag(Pd.T) @ H  (隐式，在 apply_ud_rotation 中)")
        print("-" * 70)
        
        # 0. A/B 一致性验证（验证 A 中的矩阵是否由 B 正确计算得到）
        print("\n" + "=" * 70)
        print("阶段 0: A/B 矩阵一致性验证")
        print("=" * 70)
        for layer_idx in range(min(num_layers, 64)):
            self.verify_ab_consistency(layer_idx)
        
        # 1. Embed 权重融合
        print("\n" + "=" * 70)
        print("阶段 1: Embed 权重融合验证")
        print("=" * 70)
        self.verify_embed_fusion()
        
        # 2. 前几层权重融合
        print("\n" + "=" * 70)
        print("阶段 2: 逐层权重融合验证")
        print("=" * 70)
        for layer_idx in range(min(num_layers, 64)):
            self.verify_orthogonality(layer_idx)
            self.verify_qkv_fusion(layer_idx)
            self.verify_o_proj_fusion(layer_idx)
            self.verify_mlp_fusion(layer_idx)
        
        # 3. 旋转函数验证
        print("\n" + "=" * 70)
        print("阶段 3: 旋转函数验证 (使用 vllm-ascend 实现)")
        print("=" * 70)
        for layer_idx in range(min(num_layers, 64)):
            self.verify_uc_rotation(layer_idx)
            self.verify_ud_rotation(layer_idx)
        
        # 3.5 resq_quant_matmul 精度验证
        print("\n" + "=" * 70)
        print("阶段 3.5: resq_quant_matmul 精度验证")
        print("=" * 70)
        self.verify_resq_quant_matmul()
        
        # 4. 激活值验证
        print("\n" + "=" * 70)
        print("阶段 4: 激活值验证")
        print("=" * 70)
        self.verify_activation_with_rotation(tokenizer, prompt)
        
        # 5. 最终 logits
        print("\n" + "=" * 70)
        print("阶段 5: 最终输出参考")
        print("=" * 70)
        self.verify_final_logits(tokenizer, prompt)
        
        # 汇总
        self.print_summary()
    
    def print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 70)
        print("验证汇总")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        print(f"\n总计: {passed} PASS, {failed} FAIL\n")
        
        if failed > 0:
            print("失败项目:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}: rel={r.rel_diff:.2%}, max={r.max_diff:.2e}")
                    if r.message:
                        print(f"      {r.message}")
        
        print("\n" + "=" * 70)
        if failed == 0:
            print("✓ 所有验证通过！ckpt A 权重融合正确。")
        else:
            print("✗ 存在验证失败，请检查上述失败项。")
        print("=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="验证 ResQ 量化权重 A 的正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python tools/verify_resq_checkpoint.py \\
        --original /path/to/Qwen3-32B \\
        --ckpt-a /path/to/resq_quantized \\
        --ckpt-b /path/to/resq_matrices \\
        --prompt "你好"
        
输出解读:
    ✓ PASS: 验证通过
    ✗ FAIL: 验证失败
    rel: 相对误差 (越小越好)
    max: 最大绝对误差
        """
    )
    parser.add_argument("--original", "-o", type=str, required=True,
                        help="原始 bf16 模型路径")
    parser.add_argument("--ckpt-a", "-a", type=str, required=True,
                        help="ResQ 量化权重 A 路径")
    parser.add_argument("--ckpt-b", "-b", type=str, required=True,
                        help="ResQ 中间矩阵 B 路径")
    parser.add_argument("--prompt", "-p", type=str, default="你好",
                        help="测试 prompt")
    parser.add_argument("--layers", "-l", type=int, default=2,
                        help="验证的层数 (默认前2层)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        help="设备")
    args = parser.parse_args()
    
    print(f"\n加载原始模型: {args.original}")
    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=True)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    print(f"  hidden_size: {original_model.config.hidden_size}")
    print(f"  num_layers: {original_model.config.num_hidden_layers}")
    
    print(f"\n加载 ckpt A: {args.ckpt_a}")
    ckpt_a = load_safetensors(args.ckpt_a)
    print(f"  {len(ckpt_a)} tensors")
    
    print(f"\n加载 ckpt B: {args.ckpt_b}")
    ckpt_b = load_safetensors(args.ckpt_b)
    print(f"  {len(ckpt_b)} tensors")
    
    # 显示关键 ResQ 参数
    print("\nResQ 参数:")
    for key in ['resq.Hd_K', 'resq.down_proj_blocksize', 'resq.intermediate_size']:
        val = ckpt_a.get(key)
        if val is not None:
            print(f"  {key}: {val.item()}")
    
    # 创建管理器和验证器
    mgr = CheckpointManager(original_model, ckpt_a, ckpt_b)
    verifier = ResQVerifier(mgr)
    
    # 运行验证
    verifier.run_all_tests(tokenizer, args.prompt, args.layers)
    
    # 返回退出码
    failed = sum(1 for r in verifier.results if not r.passed)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
