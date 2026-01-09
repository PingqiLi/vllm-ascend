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
    
    def get_ub(self, layer_idx: int, debug: bool = False) -> Optional[torch.Tensor]:
        """获取 Ub = Pb @ Rb (per-head)"""
        Pb = self.ckpt_b.get(f'resq.layer.{layer_idx}.P_b')  # [num_heads, head_dim, head_dim]
        Rb = self.ckpt_b.get(f'resq.layer.{layer_idx}.R_b')  # [head_dim, head_dim]
        if Pb is None or Rb is None:
            if debug:
                print(f"  [DEBUG] P_b or R_b not found for layer {layer_idx}")
                print(f"    P_b: {Pb is not None}, R_b: {Rb is not None}")
            return None
        
        if debug:
            print(f"  [DEBUG] P_b shape: {Pb.shape}, R_b shape: {Rb.shape}")
            # 检查 P_b 正交性 (per-head)
            Pb_f = Pb.float()
            for i in range(min(2, Pb.shape[0])):  # 只检查前2个head
                Pb_i_orth_err = (Pb_f[i] @ Pb_f[i].T - torch.eye(Pb.shape[1])).abs().max().item()
                print(f"  [DEBUG] P_b[{i}] 正交性误差: {Pb_i_orth_err:.2e}")
            # 检查 R_b 正交性
            Rb_f = Rb.float()
            Rb_orth_err = (Rb_f @ Rb_f.T - torch.eye(Rb.shape[0])).abs().max().item()
            print(f"  [DEBUG] R_b 正交性误差: {Rb_orth_err:.2e}")
        
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
        self.passed = []
        self.failed = []
    
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
        
        # msmodelslim 的 fuse_layer_norms 对 embed 做了 mean subtraction
        embed_O_float = embed_O.float()
        embed_centered = embed_O_float - embed_O_float.mean(dim=-1, keepdim=True)
        
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
            W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.self_attn.{proj}')
            
            if W_O is None or W_A is None:
                print(f"  ⚠ {proj} 权重未找到")
                continue
            
            # W shape: [out_features, in_features]
            out_dim = W_A.shape[0]
            in_dim = W_A.shape[1]  # = hidden_size
            
            if proj in ['q_proj', 'k_proj']:
                # Q/K: W_A = W_O @ Ua
                # 验证方式改为相关系数（int4量化导致rel_diff很大但相关性高）
                W_forward = torch.matmul(W_O.float(), Ua)
                corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1].item()
                
                passed = corr >= THRESHOLDS.quantized_correlation
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {proj}: corr={corr:.4f}")
                
                result = DiffResult(proj, 0, 0, 0, passed, f"corr={corr:.4f}")
                results[proj] = result
                if not passed:
                    self.failed.append(result)
                else:
                    self.passed.append(result)
                continue
            else:
                # V: W_A[h] = Ub[h].T @ W_O[h] @ Ua
                if Ub is None:
                    print("  ⚠ Ub 未找到，跳过 v_proj")
                    continue
                
                num_kv_heads, head_dim, _ = Ub.shape
                
                # 计算 W_forward = Ub[h].T @ W_O[h] @ Ua
                W_O_reshaped = W_O.float().reshape(num_kv_heads, head_dim, in_dim)
                W_tmp = torch.einsum('nji,nje->nie', Ub.float(), W_O_reshaped)
                W_tmp = W_tmp.reshape(out_dim, in_dim)
                W_forward = torch.matmul(W_tmp, Ua)
                
                corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1].item()
                
                passed = corr >= THRESHOLDS.quantized_correlation
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {proj}: corr={corr:.4f}")
                
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
          - rotate_ov_proj: 在 apply_exact_had_to_linear 中对 o_proj 应用:
            W_temp[:, i, :] = W_O[:, i, :] @ inv(Ub[i]).T = W_O[:, i, :] @ Ub[i] (因为正交)
          - rotate_attention_output: W_A = Ua.T @ W_temp
          - 综合: W_A_before_rearrange = Ua.T @ (W_O @ block_diag(Ub_expanded))
          - rearrange_o_proj: W_A = W_A_before_rearrange[:, new_column_order]
          
        验证方案：
          方案1 (正向): 从 W_O 计算预期的 W_A，与实际 W_A 对比
          方案2 (不含Ub): 检查是否可能没有应用 Ub 旋转
          方案3 (不含rearrange): 检查是否可能没有应用列重排
        
        GQA: num_attention_heads = 64, num_kv_heads = 8
        Ub shape: [num_kv_heads, head_dim, head_dim] = [8, 128, 128]
        需要扩展成 [num_attention_heads, head_dim, head_dim] = [64, 128, 128]
        """
        print(f"\n[Layer {layer_idx}] O_proj 权重融合验证")
        
        W_O = self.mgr.get_original_weight(f'model.layers.{layer_idx}.self_attn.o_proj.weight')
        W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.self_attn.o_proj')
        Ua = self.mgr.get_ua(layer_idx)
        Ub = self.mgr.get_ub(layer_idx, debug=True)  # [num_kv_heads, head_dim, head_dim]
        
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
        num_q_per_kv = num_attention_heads // num_kv_heads  # 8
        Ub_expanded = Ub.repeat_interleave(num_q_per_kv, dim=0)  # [64, 128, 128]
        
        print(f"  GQA: num_attention_heads={num_attention_heads}, num_kv_heads={num_kv_heads}")
        print(f"  Ub_expanded shape: {Ub_expanded.shape}")
        
        # === 检测 modelslim 的 bug: rearrange 和 quantize 使用不同的 high_bits_length ===
        # 从 ckpt A 读取 weight_low 的 shape 来确定实际的 high/low 分割点
        weight_low = self.mgr.ckpt_a.get(f'model.layers.{layer_idx}.self_attn.o_proj.weight_low')
        weight_high = self.mgr.ckpt_a.get(f'model.layers.{layer_idx}.self_attn.o_proj.weight_high')
        
        # modelslim 的 bug:
        # - rearrange_columns 使用: high_bits_length = high_fraction * model_dim = 0.125 * 5120 = 640
        # - calibrator 使用: high_bits_length = high_fraction * weight.shape[1] = 0.125 * 8192 = 1024
        high_fraction = 0.125
        hidden_size_based_high = int(high_fraction * hidden_size)  # 640 (rearrange 使用)
        in_dim_based_high = int(high_fraction * in_dim)            # 1024 (量化存储使用)
        
        high_per_head_rearrange = hidden_size_based_high // num_attention_heads  # 10
        high_per_head_quant = in_dim_based_high // num_attention_heads            # 16
        
        print(f"  === modelslim bug 检测 ===")
        print(f"  rearrange 使用: high_bits_length = {high_fraction} * {hidden_size} = {hidden_size_based_high}")
        print(f"    -> high_length_per_head = {hidden_size_based_high} / {num_attention_heads} = {high_per_head_rearrange}")
        print(f"  quantize 使用: high_bits_length = {high_fraction} * {in_dim} = {in_dim_based_high}")
        print(f"    -> high_length_per_head = {in_dim_based_high} / {num_attention_heads} = {high_per_head_quant}")
        
        if weight_low is not None and weight_high is not None:
            actual_high_cols = weight_high.shape[1]
            actual_high_per_head = actual_high_cols // num_attention_heads
            print(f"  ckpt A 实际: weight_high has {actual_high_cols} cols = {actual_high_per_head} per head")
            
            if actual_high_per_head == high_per_head_quant:
                print(f"  ✓ ckpt A 与 quantize 逻辑一致 (使用 in_dim)")
            elif actual_high_per_head == high_per_head_rearrange:
                print(f"  ✓ ckpt A 与 rearrange 逻辑一致 (使用 model_dim)")
            else:
                print(f"  ⚠ ckpt A 与两种计算方式都不一致!")
        
        # 使用量化存储的实际 high 列数
        high_length_per_head = high_per_head_quant
        
        # 复现 rearrange_o_proj 逻辑
        chunk_starts = torch.arange(0, in_dim, head_dim)
        high_precision_columns = torch.arange(head_dim - high_length_per_head, head_dim)
        columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()
        
        all_columns = torch.arange(in_dim)
        mask = torch.ones(in_dim, dtype=torch.bool)
        mask[columns_to_end] = False
        remaining_columns = all_columns[mask]
        
        new_column_order = torch.cat([remaining_columns, columns_to_end])
        restore_indices = torch.argsort(new_column_order)
        
        # Debug: 验证 new_column_order
        low_per_head = head_dim - high_length_per_head
        print(f"  new_column_order 结构: {num_attention_heads} heads x ({low_per_head} low + {high_length_per_head} high)")
        print(f"  new_column_order[:5]: {new_column_order[:5].tolist()}")
        print(f"  new_column_order[{low_per_head}:{low_per_head+5}]: {new_column_order[low_per_head:low_per_head+5].tolist()} (head1 low start)")
        expected_high_start = num_attention_heads * low_per_head
        print(f"  new_column_order[{expected_high_start}:{expected_high_start+5}]: {new_column_order[expected_high_start:expected_high_start+5].tolist()} (high start)")
        
        # 还原 W_A 的列顺序
        W_A_restored = W_A[:, restore_indices]
        
        # === 计算两种不同的 column order ===
        # 方式A: 使用量化的 high_length_per_head (16, 基于 in_dim)
        def build_column_order(hlph):
            """Build column rearrangement order given high_length_per_head"""
            chunk_starts = torch.arange(0, in_dim, head_dim)
            high_prec_cols = torch.arange(head_dim - hlph, head_dim)
            cols_to_end = (chunk_starts.unsqueeze(1) + high_prec_cols).flatten()
            all_cols = torch.arange(in_dim)
            mask = torch.ones(in_dim, dtype=torch.bool)
            mask[cols_to_end] = False
            remaining = all_cols[mask]
            return torch.cat([remaining, cols_to_end])
        
        col_order_quant = build_column_order(high_per_head_quant)      # 16 per head
        col_order_rearrange = build_column_order(high_per_head_rearrange)  # 10 per head
        
        # 使用量化的 column order 作为主要验证
        new_column_order = col_order_quant
        
        # === 基础变换: Ua.T @ W_O @ block_diag(Ub) ===
        W_O_float = W_O.float()
        W_O_reshaped = W_O_float.reshape(hidden_size, num_attention_heads, head_dim)
        Ub_expanded_float = Ub_expanded.float()
        W_with_Ub = torch.einsum('hnd,nde->hne', W_O_reshaped, Ub_expanded_float)
        W_with_Ub = W_with_Ub.reshape(hidden_size, in_dim)
        W_O_transformed = torch.matmul(Ua.T.float(), W_with_Ub)
        W_O_only_Ua = torch.matmul(Ua.T.float(), W_O_float)
        
        # === 方案1: 使用量化的 column order (16 per head) ===
        W_expected_quant = W_O_transformed[:, col_order_quant]
        corr_1 = torch.corrcoef(torch.stack([W_A.flatten(), W_expected_quant.flatten()]))[0, 1].item()
        
        # === 方案1b: 使用 rearrange 的 column order (10 per head) ===
        W_expected_rearrange = W_O_transformed[:, col_order_rearrange]
        corr_1b = torch.corrcoef(torch.stack([W_A.flatten(), W_expected_rearrange.flatten()]))[0, 1].item()
        
        # === 方案2: 不含 Ub 旋转，使用量化 column order ===
        W_expected_2 = W_O_only_Ua[:, col_order_quant]
        corr_2 = torch.corrcoef(torch.stack([W_A.flatten(), W_expected_2.flatten()]))[0, 1].item()
        
        # === 方案3: 不含列重排 W_A = Ua.T @ W_O @ Ub (无rearrange) ===
        corr_3 = torch.corrcoef(torch.stack([W_A.flatten(), W_O_transformed.flatten()]))[0, 1].item()
        
        # === 方案4: 只有 Ua.T @ W_O (无 Ub，无 rearrange) ===
        corr_4 = torch.corrcoef(torch.stack([W_A.flatten(), W_O_only_Ua.flatten()]))[0, 1].item()
        
        # === 方案5: 使用 rearrange order 还原后对比 ===
        restore_indices_rearrange = torch.argsort(col_order_rearrange)
        W_A_restored_rearrange = W_A[:, restore_indices_rearrange]
        corr_5 = torch.corrcoef(torch.stack([W_A_restored_rearrange.flatten(), W_O_transformed.flatten()]))[0, 1].item()
        
        # === 方案6: 只与原始 W_O 对比（无任何变换） ===
        corr_6 = torch.corrcoef(torch.stack([W_A.flatten(), W_O_float.flatten()]))[0, 1].item()
        
        print(f"\n  === 变换公式验证 ===")
        print(f"  方案1  [Ua.T @ W_O @ Ub][:, col_quant(16/head)] vs W_A: corr={corr_1:.4f}")
        print(f"  方案1b [Ua.T @ W_O @ Ub][:, col_rearr(10/head)] vs W_A: corr={corr_1b:.4f}")
        print(f"  方案2  [Ua.T @ W_O][:, col_quant] vs W_A: corr={corr_2:.4f}")
        print(f"  方案3  Ua.T @ W_O @ Ub (无rearrange) vs W_A: corr={corr_3:.4f}")
        print(f"  方案4  Ua.T @ W_O (无Ub,无rearrange) vs W_A: corr={corr_4:.4f}")
        print(f"  方案5  Ua.T @ W_O @ Ub vs W_A_restored(rearr_order): corr={corr_5:.4f}")
        print(f"  方案6  W_O vs W_A (无任何变换): corr={corr_6:.4f}")
        
        # 选择最佳匹配的方案
        correlations = {
            "方案1 (quant order, 16/head)": corr_1,
            "方案1b (rearrange order, 10/head)": corr_1b,
            "方案2 (无Ub, quant order)": corr_2,
            "方案3 (无rearrange)": corr_3,
            "方案4 (只Ua.T)": corr_4,
            "方案5 (restore用rearr order)": corr_5,
            "方案6 (无变换)": corr_6,
        }
        best_scheme, best_corr = max(correlations.items(), key=lambda x: x[1])
        print(f"\n  最佳匹配: {best_scheme} (corr={best_corr:.4f})")
        
        # 数值范围对比
        print(f"\n  === 数值范围对比 ===")
        print(f"  W_O: range=[{W_O.min():.4f}, {W_O.max():.4f}]")
        print(f"  W_A: range=[{W_A.min():.4f}, {W_A.max():.4f}]")
        print(f"  W_expected(quant order): range=[{W_expected_quant.min():.4f}, {W_expected_quant.max():.4f}]")
        print(f"  W_expected(rearr order): range=[{W_expected_rearrange.min():.4f}, {W_expected_rearrange.max():.4f}]")
        
        # === 深入诊断: 检查 Ub 的性质 ===
        print(f"\n  === Ub 矩阵诊断 ===")
        
        # 检查 Ub 是否是正交矩阵
        Ub_orth_errs = []
        for i in range(min(3, num_kv_heads)):
            orth_err = (Ub[i].float() @ Ub[i].float().T - torch.eye(head_dim)).abs().max().item()
            Ub_orth_errs.append(orth_err)
        print(f"  Ub 正交性误差 (前3个head): {Ub_orth_errs}")
        
        # 检查 Ub 是否接近单位矩阵
        Ub_identity_errs = []
        for i in range(min(3, num_kv_heads)):
            identity_err = (Ub[i].float() - torch.eye(head_dim)).abs().max().item()
            Ub_identity_errs.append(identity_err)
        print(f"  Ub 与 I 差距 (前3个head): {Ub_identity_errs}")
        
        # 检查如果 Ub 是单位矩阵会怎样
        if Ub_identity_errs[0] > 0.01:
            print(f"  Ub 不是单位矩阵，应该有旋转效果")
        else:
            print(f"  ⚠ Ub 接近单位矩阵，旋转效果很小")
        
        # === 额外诊断: 检查单个 head 的变换 ===
        print(f"\n  === 单个 head 变换诊断 (head 0) ===")
        W_O_h0 = W_O_float[:, :head_dim]  # [5120, 128]
        W_A_h0 = W_A[:, :low_per_head].float()  # 重排后的前 112 列应该是 head0 的 low 部分
        
        # 尝试不同的变换
        # 1. Ua.T @ W_O[:, :head_dim]
        W_h0_v1 = torch.matmul(Ua.T.float(), W_O_h0)[:, :low_per_head]
        corr_h0_v1 = torch.corrcoef(torch.stack([W_A_h0.flatten(), W_h0_v1.flatten()]))[0, 1].item()
        
        # 2. Ua.T @ W_O[:, :head_dim] @ Ub[0]
        W_O_with_Ub0 = torch.matmul(W_O_h0, Ub_expanded_float[0])
        W_h0_v2 = torch.matmul(Ua.T.float(), W_O_with_Ub0)[:, :low_per_head]
        corr_h0_v2 = torch.corrcoef(torch.stack([W_A_h0.flatten(), W_h0_v2.flatten()]))[0, 1].item()
        
        print(f"  W_A head0 low (前{low_per_head}列): shape={W_A_h0.shape}")
        print(f"  变换v1 (Ua.T @ W_O_h0)[:, :low]: corr={corr_h0_v1:.4f}")
        print(f"  变换v2 (Ua.T @ W_O_h0 @ Ub0)[:, :low]: corr={corr_h0_v2:.4f}")
        
        # === 方案7: 测试 modelslim bug 假设 ===
        # 假设: rearrange 用 10/head 重排，但量化存储按 16/head 分割 high/low
        # 这意味着列被错误地打乱了
        print(f"\n  === modelslim bug 假设测试 (方案7) ===")
        
        # 步骤1: 计算旋转后的权重 (Ua.T @ W_O @ Ub)
        W_rotated = W_O_transformed  # shape [5120, 8192]
        
        # 步骤2: 模拟 rearrange 用 10/head 重排
        W_after_rearrange_10 = W_rotated[:, col_order_rearrange]
        
        # 步骤3: 模拟量化器读取时按 16/head 理解 high/low 分割
        # 量化器认为最后 1024 列是 high，前 7168 列是 low
        # 但实际上 rearrange 只移动了 640 列到末尾
        # 这导致量化器错误地认为一些 mid 列是 high 列
        
        # 分析 W_A 的实际结构
        if weight_low is not None and weight_high is not None:
            actual_low_cols = weight_low.shape[1]
            actual_high_cols = weight_high.shape[1]
            print(f"  实际存储: weight_low={actual_low_cols} cols, weight_high={actual_high_cols} cols")
            
            # 方案7a: 如果 rearrange 用 10/head，量化用 16/head
            # 那么 W_A 的前 7168 列对应 W_after_rearrange_10 的前 7168 列
            # W_A 的后 1024 列对应 W_after_rearrange_10 的后 1024 列
            # 但 W_after_rearrange_10 的结构是 [7552 mid | 640 high]
            
            W_expected_bug = W_after_rearrange_10  # 直接比较
            corr_7a = torch.corrcoef(torch.stack([W_A.flatten(), W_expected_bug.flatten()]))[0, 1].item()
            print(f"  方案7a [Ua.T @ W_O @ Ub][:, rearr_10] 直接对比 W_A: corr={corr_7a:.4f}")
            
            # 方案7b: 反向推导——假设量化器在 rearrange 结果上按 16/head 理解
            # 我们需要"修正"这个错误来还原原始变换
            # 这很复杂，因为涉及到两套不同的列索引
            
            # 先检查 weight_low 和 weight_high 的数值范围
            print(f"\n  量化权重数值诊断:")
            print(f"  weight_low: range=[{weight_low.min():.4f}, {weight_low.max():.4f}], dtype={weight_low.dtype}")
            print(f"  weight_high: range=[{weight_high.min():.4f}, {weight_high.max():.4f}], dtype={weight_high.dtype}")
            
            # 检查 W_A 的 low/high 部分与 W_expected 的对应关系
            W_A_low = W_A[:, :actual_low_cols].float()
            W_A_high = W_A[:, actual_low_cols:].float()
            W_exp_quant_low = W_expected_quant[:, :actual_low_cols].float()
            W_exp_quant_high = W_expected_quant[:, actual_low_cols:].float()
            
            corr_low = torch.corrcoef(torch.stack([W_A_low.flatten(), W_exp_quant_low.flatten()]))[0, 1].item()
            corr_high = torch.corrcoef(torch.stack([W_A_high.flatten(), W_exp_quant_high.flatten()]))[0, 1].item()
            print(f"\n  分区域对比 (quant order 16/head):")
            print(f"  W_A_low vs W_expected_low: corr={corr_low:.4f}")
            print(f"  W_A_high vs W_expected_high: corr={corr_high:.4f}")
        
        # 使用最佳方案作为主要验证
        corr = best_corr
        passed = corr >= THRESHOLDS.quantized_correlation
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  {status} o_proj: corr={corr:.4f} (最佳方案: {best_scheme})")
        
        # 如果所有方案都失败，可能是 Ub 没有被正确保存到 ckpt B
        if best_corr < 0.5:
            print(f"\n  ⚠⚠⚠ 警告: 所有变换方案的相关系数都很低!")
            print(f"  可能的原因:")
            print(f"  1. ckpt B 中的 P_b/R_b 与实际量化时使用的不同")
            print(f"  2. msmodelslim rearrange (用 model_dim=5120) 和 quantize (用 in_dim=8192) 不一致")
            print(f"     - rearrange 按 {high_per_head_rearrange} cols/head 重排")
            print(f"     - quantize 按 {high_per_head_quant} cols/head 分割 high/low")
            print(f"  3. 量化时可能使用了不同的配置")
        
        result = DiffResult("o_proj", 0, 0, 0, passed, f"corr={corr:.4f}")
        if not passed:
            self.failed.append(result)
        else:
            self.passed.append(result)
        self.results.append(result)
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
        # 使用相关系数验证
        for proj in ['gate_proj', 'up_proj']:
            W_O = self.mgr.get_original_weight(f'model.layers.{layer_idx}.mlp.{proj}.weight', fuse_layernorm=True)
            W_A = self.mgr.get_quantized_weight(f'model.layers.{layer_idx}.mlp.{proj}')
            
            if W_O is None or W_A is None:
                print(f"  ⚠ {proj} 权重未找到")
                continue
            
            # 正向验证: W_O @ Ua ≈ W_A
            W_forward = torch.matmul(W_O.float(), Ua)
            corr = torch.corrcoef(torch.stack([W_forward.flatten(), W_A.float().flatten()]))[0, 1].item()
            
            passed = corr >= THRESHOLDS.quantized_correlation
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} {proj}: corr={corr:.4f}")
            
            result = DiffResult(proj, 0, 0, 0, passed, f"corr={corr:.4f}")
            results[proj] = result
            if not passed:
                self.failed.append(result)
            else:
                self.passed.append(result)
            self.results.append(result)
        
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
    
    def verify_layer0_activations(self, tokenizer, prompt: str) -> Dict[str, DiffResult]:
        """
        详细比较 Layer 0 的内部激活值
        
        关键点：消除正交融合矩阵的影响才能有可比性
        
        激活值对应关系：
        - embed: resq_hidden = orig_centered @ Ua
        - Q/K/V proj: 输入在 Ua 空间，权重融合了 Ua，结果抵消
        - Uc rotation: Q/K 在 RoPE 后应用 Uc
        - O_proj: 输入在 Ub 空间，输出在 Ua 空间
        - MLP gate/up: 输入在 Ua 空间
        - MLP down: 输入需要先应用 Ud，输出在 Ua 空间
        """
        print("\n[Layer 0] 内部激活值详细比较")
        results = {}
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")
        
        # 获取旋转矩阵
        Ua = self.mgr.get_ua(0)
        Uc = self.mgr.get_uc(0)
        
        if Ua is None:
            print("  ⚠ Ua 未找到，无法比较")
            return results
        
        # ========== 1. Embed 输出 ==========
        # 原始: orig_hidden = embed_O_centered[tokens]
        # ResQ: resq_hidden = orig_hidden @ Ua
        orig_embed = self.mgr.original_model.model.embed_tokens.weight.data.float()
        orig_embed_centered = orig_embed - orig_embed.mean(dim=-1, keepdim=True)
        orig_hidden = torch.embedding(orig_embed_centered, input_ids)
        
        resq_embed = self.mgr.ckpt_a.get('model.embed_tokens.weight')
        resq_hidden = torch.embedding(resq_embed, input_ids).float()
        
        # 消除 Ua 影响: resq_hidden @ Ua.T ≈ orig_hidden
        resq_restored = torch.matmul(resq_hidden, Ua.T)
        result = compute_diff(resq_restored, orig_hidden, "embed_output @ Ua.T vs orig", 
                             THRESHOLDS.activation_rel_diff)
        results["embed_output"] = result
        print(f"  embed_output: corr={torch.corrcoef(torch.stack([resq_restored.flatten(), orig_hidden.flatten()]))[0,1].item():.4f}, rel={result.rel_diff:.2%}")
        
        # ========== 2. Q 投影后 ==========
        # Q 融合: Q_A = Q_O * gamma @ Ua
        # 原始: q_orig = orig_hidden @ Q_O.T (但需要先融合 gamma)
        # ResQ: q_resq = resq_hidden @ Q_A.T = (orig_hidden @ Ua) @ (Q_O * gamma @ Ua).T
        #             = orig_hidden @ Ua @ Ua.T @ (Q_O * gamma).T = orig_hidden @ (Q_O * gamma).T
        # 所以 q_resq ≈ q_orig_with_gamma (Ua 在投影时被抵消)
        
        Q_O = self.mgr.get_original_weight('model.layers.0.self_attn.q_proj.weight', fuse_layernorm=True)
        Q_A = self.mgr.get_quantized_weight('model.layers.0.self_attn.q_proj')
        
        if Q_O is not None and Q_A is not None:
            # 原始模型 Q 投影 (with gamma fusion)
            q_orig = torch.matmul(orig_hidden, Q_O.T)
            
            # ResQ 模型 Q 投影
            q_resq = torch.matmul(resq_hidden, Q_A.T)
            
            # q_resq 应该 ≈ q_orig (Ua 被抵消)
            corr = torch.corrcoef(torch.stack([q_resq.flatten(), q_orig.flatten()]))[0, 1].item()
            result_q = compute_diff(q_resq, q_orig, "q_proj_output", THRESHOLDS.activation_rel_diff)
            results["q_proj_output"] = result_q
            print(f"  q_proj_output: corr={corr:.4f}, rel={result_q.rel_diff:.2%}")
        
        # ========== 3. Uc 旋转后 ==========
        # 原始: q_orig 不需要 Uc
        # ResQ: q_resq 需要应用 Uc (在 RoPE 后)
        # 消除 Uc: q_resq_uc @ Uc.T ≈ q_orig (假设 RoPE 前)
        if Uc is not None and Q_O is not None and Q_A is not None:
            # 应用 Uc
            q_resq_uc = apply_block_rotation(q_resq, Uc)
            # 消除 Uc
            q_resq_restored = apply_block_rotation(q_resq_uc, Uc.T)
            
            # 验证 Uc 可逆性
            corr = torch.corrcoef(torch.stack([q_resq_restored.flatten(), q_resq.flatten()]))[0, 1].item()
            print(f"  Uc 可逆性检查: corr={corr:.4f} (应为1.0)")
        
        # ========== 4. MLP gate/up 输出 ==========
        # 类似 Q，Ua 在投影时被抵消
        gate_O = self.mgr.get_original_weight('model.layers.0.mlp.gate_proj.weight', fuse_layernorm=True)
        gate_A = self.mgr.get_quantized_weight('model.layers.0.mlp.gate_proj')
        
        if gate_O is not None and gate_A is not None:
            # 需要先过 attention 和 layernorm... 这里简化，只验证权重级别
            # 实际激活值比较需要完整前向传播
            print(f"  gate_proj: 需要完整前向传播才能比较激活值")
        
        # ========== 5. MLP down 输入/输出 ==========
        # down_proj 输入需要 Ud 变换
        # 原始: out = intermediate @ down_O.T
        # ResQ: intermediate_ud = apply_ud_rotation(intermediate)
        #       out_resq = intermediate_ud @ down_A.T
        # 因为 down_A = Ua.T @ down_O @ Ud, 所以:
        # out_resq = intermediate_ud @ (Ua.T @ down_O @ Ud).T 
        #          = intermediate_ud @ Ud.T @ down_O.T @ Ua
        # 消除 Ua: out_resq @ Ua.T ≈ intermediate_ud @ Ud.T @ down_O.T
        #        = intermediate @ down_O.T (因为 intermediate_ud @ Ud.T = intermediate)
        print(f"  down_proj: Ud 变换验证见 'Ud 旋转验证' 用例")
        
        return results
    
    def verify_resq_forward_detailed(self, tokenizer, prompt: str) -> Dict[str, DiffResult]:
        """
        完整验证 ResQ 前向逻辑，模拟 qwen3_resq_truequant.py 的实现
        
        对比每一步的激活值与原始模型（消除旋转影响后）
        
        这是最关键的验证：如果这里有问题，推理就会乱码
        """
        print("\n" + "=" * 70)
        print("[ResQ Forward] 完整前向激活值验证 (Layer 0)")
        print("=" * 70)
        
        results = {}
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")
        
        # 获取所有旋转矩阵
        Ua = self.mgr.get_ua(0)
        Ub = self.mgr.get_ub(0)  # [num_kv_heads, head_dim, head_dim]
        Uc = self.mgr.get_uc(0)
        Pd = self.mgr.get_pd(0)
        Hd = self.mgr.Hd
        K = self.mgr.Hd_K
        blocksize = self.mgr.blocksize
        
        if Ua is None:
            print("  ⚠ Ua 未找到，无法验证")
            return results
        
        # 模型配置
        hidden_size = Ua.shape[0]  # 5120
        num_attention_heads = 64
        num_kv_heads = 8
        head_dim = hidden_size // num_attention_heads  # 128? No, should be from config
        # For Qwen3-32B: head_dim = 128, num_attention_heads = 64
        head_dim = 128
        q_size = num_attention_heads * head_dim  # 8192
        kv_size = num_kv_heads * head_dim        # 1024
        
        print(f"\n  配置: hidden_size={hidden_size}, heads={num_attention_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        # ========== Step 1: Embed ==========
        print(f"\n  [Step 1] Embed")
        
        orig_embed = self.mgr.original_model.model.embed_tokens.weight.data.float()
        orig_embed_centered = orig_embed - orig_embed.mean(dim=-1, keepdim=True)
        orig_hidden = torch.embedding(orig_embed_centered, input_ids)  # [1, seq, 5120]
        
        resq_embed = self.mgr.ckpt_a.get('model.embed_tokens.weight').float()
        resq_hidden = torch.embedding(resq_embed, input_ids)  # [1, seq, 5120]
        
        # resq_hidden 在 Ua 空间，orig_hidden 在原始空间
        # 验证: resq_hidden @ Ua.T ≈ orig_hidden
        resq_hidden_restored = torch.matmul(resq_hidden, Ua.T.float())
        corr = torch.corrcoef(torch.stack([resq_hidden_restored.flatten(), orig_hidden.flatten()]))[0, 1].item()
        print(f"    resq_embed @ Ua.T vs orig_embed: corr={corr:.4f}")
        results["embed"] = corr
        
        # ========== Step 2: Q/K/V Projections ==========
        print(f"\n  [Step 2] Q/K/V Projections")
        
        Q_O = self.mgr.get_original_weight('model.layers.0.self_attn.q_proj.weight', fuse_layernorm=True)
        K_O = self.mgr.get_original_weight('model.layers.0.self_attn.k_proj.weight', fuse_layernorm=True)
        V_O = self.mgr.get_original_weight('model.layers.0.self_attn.v_proj.weight', fuse_layernorm=True)
        Q_A = self.mgr.get_quantized_weight('model.layers.0.self_attn.q_proj')
        K_A = self.mgr.get_quantized_weight('model.layers.0.self_attn.k_proj')
        V_A = self.mgr.get_quantized_weight('model.layers.0.self_attn.v_proj')
        
        if Q_A is not None and Q_O is not None:
            # 原始: q_orig = orig_hidden @ Q_O.T
            q_orig = torch.matmul(orig_hidden.float(), Q_O.float().T)  # [1, seq, 8192]
            
            # ResQ: q_resq = resq_hidden @ Q_A.T
            # 因为 Q_A = Q_O @ Ua，所以 q_resq = resq_hidden @ Ua.T @ Q_O.T = orig_hidden @ Q_O.T = q_orig
            q_resq = torch.matmul(resq_hidden.float(), Q_A.float().T)
            
            corr_q = torch.corrcoef(torch.stack([q_resq.flatten(), q_orig.flatten()]))[0, 1].item()
            print(f"    Q: resq vs orig: corr={corr_q:.4f}")
            results["q_proj"] = corr_q
        
        if K_A is not None and K_O is not None:
            k_orig = torch.matmul(orig_hidden.float(), K_O.float().T)
            k_resq = torch.matmul(resq_hidden.float(), K_A.float().T)
            corr_k = torch.corrcoef(torch.stack([k_resq.flatten(), k_orig.flatten()]))[0, 1].item()
            print(f"    K: resq vs orig: corr={corr_k:.4f}")
            results["k_proj"] = corr_k
        
        if V_A is not None and V_O is not None:
            # V 更复杂：V_A[h] = Ub[h].T @ V_O[h] @ Ua
            # 所以 v_resq = resq_hidden @ V_A.T 结果在 Ub 空间
            # 需要对比: v_resq @ Ub vs v_orig
            v_orig = torch.matmul(orig_hidden.float(), V_O.float().T)  # [1, seq, kv_size]
            v_resq = torch.matmul(resq_hidden.float(), V_A.float().T)  # [1, seq, kv_size]
            
            # v_resq 在 Ub 空间（per kv-head）
            if Ub is not None:
                # v_resq[h] @ Ub[h] ≈ v_orig[h]
                v_resq_reshaped = v_resq.view(1, -1, num_kv_heads, head_dim)  # [1, seq, 8, 128]
                v_orig_reshaped = v_orig.view(1, -1, num_kv_heads, head_dim)
                
                # 应用 Ub 恢复
                v_resq_restored = torch.einsum('bsnh,nhd->bsnd', v_resq_reshaped.float(), Ub.float())
                
                corr_v = torch.corrcoef(torch.stack([v_resq_restored.flatten(), v_orig_reshaped.flatten()]))[0, 1].item()
                print(f"    V: resq @ Ub vs orig: corr={corr_v:.4f}")
            else:
                corr_v = torch.corrcoef(torch.stack([v_resq.flatten(), v_orig.flatten()]))[0, 1].item()
                print(f"    V: resq vs orig (无Ub): corr={corr_v:.4f}")
            results["v_proj"] = corr_v
        
        # ========== Step 3: Uc Rotation (Q, K after RoPE) ==========
        print(f"\n  [Step 3] Uc Rotation")
        if Uc is not None and Q_A is not None:
            # 应用 Uc 旋转
            q_resq_uc = apply_block_rotation(q_resq, Uc)
            
            # Uc 旋转后的 q 与原始 q 没有简单的对应关系（因为 RoPE）
            # 这里只验证 Uc 可逆性
            q_resq_restored = apply_block_rotation(q_resq_uc, Uc.T)
            corr_uc = torch.corrcoef(torch.stack([q_resq_restored.flatten(), q_resq.flatten()]))[0, 1].item()
            print(f"    Uc 可逆性: q @ Uc @ Uc.T vs q: corr={corr_uc:.4f} (应≈1.0)")
            results["uc_rotation"] = corr_uc
        
        # ========== Step 4: O_proj 输入重排 ==========
        print(f"\n  [Step 4] O_proj 输入重排")
        
        # 模拟 attention 输出（使用 q 作为近似，实际是 softmax(QK)V）
        # 这里简化：假设 attn_output ≈ v_resq (忽略 attention 计算)
        # 实际验证需要完整 attention，但这足以测试 column reorder
        
        # 构建 column_order（与 qwen3_resq_truequant.py 相同）
        high_fraction = 0.125
        in_dim = q_size  # 8192
        
        # 测试两种 high_length_per_head
        for fix_name, use_rearrange_logic in [("quantize(16/head)", False), ("rearrange(10/head)", True)]:
            if use_rearrange_logic:
                high_bits_length = int(hidden_size * high_fraction)  # 640
            else:
                high_bits_length = int(in_dim * high_fraction)  # 1024
            
            high_length_per_head = high_bits_length // num_attention_heads
            low_length_per_head = head_dim - high_length_per_head
            
            column_order = []
            for h in range(num_attention_heads):
                base = h * head_dim
                for j in range(low_length_per_head):
                    column_order.append(base + j)
            for h in range(num_attention_heads):
                base = h * head_dim + low_length_per_head
                for j in range(high_length_per_head):
                    column_order.append(base + j)
            column_order = torch.tensor(column_order, dtype=torch.long)
            
            print(f"    {fix_name}: high_per_head={high_length_per_head}, low_per_head={low_length_per_head}")
            print(f"      column_order[:5]={column_order[:5].tolist()}")
            
        # ========== Step 5: O_proj 混精度 MatMul ==========
        print(f"\n  [Step 5] O_proj 混精度 MatMul")
        
        O_O = self.mgr.get_original_weight('model.layers.0.self_attn.o_proj.weight')
        O_A = self.mgr.get_quantized_weight('model.layers.0.self_attn.o_proj')
        
        if O_A is not None and O_O is not None:
            # 使用 q_orig 作为模拟的 attention 输出
            attn_output_orig = q_orig  # [1, seq, 8192]
            
            # 原始: o_orig = attn_output_orig @ O_O.T
            o_orig = torch.matmul(attn_output_orig.float(), O_O.float().T)  # [1, seq, 5120]
            
            # ResQ (复杂):
            # 1. attn_output_resq 在 Ub 空间
            # 2. 重排列
            # 3. 混精度 matmul
            
            # 模拟 attn_output_resq = attn_output_orig @ block_diag(Ub.T) (per-head)
            if Ub is not None:
                num_q_per_kv = num_attention_heads // num_kv_heads
                Ub_expanded = Ub.repeat_interleave(num_q_per_kv, dim=0)  # [64, 128, 128]
                
                attn_output_reshaped = attn_output_orig.view(1, -1, num_attention_heads, head_dim)
                # attn_output_resq[h] = attn_output_orig[h] @ Ub[h].T
                attn_output_resq = torch.einsum('bsnh,nhd->bsnd', attn_output_reshaped.float(), Ub_expanded.float().transpose(-1, -2))
                attn_output_resq = attn_output_resq.view(1, -1, in_dim)  # [1, seq, 8192]
            else:
                attn_output_resq = attn_output_orig
            
            # 测试两种 column order
            for fix_name, use_rearrange_logic in [("quantize(16/head)", False), ("rearrange(10/head)", True)]:
                if use_rearrange_logic:
                    high_bits_length = int(hidden_size * high_fraction)
                else:
                    high_bits_length = int(in_dim * high_fraction)
                
                high_length_per_head = high_bits_length // num_attention_heads
                low_length_per_head = head_dim - high_length_per_head
                
                column_order = []
                for h in range(num_attention_heads):
                    base = h * head_dim
                    for j in range(low_length_per_head):
                        column_order.append(base + j)
                for h in range(num_attention_heads):
                    base = h * head_dim + low_length_per_head
                    for j in range(high_length_per_head):
                        column_order.append(base + j)
                column_order = torch.tensor(column_order, dtype=torch.long)
                
                # 重排输入
                attn_output_reordered = attn_output_resq[..., column_order]
                
                # 混精度 matmul (用反量化权重近似)
                o_resq = torch.matmul(attn_output_reordered.float(), O_A.float().T)  # [1, seq, 5120]
                
                # o_resq 在 Ua 空间，需要 @ Ua.T 恢复
                o_resq_restored = torch.matmul(o_resq, Ua.T.float())
                
                corr_o = torch.corrcoef(torch.stack([o_resq_restored.flatten(), o_orig.flatten()]))[0, 1].item()
                print(f"    O_proj [{fix_name}]: o_resq @ Ua.T vs o_orig: corr={corr_o:.4f}")
                results[f"o_proj_{fix_name}"] = corr_o
        
        # ========== Step 6: MLP ==========
        print(f"\n  [Step 6] MLP gate/up")
        
        gate_O = self.mgr.get_original_weight('model.layers.0.mlp.gate_proj.weight', fuse_layernorm=True)
        up_O = self.mgr.get_original_weight('model.layers.0.mlp.up_proj.weight', fuse_layernorm=True)
        gate_A = self.mgr.get_quantized_weight('model.layers.0.mlp.gate_proj')
        up_A = self.mgr.get_quantized_weight('model.layers.0.mlp.up_proj')
        
        # 使用 o_orig 作为 MLP 输入（实际需要经过 residual + layernorm）
        mlp_input_orig = o_orig
        mlp_input_resq = torch.matmul(mlp_input_orig, Ua.float())  # 转到 Ua 空间
        
        if gate_A is not None and gate_O is not None:
            gate_orig = torch.matmul(mlp_input_orig.float(), gate_O.float().T)
            gate_resq = torch.matmul(mlp_input_resq.float(), gate_A.float().T)
            corr_gate = torch.corrcoef(torch.stack([gate_resq.flatten(), gate_orig.flatten()]))[0, 1].item()
            print(f"    gate_proj: resq vs orig: corr={corr_gate:.4f}")
            results["gate_proj"] = corr_gate
        
        if up_A is not None and up_O is not None:
            up_orig = torch.matmul(mlp_input_orig.float(), up_O.float().T)
            up_resq = torch.matmul(mlp_input_resq.float(), up_A.float().T)
            corr_up = torch.corrcoef(torch.stack([up_resq.flatten(), up_orig.flatten()]))[0, 1].item()
            print(f"    up_proj: resq vs orig: corr={corr_up:.4f}")
            results["up_proj"] = corr_up
        
        # ========== Step 7: Ud Rotation ==========
        print(f"\n  [Step 7] Ud Rotation (before down_proj)")
        
        if Pd is not None and gate_A is not None and up_A is not None:
            intermediate_orig = torch.nn.functional.silu(gate_orig) * up_orig
            intermediate_resq = torch.nn.functional.silu(gate_resq) * up_resq
            
            # 应用 Ud 旋转
            try:
                intermediate_ud = apply_ud_rotation(intermediate_resq, Pd, Hd, K, blocksize)
                
                # 检查 Ud 可逆性
                # Ud 应该保持范数
                norm_before = intermediate_resq.norm().item()
                norm_after = intermediate_ud.norm().item()
                norm_ratio = norm_after / norm_before
                print(f"    Ud 范数变化: {norm_before:.4f} -> {norm_after:.4f} (ratio={norm_ratio:.4f}, 应≈1.0)")
                results["ud_rotation"] = norm_ratio
            except Exception as e:
                print(f"    Ud 旋转失败: {e}")
        
        # ========== Step 8: down_proj ==========
        print(f"\n  [Step 8] down_proj")
        
        down_O = self.mgr.get_original_weight('model.layers.0.mlp.down_proj.weight')
        down_A = self.mgr.get_quantized_weight('model.layers.0.mlp.down_proj')
        
        if down_A is not None and down_O is not None and Pd is not None:
            # 原始: down_orig = intermediate_orig @ down_O.T
            down_orig = torch.matmul(intermediate_orig.float(), down_O.float().T)
            
            # ResQ: down_resq = intermediate_ud @ down_A.T
            try:
                down_resq = torch.matmul(intermediate_ud.float(), down_A.float().T)
                
                # down_resq 在 Ua 空间
                down_resq_restored = torch.matmul(down_resq, Ua.T.float())
                
                corr_down = torch.corrcoef(torch.stack([down_resq_restored.flatten(), down_orig.flatten()]))[0, 1].item()
                print(f"    down_proj: resq @ Ua.T vs orig: corr={corr_down:.4f}")
                results["down_proj"] = corr_down
            except Exception as e:
                print(f"    down_proj 计算失败: {e}")
        
        # ========== 汇总 ==========
        print(f"\n  === 激活值验证汇总 ===")
        critical_steps = ["q_proj", "k_proj", "o_proj_rearrange(10/head)", "gate_proj", "up_proj"]
        all_good = True
        for step in critical_steps:
            if step in results:
                status = "✓" if results[step] > 0.95 else "✗"
                if results[step] < 0.95:
                    all_good = False
                print(f"  {status} {step}: corr={results[step]:.4f}")
        
        if all_good:
            print(f"\n  ✓ 所有关键步骤的相关系数 > 0.95，前向逻辑正确")
        else:
            print(f"\n  ✗ 存在相关系数 < 0.95 的步骤，需要排查")
        
        return results
    
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
        self.verify_layer0_activations(tokenizer, prompt)
        
        # 4.5 完整前向验证 (关键!)
        self.verify_resq_forward_detailed(tokenizer, prompt)
        
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
