#!/usr/bin/env python3
"""
ResQ TrueQuant 诊断脚本 - 排查推理异常是实现问题还是 msit 量化问题

Usage: python tools/diagnose_resq_truequant.py /path/to/resq_checkpoint
"""

import argparse
import torch
import json
from pathlib import Path
from safetensors import safe_open
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiagResult:
    """诊断结果"""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.errors.append(f"❌ {msg}")
    
    def add_warning(self, msg: str):
        self.warnings.append(f"⚠️  {msg}")
    
    def add_ok(self, msg: str):
        self.info.append(f"✓ {msg}")


def load_safetensors(path: str) -> dict:
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def check_checkpoint_structure(ckpt_path: Path, result: DiagResult):
    """检查 checkpoint 结构"""
    # 检查必要文件
    if not (ckpt_path / 'config.json').exists():
        result.add_error("config.json 不存在")
    else:
        with open(ckpt_path / 'config.json') as f:
            config = json.load(f)
        arch = config.get('architectures', [])
        if 'Qwen3ResQTrueQuantForCausalLM' not in arch:
            result.add_warning(f"config.json architectures={arch}, 应为 Qwen3ResQTrueQuantForCausalLM")
        else:
            result.add_ok("config.json architectures 正确")
    
    # 检查 safetensors 文件
    st_files = list(ckpt_path.glob("*.safetensors"))
    if not st_files:
        result.add_error("没有找到 .safetensors 文件")
        return None
    
    result.add_ok(f"找到权重文件: {st_files[0].name}")
    return st_files[0]


def check_resq_params(weights: dict, result: DiagResult):
    """检查 ResQ 参数"""
    # 必须存在的 ResQ 参数
    required = ['resq.Hd', 'resq.Hd_K', 'resq.down_proj_blocksize']
    for key in required:
        if key not in weights:
            result.add_error(f"缺少必要参数: {key}")
        else:
            t = weights[key]
            if t.numel() == 1:
                result.add_ok(f"{key} = {t.item()}")
            else:
                result.add_ok(f"{key}: shape={tuple(t.shape)}")
    
    # 检查每层的 Uc 和 Pd
    uc_keys = [k for k in weights if 'resq.layer.' in k and '.Uc' in k]
    pd_keys = [k for k in weights if 'resq.layer.' in k and '.Pd' in k]
    
    if not uc_keys:
        result.add_error("没有找到 resq.layer.*.Uc 旋转矩阵")
    else:
        result.add_ok(f"找到 {len(uc_keys)} 个 Uc 旋转矩阵")
    
    if not pd_keys:
        result.add_error("没有找到 resq.layer.*.Pd 旋转矩阵")
    else:
        result.add_ok(f"找到 {len(pd_keys)} 个 Pd 旋转矩阵")


def check_rotation_orthogonality(weights: dict, result: DiagResult):
    """检查旋转矩阵正交性"""
    # 只检查第一层
    uc_key = 'resq.layer.0.Uc'
    if uc_key in weights:
        Uc = weights[uc_key].float()
        if Uc.shape[0] == Uc.shape[1]:
            UUT = torch.matmul(Uc, Uc.T)
            I = torch.eye(Uc.shape[0])
            err = (UUT - I).abs().max().item()
            if err < 1e-3:
                result.add_ok(f"Uc 正交 (误差={err:.6f})")
            else:
                result.add_warning(f"Uc 可能不正交 (误差={err:.4f})")
    
    pd_key = 'resq.layer.0.Pd'
    if pd_key in weights:
        Pd = weights[pd_key].float()
        if Pd.shape[0] == Pd.shape[1]:
            PPT = torch.matmul(Pd, Pd.T)
            I = torch.eye(Pd.shape[0])
            err = (PPT - I).abs().max().item()
            if err < 1e-3:
                result.add_ok(f"Pd 正交 (误差={err:.6f})")
            else:
                result.add_warning(f"Pd 可能不正交 (误差={err:.4f})")
    
    # 检查 Hd 归一化
    if 'resq.Hd' in weights:
        Hd = weights['resq.Hd']
        K = Hd.shape[0]
        expected = 1.0 / (K ** 0.5)
        actual = Hd.abs().max().item()
        if abs(actual - expected) < 0.01:
            result.add_ok(f"Hd 已归一化 (元素≈±{expected:.4f})")
        elif abs(actual - 1.0) < 0.01:
            result.add_warning(f"Hd 未归一化 (元素≈±1), 可能导致数值问题")
        else:
            result.add_warning(f"Hd 归一化状态不明 (max={actual:.4f})")


def check_weight_dimensions(weights: dict, result: DiagResult):
    """检查权重维度是否合理"""
    # 检查 layer 0 的 q_proj
    prefix = "model.layers.0.self_attn.q_proj"
    
    low_key = f"{prefix}.weight_low"
    high_key = f"{prefix}.weight_high"
    
    if low_key not in weights or high_key not in weights:
        result.add_error(f"缺少量化权重: {prefix}.weight_low/high")
        return
    
    w_low = weights[low_key]
    w_high = weights[high_key]
    
    out_dim, in_low = w_low.shape
    _, in_high = w_high.shape
    in_total = in_low + in_high
    high_frac = in_high / in_total
    
    result.add_ok(f"q_proj 维度: out={out_dim}, in_low={in_low}, in_high={in_high}")
    
    if abs(high_frac - 0.125) > 0.01:
        result.add_warning(f"high_fraction={high_frac:.3f}, 期望约 0.125")
    else:
        result.add_ok(f"high_fraction={high_frac:.3f} (正常)")


def check_weight_ranges(weights: dict, result: DiagResult):
    """检查权重值域"""
    prefix = "model.layers.0.self_attn.q_proj"
    
    # weight_low 应该在 [-8, 7] (int4)
    low_key = f"{prefix}.weight_low"
    if low_key in weights:
        w = weights[low_key]
        if w.dtype == torch.int8:
            min_v, max_v = w.min().item(), w.max().item()
            if min_v < -8 or max_v > 7:
                result.add_error(f"weight_low 超出 int4 范围: [{min_v}, {max_v}]")
            else:
                result.add_ok(f"weight_low 在 int4 范围 [{min_v}, {max_v}]")
    
    # weight_high 应该在 [-128, 127] (int8)
    high_key = f"{prefix}.weight_high"
    if high_key in weights:
        w = weights[high_key]
        if w.dtype == torch.int8:
            min_v, max_v = w.min().item(), w.max().item()
            result.add_ok(f"weight_high 在 int8 范围 [{min_v}, {max_v}]")


def test_quantized_matmul(weights: dict, result: DiagResult):
    """测试量化 matmul 精度"""
    prefix = "model.layers.0.self_attn.q_proj"
    
    required = ['weight_low', 'weight_high', 'scale_low', 'scale_high']
    if not all(f"{prefix}.{k}" in weights for k in required):
        result.add_warning("无法测试量化 matmul: 缺少必要权重")
        return
    
    w_low = weights[f"{prefix}.weight_low"]
    w_high = weights[f"{prefix}.weight_high"]
    s_low = weights[f"{prefix}.scale_low"]
    s_high = weights[f"{prefix}.scale_high"]
    
    out_dim, in_low = w_low.shape
    _, in_high = w_high.shape
    in_total = in_low + in_high
    
    # 生成随机输入
    x = torch.randn(1, 4, in_total, dtype=torch.float32)
    x_low = x[..., :in_low]
    x_high = x[..., in_low:]
    
    # 方法1: bf16 反量化
    s_low_exp = s_low.view(-1, 1) if s_low.dim() == 1 else s_low
    s_high_exp = s_high.view(-1, 1) if s_high.dim() == 1 else s_high
    
    w_low_dq = w_low.float() * s_low_exp.float()
    w_high_dq = w_high.float() * s_high_exp.float()
    w_full = torch.cat([w_low_dq, w_high_dq], dim=1)
    y_fp = torch.matmul(x, w_full.T)
    
    # 方法2: 量化 matmul
    x_low_max = x_low.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    x_high_max = x_high.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    lxScale = x_low_max / 7.0
    rxScale = x_high_max / 127.0
    
    x_low_q = torch.round(x_low / lxScale).clamp(-8, 7)
    x_high_q = torch.round(x_high / rxScale).clamp(-128, 127)
    
    y_low = torch.matmul(x_low_q, w_low.float().T) * s_low_exp.T * lxScale
    y_high = torch.matmul(x_high_q, w_high.float().T) * s_high_exp.T * rxScale
    y_quant = y_low + y_high
    
    # 对比
    diff = (y_fp - y_quant).abs()
    rel_err = (diff / (y_fp.abs() + 1e-10)).mean().item()
    
    if rel_err < 0.05:
        result.add_ok(f"量化 matmul 相对误差: {rel_err:.2%} (正常)")
    elif rel_err < 0.15:
        result.add_warning(f"量化 matmul 相对误差: {rel_err:.2%} (较高)")
    else:
        result.add_error(f"量化 matmul 相对误差: {rel_err:.2%} (异常)")


def print_summary(result: DiagResult):
    """打印诊断总结"""
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    if result.errors:
        print(f"\n🔴 发现 {len(result.errors)} 个错误:")
        for e in result.errors:
            print(f"   {e}")
    
    if result.warnings:
        print(f"\n🟡 发现 {len(result.warnings)} 个警告:")
        for w in result.warnings:
            print(f"   {w}")
    
    if not result.errors and not result.warnings:
        print("\n🟢 所有检查通过!")
    
    print("\n" + "-" * 60)
    print("详细信息:")
    for i in result.info:
        print(f"   {i}")
    
    print("\n" + "=" * 60)
    if result.errors:
        print("结论: 权重文件存在问题，请检查 msit 量化流程")
    elif result.warnings:
        print("结论: 权重文件可能有问题，建议进一步排查")
    else:
        print("结论: 权重文件正常，问题可能在模型实现层面")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ResQ TrueQuant 诊断")
    parser.add_argument("checkpoint", type=str, help="ResQ checkpoint 目录")
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ 路径不存在: {ckpt_path}")
        return
    
    result = DiagResult()
    
    print(f"诊断: {ckpt_path}")
    print("-" * 60)
    
    # 1. 检查结构
    st_file = check_checkpoint_structure(ckpt_path, result)
    if not st_file:
        print_summary(result)
        return
    
    # 2. 加载权重
    weights = load_safetensors(str(st_file))
    result.add_ok(f"加载 {len(weights)} 个 tensor")
    
    # 3. 检查 ResQ 参数
    check_resq_params(weights, result)
    
    # 4. 检查旋转矩阵
    check_rotation_orthogonality(weights, result)
    
    # 5. 检查权重维度
    check_weight_dimensions(weights, result)
    
    # 6. 检查权重值域
    check_weight_ranges(weights, result)
    
    # 7. 测试量化精度
    test_quantized_matmul(weights, result)
    
    # 打印总结
    print_summary(result)


if __name__ == "__main__":
    main()
