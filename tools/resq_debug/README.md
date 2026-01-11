# ResQ Debug Tools

用于调试和验证 ResQ 量化模型前向推理的工具集。

## 文件说明

| 文件 | 用途 |
|------|------|
| `save_activations.py` | 保存和比较激活值（两阶段，避免 OOM） |
| `compare_forward.py` | 实时对比原始模型 vs ResQ 模型 |
| `modeling_qwen3_resq.py` | ResQ Qwen3 模型定义（基于 transformers） |
| `fake_quant_forward.py` | 伪量化验证（注入 Hadamard 变换） |

## 快速开始

### 前提条件

你需要准备以下文件：

| 文件 | 说明 |
|------|------|
| **权重 O** | 原始 Qwen3 bf16 模型目录 |
| **权重 A** | msmodelslim 量化得到的 checkpoint_A.pt（包含量化权重和 resq.Hd/Uc/Pd） |
| **权重 B** | transforms_B.pt（辅助验证矩阵，可选） |

---

## 方案 1：两阶段保存和比较激活值（推荐）

适用于内存有限或需要在不同机器上运行的场景。

### 步骤 1a：保存原始模型激活值（CPU）

```bash
python -m tools.resq_debug.save_activations \
    --mode original \
    --model /path/to/qwen3-32b-bf16 \
    --output orig_acts.pt \
    --prompt "What is artificial intelligence?" \
    --max-tokens 20 \
    --device cpu
```

### 步骤 1b：保存 ResQ 模型激活值（NPU）

```bash
python -m tools.resq_debug.save_activations \
    --mode resq \
    --model /path/to/qwen3-32b-bf16 \
    --ckpt-a /path/to/checkpoint_A.pt \
    --ckpt-b /path/to/transforms_B.pt \
    --output resq_acts.pt \
    --prompt "What is artificial intelligence?" \
    --max-tokens 20 \
    --device npu
```

### 步骤 2a：比较激活值（显示所有细节）

```bash
python -m tools.resq_debug.save_activations \
    --mode compare \
    --orig-file orig_acts.pt \
    --resq-file resq_acts.pt
```

### 步骤 2b：比较激活值（只看摘要，推荐用于 64 层模型）

```bash
python -m tools.resq_debug.save_activations \
    --mode compare \
    --orig-file orig_acts.pt \
    --resq-file resq_acts.pt \
    --summary-only
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layers` | `all` | 保存哪些层：`all` 或 `0,15,31,63` |
| `--max-tokens` | `20` | 生成的最大 token 数 |
| `--prompt` | `"Hello, how are you?"` | 测试 prompt |
| `--summary-only` | `False` | 比较时只显示摘要 |

---

## 方案 2：实时对比（需要足够内存）

两个模型同时加载，实时对比激活值。

```bash
python -m tools.resq_debug.compare_forward \
    --original /path/to/qwen3-32b-bf16 \
    --ckpt-a /path/to/checkpoint_A.pt \
    --ckpt-b /path/to/transforms_B.pt \
    --prompt "What is artificial intelligence?" \
    --layers 0,31,63 \
    --orig-device cpu \
    --resq-device npu \
    --max-tokens 20
```

---

## 方案 3：伪量化验证

在原始模型上动态注入 ResQ Hadamard 变换，验证变换逻辑是否正确。

```bash
python -m tools.resq_debug.fake_quant_forward \
    --model /path/to/qwen3-32b-bf16 \
    --transforms /path/to/checkpoint_A.pt \
    --prompt "Hello, how are you?" \
    --max-tokens 10 \
    --device cpu
```

---

## 输出示例

### 完整比较输出

```
--- Layer 0 ---
  ✓ input_ln: corr=0.9999, rel_err=0.0012, scale=1.0001
  ✓ attn.qkv_input: corr=0.9999, rel_err=0.0012, scale=1.0001
  ✓ attn.q_proj: corr=0.9987, rel_err=0.0523, scale=0.9876
  ✗ mlp.down: corr=0.8234, rel_err=0.4523 ⚠, scale=0.5432
```

### 摘要输出 (`--summary-only`)

```
Comparing 64 layers: 0...63
============================================================

--- Generation ---
  Original: 'AI is a field of computer science...'
  ResQ:     'AI is a field of computer science...'
  Token match: 18/20 (90.0%)

============================================================
Summary
============================================================
Total checks: 832 passed, 12 failed

Per-layer summary:
  Layers all passed: 61
  Layers with failures: [0, 31, 63]

============================================================
✗ FAIL: 12 activation checks failed
```

---

## 判断标准

| 指标 | 通过条件 | 说明 |
|------|----------|------|
| `corr` | > 0.99 | 相关系数 |
| `rel_err` | < 0.1 | 相对误差 |
| `scale` | 0.9 ~ 1.1 | 缩放比例 |

如果某层失败，问题可能在该层的：
- 权重反量化
- 旋转矩阵（Uc/Pd/Hd）应用
- LayerNorm 融合

---

## 注意事项

1. **两次运行的 `--prompt` 必须完全一致**
2. 原始模型 bf16 约需 2x 参数量内存（32B 模型约 64GB）
3. ResQ 量化模型约需 0.5x 参数量内存（32B 模型约 16GB）
4. 保存激活值文件可能较大（每层约 10MB，64 层约 640MB）
