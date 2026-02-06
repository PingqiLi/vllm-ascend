# ResQ Debug Tools

用于调试 ResQ 量化模型推理问题的工具集。

## Checkpoint 说明

| 名称 | 说明 |
|------|------|
| CKPT_O | 原始 Qwen3 bf16 模型（HuggingFace 格式，含 `model.safetensors.index.json`） |
| CKPT_A | 转换后可直接推理的 ResQ checkpoint（由 `convert_resq_ckpt.py` 生成） |
| CKPT_B | 辅助旋转矩阵（P_a、R_a、P_b、R_b，用于计算 Ua、Ub，离线对比时使用） |

## 文件说明

| 文件 | 说明 |
|------|------|
| `../convert_resq_ckpt.py` | 将原始 ResQ checkpoint 转换为 vLLM 格式（CKPT_A），自动补齐缺失权重 |
| `check_completeness.py` | 验证 CKPT_A 的参数完备性和数值正确性 |
| `run_online.py` | 在线推理脚本（CKPT_A），自动保存指定层的激活值 |
| `run_reference.py` | 保存 CKPT_O bf16 参考激活（用 `device_map="auto"` 分布多卡，不 OOM） |
| `compare_online.py` | 在线 vs 离线对比，验证融合权重、MLP chain、embedding chain |
| `compare_reference.py` | CKPT_O vs CKPT_A 跨模型激活对比，定位误差累积层 |
| `debug_compare.py` | 离线全模型对比（需加载完整 bf16 模型，小模型适用，大模型会 OOM） |
| `wrappers.py` | `ResQLinearWrapper` 和 `W8A8LinearWrapper` 实现 |

---

## 完整流程

以下变量需替换为实际路径：

```bash
export CKPT_O=/path/to/original_bf16_model       # 原始 Qwen3 bf16 模型
export CKPT_A=/path/to/converted_resq_ckpt        # 转换后的 ResQ checkpoint
export CKPT_B=/path/to/rotation_matrices          # 辅助旋转矩阵（离线对比时使用）
export RAW_CKPT=/path/to/raw_resq_ckpt            # 转换前的原始 ResQ checkpoint（convert 的输入）
```

---

### Step 1: 转换 Checkpoint

将原始 ResQ checkpoint 转为 vLLM 可用的 CKPT_A。`--orig_model` 会自动检测
缺失的 passthrough 权重（如 `model.embed_tokens.weight`、`model.norm.weight`、
`lm_head.weight`、各层 layernorm 等），并从 CKPT_O 中复制补齐。

```bash
python -m tools.convert_resq_ckpt \
    --input_path ${RAW_CKPT} \
    --output_path ${CKPT_A} \
    --orig_model ${CKPT_O}
```

预期输出示例：
```
Removed stale file: model-00001-of-00007.safetensors
...
down_proj quantization type: W8A8_DYNAMIC, is_resq: False
Detected 64 layers.
Converting safetensors: 100%|███| 7/7

Copying 3 missing weight(s) from original model:
  model.embed_tokens.weight: shape=[151936, 4096] dtype=torch.bfloat16
  model.norm.weight: shape=[4096] dtype=torch.bfloat16
  lm_head.weight: shape=[151936, 4096] dtype=torch.bfloat16

Updated model.safetensors.index.json with ... entries.
Conversion complete.
```

---

### Step 2: 验证 Checkpoint 完备性

确认 CKPT_A 包含所有必需的参数，且 passthrough 权重数值与 CKPT_O 一致。

```bash
python -m tools.resq_debug.check_completeness \
    --ckpt-o ${CKPT_O} \
    --ckpt-a ${CKPT_A} \
    --check-values
```

预期输出：
```
Summary:
  MISSING:          0
  All parameters accounted for.

Value comparison for 195 passthrough parameters:
  All passthrough values match.
```

如果有 MISSING 参数或数值不匹配，需回到 Step 1 排查。

---

### Step 3: 在线推理 + 保存激活

运行 vllm-ascend 在线推理，自动保存指定层的输入/输出激活值。
vLLM 使用量化权重推理，不会 OOM。

```bash
# 保存单层（如最后一层 63）
python -m tools.resq_debug.run_online \
    --model ${CKPT_A} \
    --prompt "The quick brown fox jumps over" \
    --max-tokens 32 \
    --diag-layers 63 \
    --diag-dir /tmp/resq_online_acts
```

保存多层激活：
```bash
python -m tools.resq_debug.run_online \
    --model ${CKPT_A} \
    --prompt "The quick brown fox jumps over" \
    --max-tokens 32 \
    --diag-layers 0,15,31,47,63 \
    --diag-dir /tmp/resq_online_acts
```

保存所有层激活：
```bash
python -m tools.resq_debug.run_online \
    --model ${CKPT_A} \
    --diag-layers all \
    --diag-dir /tmp/resq_online_acts
```

保存的文件结构：
```
/tmp/resq_online_acts/
├── input_ids.pt                                   # tokenized input ids
├── model_layers_0_self_attn_qkv_proj.pt           # 融合 qkv 投影
├── model_layers_0_self_attn_o_proj.pt             # o 投影
├── model_layers_0_mlp_gate_up_proj.pt             # 融合 gate_up 投影
├── model_layers_0_mlp_down_proj.pt                # W8A8 down 投影
├── model_layers_63_self_attn_qkv_proj.pt
├── ...
```

每个 `.pt` 文件包含：
```python
{'prefix': str, 'input': Tensor, 'output': Tensor}
```

> **注意**：激活只在第一次真实推理时保存，通过 `_is_real_inference()` 跳过 vLLM 的预热/profile 阶段。

---

### Step 4: 在线 vs 离线对比

用保存的在线激活，喂给离线 wrapper 重新计算，对比结果。
**只按需加载单层权重，不会 OOM。**

#### 基本对比（自动检测所有已保存的层）

```bash
python -m tools.resq_debug.compare_online \
    --ckpt-a ${CKPT_A} \
    --diag-dir /tmp/resq_online_acts
```

#### 指定层对比（如最后一层）

```bash
python -m tools.resq_debug.compare_online \
    --ckpt-a ${CKPT_A} \
    --diag-dir /tmp/resq_online_acts \
    --layers 63 \
    --check-chain
```

#### 完整对比（含 MLP chain + embedding chain 检查）

```bash
python -m tools.resq_debug.compare_online \
    --ckpt-a ${CKPT_A} \
    --diag-dir /tmp/resq_online_acts \
    --check-chain \
    --check-embedding
```

**各检查项说明：**

| 检查项 | 说明 | 对应 flag |
|--------|------|-----------|
| qkv_proj | 融合 q+k+v vs 独立 wrapper | 默认 |
| o_proj | 含 column reorder 的 o_proj | 默认 |
| gate_up_proj | 融合 gate+up vs 独立 wrapper | 默认 |
| down_proj | W8A8 down_proj | 默认 |
| MLP chain | `down_proj.input == silu(gate) * up` | `--check-chain` |
| Embedding chain | `embed→RMSNorm→qkv_proj.input` (layer 0) | `--check-embedding` |
| Attention+residual | `gate_up.input == post_ln(embed + o_proj.out)` (layer 0) | `--check-embedding` |
| lm_head diagnostics | lm_head/embed_tokens/model.norm 存在性和数值 | 默认 |

预期输出：
```
Total: Passed=27, Failed=0
All checks passed.
```

---

### Step 5: CKPT_O vs CKPT_A 跨模型对比

用 CKPT_O（bf16）的完整 forward pass 激活与 CKPT_A（量化）对比，定位误差累积层。
**使用 `device_map="auto"` 自动分布到多卡，大模型不会 OOM。**

> **调试逻辑**：如果最后一层（如 63）的激活 O≈A（考虑旋转域影响），
> 则乱码问题可归因于 `model.norm.weight` 或 `lm_head.weight`。

#### 5a: 保存 CKPT_O 参考激活

```bash
python -m tools.resq_debug.run_reference \
    --model ${CKPT_O} \
    --prompt "The quick brown fox jumps over" \
    --diag-layers 63 \
    --diag-dir /tmp/resq_ref_acts
```

保存多层：
```bash
python -m tools.resq_debug.run_reference \
    --model ${CKPT_O} \
    --prompt "The quick brown fox jumps over" \
    --diag-layers 0,15,31,47,63 \
    --diag-dir /tmp/resq_ref_acts
```

保存的文件结构（bf16 模型保存独立投影，不做融合）：
```
/tmp/resq_ref_acts/
├── input_ids.pt
├── model_layers_63_self_attn_q_proj.pt
├── model_layers_63_self_attn_k_proj.pt
├── model_layers_63_self_attn_v_proj.pt
├── model_layers_63_self_attn_o_proj.pt
├── model_layers_63_mlp_gate_proj.pt
├── model_layers_63_mlp_up_proj.pt
├── model_layers_63_mlp_down_proj.pt
```

> **注意**：`run_reference.py` 和 `run_online.py` 的 `--prompt` 必须一致，否则激活对比无意义。

#### 可比性说明

两边的激活是可比的，原因：

1. **Dry run 已排除** — `resq_linear.py` 和 `w8a8_dynamic.py` 中的 `_is_real_inference()` 通过
   `get_forward_context().attn_metadata is not None` 判断，vLLM 的 profile/warmup 阶段
   `attn_metadata=None`，因此 dry run 激活不会被保存。`_acts_saved` flag 进一步保证只保存第一次。
2. **输入一致** — 两边使用相同 tokenizer，token 序列相同。`compare_reference.py` 启动时
   会检查 `input_ids.pt` 是否一致，不一致会 WARNING。
3. **无 padding 干扰** — `run_online.py` 使用 `enforce_eager=True` 且只发一条 prompt，
   prefill 阶段是纯 prompt tokens，与 HF 直接 forward 等价。

#### 5b: 保存 CKPT_A 在线激活

保存相同层、相同 prompt 的量化在线激活（与 Step 3 相同，此处为方便对照列出）：

```bash
python -m tools.resq_debug.run_online \
    --model ${CKPT_A} \
    --prompt "The quick brown fox jumps over" \
    --max-tokens 32 \
    --diag-layers 63 \
    --diag-dir /tmp/resq_online_acts
```

> **注意**：`--prompt` 必须与 5a 完全一致。

#### 5c: 对比 CKPT_O vs CKPT_A 激活

> **推荐**：提供 `--ckpt-b` 参数来启用旋转域感知对比。不提供时，旋转域中的
> 激活值只比较 norm ratio（cosine 在不同旋转域中无意义）。

带旋转矩阵的完整对比（推荐）：
```bash
python -m tools.resq_debug.compare_reference \
    --ref-dir /tmp/resq_ref_acts \
    --online-dir /tmp/resq_online_acts \
    --ckpt-b ${CKPT_B} \
    --layers 63
```

不带旋转矩阵（旋转域只比 norm ratio）：
```bash
python -m tools.resq_debug.compare_reference \
    --ref-dir /tmp/resq_ref_acts \
    --online-dir /tmp/resq_online_acts \
    --layers 63
```

自动检测所有公共层：
```bash
python -m tools.resq_debug.compare_reference \
    --ref-dir /tmp/resq_ref_acts \
    --online-dir /tmp/resq_online_acts \
    --ckpt-b ${CKPT_B}
```

**旋转域分类与对比项说明：**

ResQ 是基于旋转的量化算法，不同激活点处于不同的旋转域中。直接对比
cosine similarity 在很多情况下是无意义的，需要用 CKPT_B 中的旋转矩阵
先 unrotate 回原始域再比较。

| 对比项 | 旋转域 | 对比方式 |
|--------|--------|----------|
| Q output | **INVARIANT** | 直接 cosine + rel_err（旋转吸收入权重） |
| K output | **INVARIANT** | 直接 cosine + rel_err（旋转吸收入权重） |
| Gate+Up output | **INVARIANT** | 直接 cosine + rel_err（旋转吸收入权重） |
| down_proj input | **~INVARIANT** | 直接 cosine + rel_err（pre-Ud，SiLU(gate)*up 近似不变） |
| Layer input | Ua-ROTATED | 需 Ua 逆旋转后比较，无 CKPT_B 时只比 norm ratio |
| V output | Ub-ROTATED | 需 Ub 逆旋转后比较（per-head 旋转） |
| o_proj output | Ua-ROTATED | 需 Ua 逆旋转后比较（进入残差流） |
| MLP input | Ua-ROTATED | 需 Ua 逆旋转后比较 |
| down_proj output | Ua-ROTATED | 需 Ua 逆旋转后比较（进入残差流） |
| o_proj input | COMPLEX | 只比 norm ratio（attention 输出混合旋转域） |

**结果解读（标签含义）：**

| 标签 | 含义 |
|------|------|
| `[inv]` | 不变域，直接可比。`corr > 0.95, rel_err < 0.2` 为 PASS |
| `[unrot]` | 已通过 CKPT_B 旋转矩阵 unrotate，等价于 inv 域比较 |
| `[rot]` | 旋转域，未提供 CKPT_B。只比较 `norm_ratio ∈ (0.9, 1.1)` |

| 失败情况 | 含义 |
|----------|------|
| `[inv]` 失败 | 量化误差过大或实现 bug |
| `[unrot]` 失败 | 量化误差过大（旋转处理正确） |
| `[rot]` 失败 | norm 未保持，大概率是真正的 bug |

---

## 问题排查指南

### 如果在线推理乱码

按以下顺序排查：

1. **Checkpoint 完备性** — 运行 Step 2，确认 MISSING=0 且 passthrough 值一致
2. **逐层对比** — 运行 Step 3 + 4，从少量层开始，逐步扩展到所有层
3. **Chain check** — `--check-chain` 和 `--check-embedding` 验证非线性组件
4. **跨模型对比** — 运行 Step 5，比较 CKPT_O vs CKPT_A 最后一层激活
5. **lm_head/model.norm** — 如果最后一层激活匹配，问题在 `model.norm.weight` 或 `lm_head.weight`

### 如果 qkv_proj 或 gate_up_proj 对比失败

→ shard 合并顺序或 scale 合并有问题

### 如果 o_proj 对比失败

→ column reorder 逻辑不一致

### 如果 down_proj 对比失败

→ W8A8 matmul 路径不一致

### 如果 MLP chain 对比失败

→ silu/activation function 不一致

### 如果 embedding chain 对比失败

→ embed_tokens 权重缺失或 RMSNorm 参数错误

---

## 参数参考

### convert_resq_ckpt.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_path` | 是 | - | 转换前的原始 ResQ checkpoint 路径 |
| `--output_path` | 是 | - | 转换后输出目录 (CKPT_A) |
| `--orig_model` | 否 | - | 原始 bf16 模型路径 (CKPT_O)，用于补齐缺失的 passthrough 权重 |

### check_completeness.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt-o` | 是 | - | 原始 bf16 模型路径 (CKPT_O) |
| `--ckpt-a` | 是 | - | 转换后的 checkpoint 路径 (CKPT_A) |
| `--check-values` | 否 | false | 同时比较 passthrough 参数的实际数值 |

### run_online.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | - | CKPT_A 路径 |
| `--prompt` | 否 | `"The quick brown fox jumps over"` | 输入 prompt |
| `--max-tokens` | 否 | `32` | 最大生成 token 数 |
| `--diag-layers` | 否 | `"0"` | 保存激活的层（逗号分隔或 `all`） |
| `--diag-dir` | 否 | `/tmp/resq_online_acts` | 激活保存目录 |
| `--tp` | 否 | `1` | Tensor Parallel size |
| `--max-model-len` | 否 | `4096` | 最大上下文长度 |

### compare_online.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt-a` | 是 | - | CKPT_A 路径 |
| `--diag-dir` | 否 | `/tmp/resq_online_acts` | 在线激活保存目录 |
| `--layers` | 否 | 自动检测 | 对比的层索引（逗号分隔） |
| `--head-dim` | 否 | `128` | 注意力头维度 |
| `--check-chain` | 否 | false | 检查 MLP activation chain |
| `--check-embedding` | 否 | false | 检查 embedding + attention residual chain |

### run_reference.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | - | 原始 bf16 模型路径 (CKPT_O) |
| `--prompt` | 否 | `"The quick brown fox jumps over"` | 输入 prompt（需与 `run_online.py` 一致） |
| `--diag-layers` | 否 | `"63"` | 保存激活的层（逗号分隔或 `all`） |
| `--diag-dir` | 否 | `/tmp/resq_ref_acts` | 激活保存目录 |

### compare_reference.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--ref-dir` | 是 | - | bf16 参考激活目录（`run_reference.py` 输出） |
| `--online-dir` | 是 | - | 量化在线激活目录（`run_online.py` 输出） |
| `--ckpt-b` | 否 | - | 旋转矩阵路径 (CKPT_B)。启用旋转域感知对比：对 Ua/Ub 旋转的激活值先 unrotate 再比较 cosine。不提供时旋转域只比 norm ratio |
| `--layers` | 否 | 自动检测 | 对比的层索引（逗号分隔），默认取两目录交集 |

### debug_compare.py（小模型适用，大模型会 OOM）

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | - | 原始 bf16 模型路径 (CKPT_O) |
| `--ckpt-a` | 是 | - | CKPT_A 路径 |
| `--ckpt-b` | 否 | - | 旋转矩阵 checkpoint 路径 (CKPT_B) |
| `--layers` | 否 | `all` | 对比的层（`all` 或逗号分隔如 `0,31,63`） |
| `--device` | 否 | `cpu` | 运行设备（`cpu`/`npu`） |
| `--true-quant` | 否 | false | 使用 NPU 真量化算子 |
| `--seq-len` | 否 | `4` | 输入序列长度 |
| `--head-dim` | 否 | `128` | 注意力头维度 |
| `--no-uc` | 否 | false | 禁用 Uc 旋转 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RESQ_DIAG_LAYERS` | (空) | 保存激活的层索引（逗号分隔或 `all`，空=不保存） |
| `RESQ_DIAG_DIR` | `/tmp/resq_online_acts` | 激活保存目录 |

> `run_online.py` 会根据 `--diag-layers` 和 `--diag-dir` 参数自动设置这两个环境变量。
