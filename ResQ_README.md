# vllm-ascend ResQ 推理适配技术文档

## 1. 概述

本文档说明 vllm-ascend 为支持 ResQ 混合精度量化模型推理所做的适配工作。

ResQ 是一种 4/8-bit 混合精度量化方法，量化工具侧（msit/msmodelslim）将旋转矩阵融合进权重后，导出 `[weight_low (int4) | weight_high (int8)]` 分体权重。推理侧需要：

1. 正确加载分体权重与 per-channel scale
2. 对 o_proj 执行列重排，还原量化工具侧的 per-head `[low | high]` 布局
3. 对 down_proj 执行在线旋转（perm_rd 模式：per-group split + 块 Hadamard）
4. 调用 NPU 混合精度算子 `npu_mixprecise_quant_matmul` 完成推理
5. 适配 vLLM 的 Tensor Parallel 分片逻辑

---

## 2. 文件结构

```
vllm_ascend/quantization/
├── quant_config.py      # AscendQuantConfig：模型量化类型路由
├── utils.py             # 量化方法注册表 + per-layer 类型解析
└── resq_linear.py       # ResQLinearMethod：ResQ 推理核心实现
```

### 各文件职责

**`quant_config.py`**：
- 注册 `AscendQuantConfig` 为 vLLM 量化配置
- 识别 `model_quant_type == "RESQ"` 后绕过 `is_layer_skipped_ascend` 检查，直接路由到 `AscendLinearMethod`
- 定义 Qwen3 的 packed modules 映射（`qkv_proj → [q, k, v]`，`gate_up_proj → [gate, up]`）
- `AscendLinearMethod.create_weights()` 检测到内部 quant method 有 `create_weights` 时委托给 `ResQLinearMethod`

**`utils.py`**：
- `ASCEND_QUANTIZATION_METHOD_MAP` 中注册 `"RESQ": {"linear": ResQLinearMethod}`
- `get_quant_method()` 对 RESQ 模型做 per-layer 分发：`quant_model_description.json` 中标记为 `"RESQ"` 的层用 `ResQLinearMethod`，标记为 `"W8A8_DYNAMIC"` 的层走标准 W8A8 路径
- `_get_layer_quant_type()` 同时支持标准 `.weight` key 和 ResQ 的 `.weight_low` key

**`resq_linear.py`**：ResQ 推理的完整实现，见下文详述。

---

## 3. ResQLinearMethod 详解

### 3.1 权重注册（`create_weights`）

为每个 ResQ 层注册以下参数：

| 参数 | dtype | 说明 |
|------|-------|------|
| `weight_low` | int8 | 4-bit 量化权重（加载时为 int8，后续 pack 为 int4） |
| `weight_high` | int8 | 8-bit 量化权重 |
| `scale_low` | float32 | 4-bit 部分 per-channel scale |
| `scale_high` | float32 | 8-bit 部分 per-channel scale |
| `high_fraction` | float32 | 8-bit 通道占比（标量） |

**down_proj 额外参数**：

| 参数 | dtype | 说明 |
|------|-------|------|
| `rd_block_size` | int32 | 块 Hadamard 大小（标量，perm_rd 模式） |
| `perm_group_size` | int32 | per-group 划分大小（标量，= intermediate_size / max_tp） |

**o_proj 额外 buffer**：`o_proj_column_order`（列重排索引）。

### 3.2 权重加载（`weight_loader` + `_merge_shards`）

vLLM 对 packed modules（如 `qkv_proj`）会分 shard 加载（q=0, k=1, v=2）。`weight_loader` 将各 shard 暂存到 `param._shards` 字典，`_merge_shards` 在 `process_weights_after_loading` 阶段按 q→k→v 顺序沿 dim=0 拼接：

```python
# 排序: q(0) → k(1) → v(2)
sorted_keys = sorted(param._shards.keys(), key=shard_sort_key)
merged = torch.cat([param._shards[k] for k in sorted_keys], dim=0)
```

对标量参数（`high_fraction`, `rd_block_size`, `perm_group_size`）直接取第一个 shard 的值。

### 3.3 权重后处理（`process_weights_after_loading`）

加载完成后依次执行：

```
_merge_shards()          # 拼接 shard
  ↓
_setup_o_proj_column_order()  # [仅 o_proj] 计算列重排索引
_setup_perm_rd()              # [仅 down_proj] 提取 block_size，预计算 Hadamard
  ↓
int4 packing             # weight_low: int8 → pack 成 quint4x2
scale 转换               # float32 → uint64-packed（NPU 算子要求）
  ↓
_tp_slice()              # Tensor Parallel 切分
  ↓
NZ format cast           # [可选] 转为 FRACTAL_NZ 格式
```

#### int4 打包（`_pack_int4_to_int8_signed`）

将 `int8` 值域 `[-8, 7]` 的 tensor 按列两两打包为一个 `int8`：

```python
x_unsigned = where(x < 0, x + 16, x)   # 转为无符号 [0, 15]
packed = low_nibble | (high_nibble << 4)
# shape: [K, N] → [K, N/2]
```

#### scale 转换（`_convert_scales`）

NPU `npu_mixprecise_quant_matmul` 要求 scale 以 `uint64` 格式传入：

```python
# float32 [N] → view as uint32 → 填入 uint64 偶数位 → view as int64
scale_u64[::2] = scale_u32
# shape: [1, N] → [1, 2N] (as uint32) → [1, N] (as int64)
```

### 3.4 o_proj 列重排（`_setup_o_proj_column_order`）

**背景**：量化工具侧对 o_proj 做了 per-head 列重排，将每个 head 的高精度列移到末尾，形成全局 `[all_low | all_high]` 布局。推理时 o_proj 的输入（即 attention output）仍是原始 per-head 布局，需要在线重排激活以匹配权重。

```python
# 原始: [head0: 128 cols, head1: 128 cols, ...]
# 每头最后 high_per_head 列是高精度
# 重排为: [head0_low, head1_low, ... | head0_high, head1_high, ...]

high_per_head = in_high // num_heads   # 从权重形状反推，避免对齐误差
order = _get_column_reorder(in_dim, head_dim=128, high_per_head)
```

`apply()` 中在 matmul 之前对激活执行重排：
```python
x = x[..., layer.o_proj_column_order]
```

### 3.5 perm_rd 在线旋转（`_setup_perm_rd` + `_apply_perm_rd`）

**背景**：量化工具侧在 perm_rd 模式下，将 MassDiff 置换和块 Hadamard 旋转融合进了 gate/up/down_proj 的权重。推理时 down_proj 的输入（SwiGLU 输出）需要在线执行逆变换：per-group split → 块 Hadamard。

#### 初始化

```python
def _setup_perm_rd(layer):
    # 从 checkpoint 读取 rd_block_size 和 perm_group_size
    block_size = layer.rd_block_size.item()      # e.g. 32
    group_size = layer.perm_group_size.item()     # e.g. 12800

    # 预计算归一化 Hadamard 矩阵 H_b [block_size, block_size]
    H_b = hadamard_transform(eye(block_size)) / sqrt(block_size)
    layer.h_block = H_b.npu()
```

#### 在线执行

```python
def _apply_perm_rd(layer, x_2d, in_high):
    # x_2d: [M, total_dim]  e.g. [M, 25600]

    # 1. 按 group 划分
    #    [M, 25600] → [M, num_groups, group_size]
    #    e.g. [M, 2, 12800]
    x_grouped = x_2d.view(M, num_groups, group_size)

    # 2. 每组内 split: 前 low_per_group 列 → 4-bit, 后 high_per_group 列 → 8-bit
    x_low  = x_grouped[:, :, :low_per_group]     # [M, 2, 11200]
    x_high = x_grouped[:, :, low_per_group:]      # [M, 2, 1600]

    # 3. 跨组展平
    x_low  = x_low.reshape(M, -1)                # [M, 22400]
    x_high = x_high.reshape(M, -1)               # [M, 3200]

    # 4. 对 low 和 high 分别做块 Hadamard
    x_low  = block_hadamard(x_low,  block_size, H_b)
    x_high = block_hadamard(x_high, block_size, H_b)

    return x_low, x_high
```

**为什么先 split 再 Hadamard**：量化工具侧融合权重时的顺序是 `W = Ua.T @ W @ Perm @ Rd`，然后 `rearrange_columns` 将 per-group 的 `[low | high]` gather 成全局 `[all_low | all_high]`。推理时激活需要做逆操作：先按 group 取出 `[low | high]`，再对各部分施加 Hadamard（Rd 是正交矩阵，`Rd.T = Rd` 对称）。

#### 块 Hadamard 实现

```python
def _block_hadamard(x, block_size, h_matrix):
    # x: [M, D], D % block_size == 0
    # h_matrix: [block_size, block_size] (预计算)
    M, D = x.shape
    return (x.view(M, D // block_size, block_size) @ h_matrix).view(M, D)
```

`h_matrix` 通过 butterfly 快速 Hadamard 变换生成后归一化：

```python
def _hadamard_transform(u):
    """Fast Hadamard transform (butterfly, unnormalized)."""
    n = u.shape[-1]
    x = u.clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2*h), 2, h)
        a, b = x[:,:,0,:], x[:,:,1,:]
        x = stack([a+b, a-b], dim=2).view(-1, n)
        h *= 2
    return x

H_b = _hadamard_transform(eye(block_size)) / sqrt(block_size)
```

### 3.6 Tensor Parallel 适配

ResQ 的权重是分体存储的（`weight_low` 和 `weight_high` 独立 tensor），TP 切分需要对两部分分别操作。

#### 列并行层（qkv_proj, gate_up_proj）— 按输出维度切

权重形状 `[N, K]`，scale 形状 `[1, N]`（uint64-packed），沿 N 维切分。

对 packed modules 需特殊处理交错切片：
```python
# gate_up_proj: [gate_rows | up_rows]
# 每部分独立按 TP 切，再拼接
# qkv_proj: [q_rows | k_rows | v_rows]
# 同理
```

#### 行并行层（o_proj, down_proj）— 按输入维度切

权重形状 `[N, K]`，沿 K 维切分：
- `weight_low_packed`：按 packed 列数切（int4 已 pack，列数 = 原始列数/2）
- `weight_high`：按原始列数切
- scale 不变（per-row，与输入维度无关）

### 3.7 前向推理（`apply`）

完整流程：

```
输入 x: [batch, seq_len, in_dim]
      ↓
[o_proj] 列重排: x = x[..., column_order]
      ↓
reshape → x_2d: [M, in_dim]
      ↓
[down_proj + perm_rd] _apply_perm_rd → x_low, x_high
[其他层] 直接 split: x_low = x[:, :in_low], x_high = x[:, in_low:]
      ↓
动态量化:
  x_low  → npu_dynamic_quant(quint4x2) → x_low_quant, low_act_scale
  x_high → npu_dynamic_quant(int8)      → x_high_quant, high_act_scale
      ↓
混合精度矩阵乘法:
  npu_mixprecise_quant_matmul(
      x_low_quant,  weight_low_packed.T,
      rx=x_high_quant, hweight=weight_high.T,
      lscale, hscale, lper_token_scale, rper_token_scale,
      split_kpos=in_low
  )
      ↓
输出: [batch, seq_len, out_dim] (bfloat16)
```

关键 NPU 算子：
- `torch_npu.npu_dynamic_quant`：per-token 动态量化（输出 quint4x2 或 int8）
- `torch_npu.npu_mixprecise_quant_matmul`：混合精度量化矩阵乘法，接受 4-bit 和 8-bit 两组输入，`split_kpos` 指定分界位置

---

## 4. Checkpoint 格式要求

### 4.1 `quant_model_description.json`

```json
{
  "model_quant_type": "ResQ",
  "model.layers.0.self_attn.q_proj.weight_low": "RESQ",
  "model.layers.0.self_attn.q_proj.scale_low": "RESQ",
  "model.layers.0.self_attn.q_proj.weight_high": "RESQ",
  "model.layers.0.self_attn.q_proj.scale_high": "RESQ",
  "model.layers.0.self_attn.q_proj.high_fraction": "RESQ",
  "model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC",
  "..."
}
```

- `model_quant_type: "ResQ"` 触发 RESQ 路由
- 标记为 `"RESQ"` 的层由 `ResQLinearMethod` 处理
- 标记为 `"W8A8_DYNAMIC"` 的层走标准 W8A8 路径（如自适应模式下部分 down_proj）

### 4.2 ResQ 层的张量

```
model.layers.{i}.self_attn.q_proj.weight_low      # int8, [out, low_dim]
model.layers.{i}.self_attn.q_proj.weight_high      # int8, [out, high_dim]
model.layers.{i}.self_attn.q_proj.scale_low        # float32, [out, 1]
model.layers.{i}.self_attn.q_proj.scale_high       # float32, [out, 1]
model.layers.{i}.self_attn.q_proj.high_fraction    # float32 标量
```

### 4.3 perm_rd 模式 down_proj 额外张量

```
model.layers.{i}.mlp.down_proj.rd_block_size       # int32 标量 (e.g. 32)
model.layers.{i}.mlp.down_proj.perm_group_size     # int32 标量 (e.g. 12800)
```

当 checkpoint 中不含这两个字段时，自动退化为简单的 `[low | high]` split（向后兼容）。

---

## 5. 使用方式

```bash
vllm serve /path/to/resq-quantized-model --quantization ascend
```

确保模型目录包含：
- `model.safetensors`（量化权重）
- `quant_model_description.json`（`model_quant_type: "ResQ"`）
- `config.json`（模型配置）
