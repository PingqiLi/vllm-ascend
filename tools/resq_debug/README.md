# ResQ Debug Tools

用于调试 ResQ 量化模型推理问题的激活值比较工具。

## 背景

当 vllm-ascend 使用 ResQ 量化 checkpoint (ckpt A) 推理输出乱码时，需要逐层比较量化模型与原始模型的中间激活值来定位问题。

ResQ 算法中存在旋转矩阵（Ua、Ub），导致量化模型与原始模型的激活值不在同一空间，不能直接比较。需要借助 ckpt B 中的旋转矩阵做 un-rotate 后才能比较。

## 文件说明

| 文件 | 说明 |
|------|------|
| `wrappers.py` | `ResQLinear[TrueQuant]Wrapper` 和 `W8A8Linear[TrueQuant]Wrapper` |
| `debug_compare.py` | 主工具脚本（wrapper 方式，不依赖自定义模型定义） |
| `compare.py` | 旧版比较工具（依赖 `Qwen3ResQForCausalLM` 等自定义模型定义） |

## Checkpoint 说明

| 名称 | 内容 |
|------|------|
| ckpt O | 原始 Qwen3 bf16 模型 |
| ckpt A | ResQ 量化模型（weight_high/low、scale、Uc、Pd、Hd 等） |
| ckpt B | 辅助旋转矩阵（P_a、R_a、P_b、R_b → 用于计算 Ua、Ub） |

## 量化模式

| 模式 | 说明 | 运行设备 |
|------|------|----------|
| Fake quant | load 时反量化为 float，forward 做 float matmul | CPU / GPU |
| **True quant** | 保持 int4/int8 权重，forward 用 `npu_dynamic_quant` + `npu_quant_matmul` | **NPU** |

True quant 完全复现 vllm-ascend 生产代码的量化计算路径，用于定位 NPU 推理问题。

## 使用方法

### 推荐：一条命令完成（`run`）

```bash
python -m tools.resq_debug.debug_compare run \
    --model ${CKPT_O} \
    --ckpt-a ${CKPT_A} \
    --ckpt-b ${CKPT_B} \
    --layers 0,1,2 \
    --device npu \
    --true-quant
```

流程：
1. 加载原始模型 → 前向传播 → 捕获原始激活值
2. 替换线性层为 wrapper → 前向传播 → 捕获 ResQ 激活值
3. 内存中直接比较（无需中间文件）

可选保存中间结果：

```bash
python -m tools.resq_debug.debug_compare run \
    --model ${CKPT_O} \
    --ckpt-a ${CKPT_A} \
    --ckpt-b ${CKPT_B} \
    --device npu --true-quant \
    --save-orig orig.pt \
    --save-resq resq.pt
```

### 分步执行

#### 1. 保存原始模型激活值

```bash
python -m tools.resq_debug.debug_compare save-orig \
    --model ${CKPT_O} --out orig.pt --device cpu
```

#### 2. 保存 ResQ 模型激活值

True quant（NPU）：

```bash
python -m tools.resq_debug.debug_compare save-resq \
    --model ${CKPT_O} --ckpt-a ${CKPT_A} --out resq.pt \
    --device npu --true-quant
```

Fake quant（CPU）：

```bash
python -m tools.resq_debug.debug_compare save-resq \
    --model ${CKPT_O} --ckpt-a ${CKPT_A} --out resq.pt \
    --device cpu
```

#### 3. 比较激活值

```bash
python -m tools.resq_debug.debug_compare compare \
    --orig orig.pt --resq resq.pt --ckpt-b ${CKPT_B}
```

不提供 `--ckpt-b` 时，旋转空间的激活值只能比较 norm。

## Wrapper 替换规则

| 原始层 | Wrapper (fake quant) | Wrapper (true quant) |
|--------|---------------------|---------------------|
| q/k/v/o_proj, gate/up_proj | `ResQLinearWrapper` | `ResQLinearTrueQuantWrapper` |
| down_proj | `W8A8LinearWrapper` | `W8A8LinearTrueQuantWrapper` |

额外处理：
- **o_proj**：列重排（column reorder）
- **down_proj**：Ud 旋转（Pd + Hd Hadamard）
- **q_norm / k_norm**：可选 Uc 旋转 hook（`--no-uc` 禁用）

## 比较输出说明

```
  L0.q [inv]: corr=0.9998, rel_err=0.0012, norm_ratio=1.0001   # 通过
X L0.v [rot]: norm_ratio=1.5432 (expected ~1.0)                 # 失败
```

| 标签 | 含义 |
|------|------|
| `[inv]` | 旋转不变点或已 un-rotate，比较 rel_err 和 cosine similarity |
| `[rot]` | 旋转空间（未提供 ckpt B），只能比较 norm ratio |

通过阈值：`[inv]` rel_err < 0.15 且 corr > 0.95；`[rot]` norm_ratio 偏差 < 0.1。

## 旋转空间分析

| 激活 | 空间 | 比较方式 |
|------|------|----------|
| q, k, gate, up | 原始空间（不变点） | 直接比较 |
| v | Ub 空间 | 需 `@ Ub.T` un-rotate |
| input_ln, o, down | Ua 空间 | 需 `@ Ua.T` un-rotate |

## 参数参考

### run / save-orig / save-resq 共用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layers` | `all` | 捕获的层，`all` 或逗号分隔如 `0,31,63` |
| `--seq-len` | `4` | 输入序列长度 |
| `--prompt` | `default` | 输入文本 |
| `--head-dim` | `128` | 注意力头维度 |

### run / save-resq 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--true-quant` | `false` | 使用 NPU 真量化算子 |
| `--no-uc` | `false` | 禁用 Uc 旋转 |
| `--device` | run: `npu`, save-resq: `cpu` | 运行设备 |

### run 独有参数

| 参数 | 说明 |
|------|------|
| `--save-orig` | 可选，保存原始激活到文件 |
| `--save-resq` | 可选，保存 ResQ 激活到文件 |