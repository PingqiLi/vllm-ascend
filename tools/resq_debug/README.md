# ResQ Debug Tools

比较原始 Qwen3 模型与 ResQ 量化模型的激活值。

## Checkpoint 格式

| Checkpoint | 内容 | 用途 |
|------------|------|------|
| **CKPT_O** | 原始 bf16 权重 | 原始模型推理 |
| **CKPT_A** | 量化权重 + Uc/Pd (在线旋转) | ResQ 模型推理 |
| **CKPT_B** | P_a/R_a/P_b/R_b/P_c/R_c/P_d/R_d_hadK | 激活值 un-rotate |

## CKPT_B 旋转矩阵

```
resq.layer.*.P_a, R_a → Ua = P_a @ R_a  [hidden_size, hidden_size]
resq.layer.*.P_b, R_b → Ub = P_b @ R_b  [num_kv_heads, head_dim, head_dim]
resq.layer.*.P_c, R_c → Uc = P_c @ R_c  [head_dim, head_dim] (在线应用)
resq.layer.*.P_d, R_d_hadK → Ud         [blocksize, blocksize] (在线应用)
```

## 用法

```bash
# 1. 保存原始模型激活 (只跑 layer 0，启用早停加速)
python -m tools.resq_debug.compare save-orig \
    --model ${CKPT_O} --out ./cmp/orig.pt --layers 0

# 2. 保存 ResQ 模型激活 (伪量化，只跑 layer 0)
python -m tools.resq_debug.compare save-resq \
    --ckpt-a ${CKPT_A} --out ./cmp/resq.pt --layers 0

# 3. 保存 ResQ 模型激活 (真量化，只跑 layer 0)
python -m tools.resq_debug.compare save-resq-true \
    --ckpt-a ${CKPT_A} --out ./cmp/resq_true.pt --layers 0 --device npu

# 4. 比较 (可选 CKPT_B 做 un-rotate)
python -m tools.resq_debug.compare compare \
    --orig ./cmp/orig.pt --resq ./cmp/resq.pt --ckpt-b ${CKPT_B}
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layers` | `all` | 保存哪些层: `all`, `0`, `0,1,2` 等 |
| `--seq-len` | `4` | 输入序列长度 |
| `--prompt` | `default` | 输入文本，`default` 使用内置文本 |
| `--device` | `cpu` | 运行设备 |

### 早停加速

当 `--layers` 不是 `all` 时，前向传播会在最后一个指定层之后提前终止：

```bash
# 只跑 layer 0 的前向，跳过 layer 1~63 (大幅加速)
--layers 0

# 跑 layer 0,1,2 的前向，跳过 layer 3~63
--layers 0,1,2
```

**注意**: 早停时不会保存 `logits`（因为没跑到 lm_head）。

## 激活比较逻辑

compare 模式会比较两个 .pt 文件中**共有的 keys**，支持只比较部分层：

```bash
# 两个文件都只有 layer 0 的激活，也能正常比较
python -m tools.resq_debug.compare compare \
    --orig orig_L0.pt --resq resq_L0.pt
```

**有 CKPT_B 时**:
- 计算 `Ua = P_a @ R_a`, `Ub = P_b @ R_b`
- 对 input_ln/gate/up 应用 `x @ Ua.T` 逆旋转
- 对 v 应用 `x @ Ub.T` 逆旋转 (per-kv-head)
- 然后直接比较 rel_err 和 corr

**无 CKPT_B 时**:
- 旋转不变点 `[inv]`: q, k, o, down, logits → 直接比较
- 旋转空间 `[rot]`: input_ln, v, gate, up → 只比较 norm

## 融合公式

```
embed_A = embed_O @ Ua               # 输出在 Ua 空间
Q_A = Q_O @ Ua                       # 抵消输入 Ua
K_A = K_O @ Ua                       # 抵消输入 Ua  
V_A = Ub.T @ V_O @ Ua                # 输出在 Ub 空间
O_A = Ua.T @ O_O @ Ub                # 抵消 Ub，输出回原始空间

q_resq = q_orig                      # Ua 抵消
k_resq = k_orig                      # Ua 抵消
v_resq = v_orig @ Ub                 # 在 Ub 空间
o_resq = o_orig                      # Ub 抵消
```
