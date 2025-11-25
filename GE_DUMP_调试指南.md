# Qwen3-30B-A3B W4A4 图模式精度问题调试指南

## 问题描述

**现象**：使用 vLLM serve 部署 Qwen3-30B-A3B W4A4 模型后，连续发送多个相同的 "Hello!" 请求：
- **奇数次请求**（第 1、3、5... 次）：输出正常
- **偶数次请求**（第 2、4、6... 次）：输出全是 "!!!!"（token id = 0）

**已排除**：
- Prefix cache 影响（已通过 `--no-enable-prefix-cache` 排除）
- Eager 模式正常，问题仅出现在 TorchAir 图模式下

**目标**：通过 GE dump 比对奇数次（正常）和偶数次（异常）推理的中间层激活值，定位问题根源。

---

## 一、环境准备

### 1.1 安装 msit 工具

在服务器上安装 msit 工具包（用于 GE dump 和精度比对）：

```bash
# 方法 1：从 PyPI 安装（如果可用）
pip install msit

# 方法 2：从 gitcode 克隆安装
git clone https://gitcode.com/Ascend/msit.git
cd msit
pip install -e .
```

**验证安装**：
```bash
msit --version
python -c "from msit_llm.dump import torchair_dump; print('msit_llm installed')"
```

### 1.2 检查依赖

确保环境中已安装：
- `torch_npu`
- `torchair`
- `vllm` 和 `vllm-ascend`

---

## 二、代码修改说明

### 2.1 已修改的文件

**文件**：`vllm_ascend/torchair/torchair_model_runner.py`

**修改内容**：

1. **在 `__init__` 方法中添加 dump 配置**（第 85-92 行）：
   - `self.dump_counter = 0`：推理计数器
   - `self.dump_enabled`：是否启用 dump
   - `self.dump_base_path`：临时 dump 目录
   - `self.dump_max_requests`：最多 dump 多少次请求

2. **在 `_get_torchair_lazy_compiled_model` 方法中配置 GE dump**（第 418-442 行）：
   - 使用 `msit_llm.dump.torchair_dump.get_ge_dump_config()` 配置
   - **只 dump 第 0 个 token**（首个生成 token）
   - **只 dump MoE 相关算子**：MatMul、量化矩阵乘、AllToAll 等

3. **在 `_generate_process_reqs_hidden_states` 方法中处理 dump 数据**（第 378-380 行）：
   - 每次 decode 推理后，将 dump 数据移动到 `run_1`、`run_2` ... 目录
   - 计数器自动递增
   - 达到 `dump_max_requests` 后自动停止 dump

4. **新增 `_move_dump_data_to_counter_dir` 方法**（第 396-439 行）：
   - 将 `{dump_base_path}/msit_ge_dump` 移动到 `run_{N}/msit_ge_dump`
   - 清空临时目录，为下一次 dump 做准备

### 2.2 环境变量控制

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `VLLM_ASCEND_DUMP_ENABLED` | 是否启用 dump（**必填**） | `0` | `1` |
| `VLLM_ASCEND_DUMP_PATH` | 临时 dump 目录路径 | `./dump_base` | `./dump_temp` |
| `VLLM_ASCEND_DUMP_MODE` | dump 模式：`input`、`output`、`all` | `output` | `output` |
| `VLLM_ASCEND_DUMP_MAX_REQUESTS` | 最多 dump 多少次请求 | `10` | `10` |

**注意**：
- **`VLLM_ASCEND_DUMP_ENABLED=1` 是启用 dump 的开关**
- dump 数据最终保存在与 `VLLM_ASCEND_DUMP_PATH` 同级的 `run_1`、`run_2` ... 目录中
- 只 dump **decode 阶段**的推理（prefill 阶段不 dump）

---

## 三、执行 Dump

### 3.1 一次性 Dump 多次推理

**优势**：
- **启动一次 vLLM serve**，发送多次请求，自动 dump 到不同目录
- 无需每次重启服务

**步骤**：

```bash
# 1. 设置环境变量启用 dump
export VLLM_ASCEND_DUMP_ENABLED=1
export VLLM_ASCEND_DUMP_PATH="./dump_temp"  # 临时目录
export VLLM_ASCEND_DUMP_MODE="output"       # 只 dump 输出，减少数据量
export VLLM_ASCEND_DUMP_MAX_REQUESTS=10     # 最多 dump 10 次请求

# 2. 启动 vLLM 服务
vllm serve /path/to/qwen3-30b-a3b-w4a4 \
    --trust-remote-code \
    --dtype float16 \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2048 \
    --quantization w4a4_flatquant_dynamic \
    --enforce-eager=false  # 确保使用图模式

# 3. 在另一个终端发送多次请求（例如 10 次）
for i in {1..10}; do
    echo "Request $i:"
    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "/path/to/qwen3-30b-a3b-w4a4",
        "prompt": "Hello!",
        "max_tokens": 20,
        "temperature": 0
      }'
    echo ""
    sleep 0.5  # 稍微延迟，确保 dump 数据写入完成
done

# 4. 查看生成的 dump 目录
ls -l
# 应该看到：run_1/, run_2/, run_3/, ..., run_10/, dump_temp/

# 5. 停止 vLLM 服务
```

**预期结果**：
- 生成 `run_1/` 到 `run_10/` 共 10 个目录
- 每个目录下都有 `msit_ge_dump/` 子目录，包含对应请求的 dump 数据
- 第 1、3、5、7、9 次应该输出正常
- 第 2、4、6、8、10 次应该输出 "!!!!"

### 3.2 Dump 数据目录结构

```
./
├── dump_temp/              # 临时目录（空，已被清理）
├── run_1/                  # 第 1 次请求（正常）
│   └── msit_ge_dump/
│       ├── dynamo_optimized_*.txt
│       ├── dynamo_original_*.txt
│       └── worldsize*_global_rank*/
│           └── <timestamp>/
│               └── <device_id>/
│                   └── <model_name>/
│                       └── <model_id>/
│                           └── 0/       # token_id = 0
│                               └── *.bin
├── run_2/                  # 第 2 次请求（异常）
│   └── msit_ge_dump/
│       └── （同上）
├── run_3/                  # 第 3 次请求（正常）
│   └── msit_ge_dump/
├── ...
└── run_10/                 # 第 10 次请求（异常）
    └── msit_ge_dump/
```

---

## 四、精度比对

### 4.1 比对第 1 次（正常）和第 2 次（异常）

```bash
# 比对 run_1（正常）和 run_2（异常）
msit llm compare \
  --golden-path ./run_1/msit_ge_dump \
  --my-path ./run_2/msit_ge_dump \
  --output ./comparison_1vs2

# 查看比对结果
cat ./comparison_1vs2/compare_result.csv
```

### 4.2 多组比对（验证规律）

```bash
# 比对 run_3（正常）和 run_4（异常）
msit llm compare \
  --golden-path ./run_3/msit_ge_dump \
  --my-path ./run_4/msit_ge_dump \
  --output ./comparison_3vs4

# 比对 run_5（正常）和 run_6（异常）
msit llm compare \
  --golden-path ./run_5/msit_ge_dump \
  --my-path ./run_6/msit_ge_dump \
  --output ./comparison_5vs6

# 查看结果，验证是否每次都是相同的算子发散
diff ./comparison_1vs2/compare_result.csv ./comparison_3vs4/compare_result.csv
```

### 4.3 分析比对结果

**CSV 列含义**（参考[精度比对结果参数说明](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/精度比对结果参数说明.md)）：

| 列名 | 说明 | 正常范围 |
|------|------|----------|
| `operator_name` | 算子名称 | - |
| `cosine_similarity` | 余弦相似度 | > 0.99（越接近 1 越好） |
| `max_abs_error` | 最大绝对误差 | 越小越好 |
| `mean_abs_error` | 平均绝对误差 | 越小越好 |

**定位方法**：
1. **按 `cosine_similarity` 从小到大排序**，找到相似度低的算子
2. **定位第一个 `cosine_similarity < 0.99` 的算子**
3. **检查算子类型**：
   - `QuantBatchMatmul`、`QuantMatmul` → W4A4 量化问题
   - `AllToAll`、`AllGather` → MoE 专家通信问题
   - `MatMul`、`MatMulV2` → 权重或激活问题

**示例分析**：

假设比对结果显示：
```csv
operator_name,cosine_similarity,max_abs_error,mean_abs_error
model.layers.0.self_attn.q_proj.MatMul,1.0000,0.0001,0.0000
model.layers.0.mlp.gate_proj.QuantMatmul,0.9876,0.5234,0.0123  ← 第一个发散点
model.layers.0.mlp.up_proj.QuantMatmul,0.7234,1.2345,0.2345  ← 误差传播
...
```

**结论**：
- **问题算子**：`model.layers.0.mlp.gate_proj.QuantMatmul`（MoE 的 gate 投影）
- **可能原因**：
  1. **W4A4 量化的 gate 权重在第 2 次推理时被污染**
  2. **激活量化的 scale 参数在图模式下未正确重置**
  3. **MoE 路由权重在图缓存重用时有状态残留**

---

## 五、根据比对结果定位代码

### 5.1 检查 W4A4 量化实现

**文件**：`vllm_ascend/quantization/w4a4_flatquant_dynamic.py`

**关键点**：
1. 权重转置是否正确（第 89 行和第 167 行）：
   ```python
   # 第 89 行
   self.transpose_weight = True  # 确认是 True

   # 第 167 行
   layer.weight_packed  # 确认没有 .t() 调用
   ```

2. 激活量化 scale 是否正确管理：
   ```python
   # 第 126-138 行：激活量化
   x_quantized, pertoken_scale = quantize_per_token_dynamic(
       x_reshaped, torch.int8, torch.float32, self.sym
   )
   # 检查 pertoken_scale 是否每次推理都重新计算，没有复用旧值
   ```

3. 权重 scale 是否被修改：
   ```python
   # 添加调试日志
   logger.debug(f"weight_scale hash: {hash(layer.weight_scale.data_ptr())}")
   logger.debug(f"weight_scale mean: {layer.weight_scale.mean()}")
   ```

### 5.2 检查 MoE 实现

**文件**：查找 MoE 相关实现

```bash
# 搜索 MoE 相关代码
grep -r "class.*MoE" vllm_ascend/
grep -r "AllToAll" vllm_ascend/
```

**关键检查点**：
1. 专家路由权重是否正确加载
2. 专家选择逻辑是否有状态残留
3. AllToAll 通信是否正确同步

### 5.3 检查图缓存

**可能原因**：TorchAir 图缓存导致第 2 次推理复用了错误的状态

**排查方法**：
```bash
# 清除图缓存后重新测试
rm -rf ~/.cache/torch_npu/torchair_cache/

# 或在启动时禁用图缓存
# 修改 vllm 配置：use_cached_graph=False
```

---

## 六、进阶调试方法

### 6.1 调整 dump 参数

**dump 所有算子**（如果 MoE 相关算子不够）：

修改 `torchair_model_runner.py` 第 424-428 行，注释掉 `dump_layer` 参数：
```python
torchair_dump.get_ge_dump_config(
    dump_path=self.dump_base_path,
    dump_mode=self.dump_mode,
    dump_token=[0],
    # dump_layer=dump_layers,  # 注释掉，dump 所有算子
    compiler_config=config
)
```

**注意**：dump 所有算子会产生大量数据，请确保磁盘空间充足。

### 6.2 dump 多个 token

修改第 432 行：
```python
dump_token=[0, 1, 2],  # dump 前 3 个 token
```

### 6.3 关闭融合进行比对

如果怀疑算子融合导致问题，可以关闭融合：

1. 使用提供的 `fusion_switch.json`
2. 修改 `torchair_model_runner.py` 第 429 行，添加 `fusion_switch_file` 参数：
   ```python
   fusion_switch_file = os.environ.get('VLLM_ASCEND_FUSION_SWITCH_FILE', None)
   torchair_dump.get_ge_dump_config(
       dump_path=self.dump_base_path,
       dump_mode=self.dump_mode,
       dump_token=[0],
       dump_layer=dump_layers,
       fusion_switch_file=fusion_switch_file,  # 添加这一行
       compiler_config=config
   )
   ```
3. 设置环境变量：
   ```bash
   export VLLM_ASCEND_FUSION_SWITCH_FILE="./fusion_switch.json"
   ```

---

## 七、常见问题排查

### 7.1 msit_llm 导入失败

**现象**：
```
WARNING: msit_llm not installed, GE dump disabled
```

**解决**：
```bash
pip install msit
# 或
cd /path/to/msit && pip install -e .
```

### 7.2 Dump 数据未生成

**可能原因**：
1. `VLLM_ASCEND_DUMP_ENABLED` 未设置为 `1`
2. 权限问题，无法写入目标目录
3. 模型推理报错，未到达 decode 阶段

**检查**：
```bash
# 查看日志中是否有 "GE dump enabled" 和 "GE dump configured"
grep "GE dump" <vllm_log_file>

# 查看是否有 "Moved dump data to run_X"
grep "Moved dump data" <vllm_log_file>

# 检查目录权限
ls -la ./dump_temp/
```

### 7.3 Dump 数据被覆盖

**现象**：只看到 `run_1/`，后续的 `run_2/` 等没有生成

**可能原因**：
- `_move_dump_data_to_counter_dir` 方法执行失败
- 文件移动时发生错误

**检查**：
```bash
# 查看日志中的错误信息
grep "Failed to move dump data" <vllm_log_file>
```

### 7.4 比对时找不到映射

**现象**：
```
WARNING: No mapping found for operator xxx
```

**原因**：不同推理的图结构可能略有差异（不太可能）

**解决**：
```bash
# 使用 -l debug 查看详细信息
msit llm compare \
  --golden-path ./run_1/msit_ge_dump \
  --my-path ./run_2/msit_ge_dump \
  --output ./comparison_1vs2 \
  -l debug
```

### 7.5 Dump 数据量过大

**优化方法**：
1. **只 dump 输出**：`export VLLM_ASCEND_DUMP_MODE="output"`（已默认）
2. **只 dump 第 0 个 token**：代码中已配置 `dump_token=[0]`
3. **只 dump MoE 相关算子**：代码中已配置 `dump_layer`
4. **减少 dump 次数**：`export VLLM_ASCEND_DUMP_MAX_REQUESTS=5`

---

## 八、快速参考

### 8.1 最小化 Dump 流程

```bash
# 1. 设置环境变量
export VLLM_ASCEND_DUMP_ENABLED=1
export VLLM_ASCEND_DUMP_PATH="./dump_temp"
export VLLM_ASCEND_DUMP_MAX_REQUESTS=10

# 2. 启动 vLLM serve
vllm serve <model_path> <args> --enforce-eager=false

# 3. 发送多次请求
for i in {1..10}; do
    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "<model_path>", "prompt": "Hello!", "max_tokens": 20, "temperature": 0}'
    sleep 0.5
done

# 4. 比对第 1 次（正常）和第 2 次（异常）
msit llm compare --golden-path ./run_1/msit_ge_dump --my-path ./run_2/msit_ge_dump --output ./comparison_1vs2

# 5. 查看结果
cat ./comparison_1vs2/compare_result.csv
```

### 8.2 环境变量速查表

```bash
# 必填
export VLLM_ASCEND_DUMP_ENABLED=1

# 可选（有默认值）
export VLLM_ASCEND_DUMP_PATH="./dump_temp"      # 默认: ./dump_base
export VLLM_ASCEND_DUMP_MODE="output"           # 默认: output
export VLLM_ASCEND_DUMP_MAX_REQUESTS=10          # 默认: 10
```

### 8.3 代码修改位置速查

| 修改点 | 文件 | 行号 | 作用 |
|-------|------|------|------|
| 计数器初始化 | `torchair_model_runner.py` | 85-92 | 添加 dump 配置属性 |
| GE dump 配置 | `torchair_model_runner.py` | 418-442 | 配置 msit GE dump |
| dump 后处理 | `torchair_model_runner.py` | 378-380 | 移动 dump 数据 |
| 数据移动逻辑 | `torchair_model_runner.py` | 396-439 | 实现数据移动 |

---

## 九、预期结果与下一步

### 9.1 预期发现

通过比对，你应该能够发现：

1. **第一个发散的算子**：例如 `QuantMatmul`、`AllToAll` 等
2. **发散程度**：`cosine_similarity < 0.99`
3. **问题模式**：
   - 量化算子发散 → 检查 `w4a4_flatquant_dynamic.py`
   - MoE 算子发散 → 检查 MoE 实现和权重加载
   - 通信算子发散 → 检查 AllToAll 同步逻辑

### 9.2 进一步定位

根据比对结果，可能的下一步：

1. **添加调试日志**：在可疑算子处添加 logger.debug
2. **检查权重状态**：打印权重 hash 和统计信息
3. **对比 eager 模式**：确认问题只在图模式出现
4. **清除图缓存**：排除图缓存污染

---

## 十、参考资料

- [TorchAir场景-整网算子精度比对](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/TorchAir场景-整网算子精度比对.md)
- [TorchAir场景Dump案例](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/TorchAir场景Dump案例.md)
- [大模型精度问题定位全流程](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/大模型精度问题定位全流程.md)
- [精度比对结果参数说明](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/精度比对结果参数说明.md)

---

## 十一、总结

**核心优势**：
- ✅ **一次启动，多次 dump**：无需每次重启 vLLM serve
- ✅ **自动计数器**：每次推理自动保存到 `run_1`、`run_2` ... 目录
- ✅ **MoE 优化**：只 dump MoE 相关算子，减少数据量
- ✅ **自动停止**：达到 `max_requests` 后自动禁用 dump

**使用流程**：
1. 设置 `VLLM_ASCEND_DUMP_ENABLED=1`
2. 启动 vLLM serve 一次
3. 发送多次请求（10 次）
4. 自动生成 `run_1/` 到 `run_10/` 目录
5. 使用 `msit llm compare` 比对 `run_1` 和 `run_2`
6. 定位第一个发散的算子
7. 修复代码

祝调试顺利！
