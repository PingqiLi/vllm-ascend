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

1. **在 `__init__` 方法中添加 dump 配置加载**（第 85-90 行）：
   - `self.dump_counter = 0`：推理计数器
   - `self.dump_config = self._load_dump_config()`：加载 JSON 配置文件
   - `self.dump_enabled`：从配置中读取是否启用 dump

2. **新增 `_load_dump_config` 方法**（第 92-140 行）：
   - 通过环境变量 `VLLM_ASCEND_DUMP_CONFIG` 指定配置文件路径
   - 解析 JSON 配置文件并设置默认值
   - 处理文件不存在或格式错误的情况

3. **在 `_get_torchair_lazy_compiled_model` 方法中配置 GE dump**（第 516-544 行）：
   - 使用 `msit_llm.dump.torchair_dump.get_ge_dump_config()` 配置
   - 从 dump_config 中读取所有参数（dump_path、dump_mode、dump_token、dump_layer、fusion_switch_file）

4. **在 `_generate_process_reqs_hidden_states` 方法中处理 dump 数据**（第 426-428 行）：
   - 每次 decode 推理后，将 dump 数据移动到 `run_1`、`run_2` ... 目录
   - 计数器自动递增
   - 达到 `max_requests` 后自动停止 dump

5. **`_move_dump_data_to_counter_dir` 方法**（第 444-490 行）：
   - 将 `{dump_path}/msit_ge_dump` 移动到 `run_{N}/msit_ge_dump`
   - 清空临时目录，为下一次 dump 做准备

### 2.2 配置文件说明

**环境变量**：`VLLM_ASCEND_DUMP_CONFIG` - 指向 dump 配置 JSON 文件的路径

**配置文件示例**（`dump_config.json`）：

```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "output",
  "dump_token": [0],
  "dump_layer": [
    "MatMul",
    "MatMulV2",
    "BatchMatMul",
    "QuantBatchMatmul",
    "QuantMatmul",
    "AllToAll",
    "AllGather",
    "ReduceScatter"
  ],
  "fusion_switch_file": null,
  "max_requests": 10
}
```

**配置参数说明**：

| 参数 | 说明 | 默认值 | 示例 |
|-----|------|--------|------|
| `dump_enabled` | 是否启用 dump（**必填**） | `false` | `true` |
| `dump_path` | 临时 dump 目录路径 | `./dump_base` | `"./dump_temp"` |
| `dump_mode` | dump 模式：`input`、`output`、`all` | `output` | `"output"` |
| `dump_token` | 指定要 dump 的 token 索引 | `null`（全部） | `[0]` 或 `[0,1,2]` |
| `dump_layer` | 指定要 dump 的算子名称 | `null`（全部） | `["MatMul", "QuantMatmul"]` |
| `fusion_switch_file` | 融合开关配置文件路径 | `null` | `"./fusion_switch.json"` |
| `max_requests` | 最多 dump 多少次请求 | `10` | `10` |

**注意**：
- **`dump_enabled: true` 是启用 dump 的开关**
- dump 数据最终保存在与 `dump_path` 同级的 `run_1`、`run_2` ... 目录中
- 只 dump **decode 阶段**的推理（prefill 阶段不 dump）
- `dump_token` 和 `dump_layer` 为 `null` 时会 dump 全量数据，建议指定范围以减少数据量

---

## 三、执行 Dump

### 3.1 一次性 Dump 多次推理

**优势**：
- **启动一次 vLLM serve**，发送多次请求，自动 dump 到不同目录
- 无需每次重启服务
- 所有配置集中在一个 JSON 文件中，易于管理

**步骤**：

```bash
# 1. 创建或编辑 dump 配置文件
cat > dump_config.json <<EOF
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "output",
  "dump_token": [0],
  "dump_layer": [
    "MatMul",
    "QuantBatchMatmul",
    "QuantMatmul",
    "AllToAll"
  ],
  "fusion_switch_file": null,
  "max_requests": 10
}
EOF

# 2. 设置环境变量指向配置文件
export VLLM_ASCEND_DUMP_CONFIG="./dump_config.json"

# 3. 启动 vLLM 服务
vllm serve /path/to/qwen3-30b-a3b-w4a4 \
    --trust-remote-code \
    --dtype float16 \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2048 \
    --quantization w4a4_flatquant_dynamic \
    --enforce-eager=false  # 确保使用图模式

# 4. 在另一个终端发送多次请求（例如 10 次）
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

# 5. 查看生成的 dump 目录
ls -l
# 应该看到：run_1/, run_2/, run_3/, ..., run_10/, dump_temp/

# 6. 停止 vLLM 服务
```

**预期结果**：
- 生成 `run_1/` 到 `run_10/` 共 10 个目录
- 每个目录下都有 `msit_ge_dump/` 子目录，包含对应请求的 dump 数据
- 第 1、3、5、7、9 次应该输出正常
- 第 2、4、6、8、10 次应该输出 "!!!!"

### 3.2 配置文件变体

**最小配置**（只 dump，不限制算子）：
```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp"
}
```

**关闭融合配置**：
```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "output",
  "dump_token": [0],
  "fusion_switch_file": "./fusion_switch.json",
  "max_requests": 10
}
```

**只 dump 特定算子**：
```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_layer": ["QuantMatmul", "AllToAll"]
}
```

### 3.3 Dump 数据目录结构

```
./
├── dump_config.json        # dump 配置文件
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

---

## 五、配置文件管理

### 5.1 针对不同场景的配置

**场景 1：快速定位（只 dump MoE 相关算子）**

```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "output",
  "dump_token": [0],
  "dump_layer": ["QuantMatmul", "AllToAll"],
  "max_requests": 5
}
```

**场景 2：全量 dump（所有算子）**

```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "all",
  "dump_token": null,
  "dump_layer": null,
  "max_requests": 3
}
```

**场景 3：关闭融合进行比对**

```json
{
  "dump_enabled": true,
  "dump_path": "./dump_fusion_off",
  "dump_mode": "output",
  "dump_token": [0],
  "fusion_switch_file": "./fusion_switch.json",
  "max_requests": 10
}
```

### 5.2 配置文件验证

验证配置文件格式是否正确：

```bash
# 使用 jq 验证 JSON 格式
jq . dump_config.json

# 或使用 Python
python -c "import json; print(json.load(open('dump_config.json')))"
```

---

## 六、常见问题排查

### 6.1 配置文件未加载

**现象**：
```
WARNING: Dump config file not found: xxx, dump disabled
```

**解决**：
```bash
# 检查环境变量
echo $VLLM_ASCEND_DUMP_CONFIG

# 检查文件是否存在
ls -la ./dump_config.json

# 使用绝对路径
export VLLM_ASCEND_DUMP_CONFIG="/absolute/path/to/dump_config.json"
```

### 6.2 JSON 格式错误

**现象**：
```
WARNING: Failed to parse dump config xxx: Expecting ',' delimiter, dump disabled
```

**解决**：
```bash
# 使用 jq 检查格式
jq . dump_config.json

# 常见错误：最后一项有多余逗号
{
  "dump_enabled": true,  # ← 多余逗号
}
```

### 6.3 Dump 数据未生成

**可能原因**：
1. `dump_enabled` 设置为 `false`
2. 配置文件路径错误
3. 权限问题，无法写入目标目录

**检查**：
```bash
# 查看日志中是否有 "GE dump enabled"
grep "GE dump" <vllm_log_file>

# 查看是否有 "GE dump configured"
grep "GE dump configured" <vllm_log_file>

# 查看是否有 "Moved dump data to run_X"
grep "Moved dump data" <vllm_log_file>
```

### 6.4 msit_llm 导入失败

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

---

## 七、快速参考

### 7.1 最小化 Dump 流程

```bash
# 1. 创建配置文件
cat > dump_config.json <<'EOF'
{"dump_enabled": true, "dump_path": "./dump_temp"}
EOF

# 2. 设置环境变量
export VLLM_ASCEND_DUMP_CONFIG="./dump_config.json"

# 3. 启动 vLLM serve
vllm serve <model_path> <args> --enforce-eager=false

# 4. 发送多次请求
for i in {1..10}; do
    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "<model_path>", "prompt": "Hello!", "max_tokens": 20, "temperature": 0}'
    sleep 0.5
done

# 5. 比对
msit llm compare --golden-path ./run_1/msit_ge_dump --my-path ./run_2/msit_ge_dump --output ./comparison

# 6. 查看结果
cat ./comparison/compare_result.csv
```

### 7.2 配置文件模板

**基础模板**：
```json
{
  "dump_enabled": true,
  "dump_path": "./dump_temp",
  "dump_mode": "output",
  "dump_token": [0],
  "dump_layer": null,
  "fusion_switch_file": null,
  "max_requests": 10
}
```

### 7.3 代码修改位置速查

| 修改点 | 文件 | 行号 | 作用 |
|-------|------|------|------|
| 配置加载 | `torchair_model_runner.py` | 85-90 | 加载 dump 配置 |
| 配置解析 | `torchair_model_runner.py` | 92-140 | 解析 JSON 文件 |
| GE dump 配置 | `torchair_model_runner.py` | 516-544 | 配置 msit GE dump |
| dump 后处理 | `torchair_model_runner.py` | 426-428 | 移动 dump 数据 |
| 数据移动逻辑 | `torchair_model_runner.py` | 444-490 | 实现数据移动 |

---

## 八、总结

**核心优势**：
- ✅ **配置集中管理**：所有参数在一个 JSON 文件中，清晰易维护
- ✅ **灵活配置**：支持所有 msit GE dump 参数
- ✅ **一次启动，多次 dump**：无需每次重启 vLLM serve
- ✅ **自动计数器**：每次推理自动保存到 `run_1`、`run_2` ... 目录
- ✅ **MoE 优化**：可指定只 dump MoE 相关算子
- ✅ **错误处理**：配置文件错误时优雅降级，不影响正常使用

**使用流程**：
1. 创建 `dump_config.json` 配置文件
2. 设置 `export VLLM_ASCEND_DUMP_CONFIG="./dump_config.json"`
3. 启动 vLLM serve 一次
4. 发送多次请求（10 次）
5. 自动生成 `run_1/` 到 `run_10/` 目录
6. 使用 `msit llm compare` 比对 `run_1` 和 `run_2`
7. 定位第一个发散的算子
8. 修复代码

祝调试顺利！
