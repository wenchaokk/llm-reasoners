# 经验生成指南：不使用 ChatGPT 生成 WorldModel 经验

## 概述

这个工具提供了**三种方法**来生成 WorldModel 经验，**完全不需要 ChatGPT**：

1. **基于规则生成**（推荐）：使用 Blocksworld 的规则自动生成经验
2. **从数据集提取**：从已有的数据集中提取经验
3. **使用本地模型生成**：使用本地小模型生成经验

## 方法 1：基于规则生成（最简单，推荐）

使用 Blocksworld 的规则自动生成经验，不需要任何模型或数据集。

### 使用方法

```bash
python generate_experiences.py --method rule --num_experiences 100
```

### 特点

- ✅ **完全免费**：不需要任何 API 或模型
- ✅ **速度快**：基于规则，瞬间生成
- ✅ **准确性高**：使用 Blocksworld 的确定性规则
- ✅ **可扩展**：可以生成大量经验

### 生成的经验示例

```
状态: hand is empty, the red block is on the table, the blue block is on top of the red block
动作: pick up the blue block
结果: is holding the blue block, the red block is on the table, the red block is clear
```

## 方法 2：从数据集提取

从已有的 Blocksworld 数据集中提取经验。

### 使用方法

```bash
python generate_experiences.py --method dataset \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 50
```

### 特点

- ✅ **真实数据**：来自实际的问题实例
- ✅ **多样性好**：覆盖不同的状态和动作组合
- ✅ **免费**：不需要 API

### 数据文件

可以使用任何 Blocksworld 数据集文件，例如：
- `examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json`
- `examples/CoT/blocksworld/data/split_v1/split_v1_step_5_data.json`
- 等等

## 方法 3：使用本地模型生成

使用本地小模型（如 LlamaCpp）生成经验。

### 使用方法

```bash
python generate_experiences.py --method local_model \
    --model_path path/to/model.gguf \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 50
```

### 特点

- ✅ **使用本地模型**：不需要 ChatGPT API
- ✅ **可定制**：可以使用不同的模型
- ⚠️ **需要模型文件**：需要先下载模型
- ⚠️ **速度较慢**：需要模型推理

### 模型要求

- LlamaCpp 格式（`.gguf` 文件）
- 例如：`llama-2-7b-chat.gguf`, `llama-3-8b-instruct.gguf`

## 完整参数说明

```bash
python generate_experiences.py \
    --method rule \                    # 方法：rule, dataset, local_model
    --data_file path/to/data.json \    # 数据集文件（dataset/local_model 需要）
    --model_path path/to/model.gguf \  # 模型路径（local_model 需要）
    --memory_file world_model_memory.json \  # 内存文件路径
    --num_experiences 100              # 要生成的经验数量
```

## 使用示例

### 示例 1：快速生成 100 条经验（基于规则）

```bash
python generate_experiences.py --method rule --num_experiences 100
```

输出：
```
============================================================
WorldModel 经验生成器
============================================================
方法: rule
内存文件: world_model_memory.json
目标经验数: 100

当前内存中有 0 条经验

开始使用 'rule' 方法生成经验...

生成 45 条经验，正在存入内存...
✓ 成功存入 45 条经验
  内存中总共有 45 条经验

============================================================
✓ 经验生成完成！
============================================================

内存文件: world_model_memory.json
总经验数: 45
```

### 示例 2：从数据集提取经验

```bash
python generate_experiences.py --method dataset \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 50
```

### 示例 3：使用本地模型生成经验

```bash
python generate_experiences.py --method local_model \
    --model_path models/llama-2-7b-chat.gguf \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 30
```

## 在代码中使用

```python
from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences

# 创建内存存储
memory_store = MemoryStore(memory_file="world_model_memory.json")

# 方法 1：基于规则生成
generate_experiences(
    memory_store=memory_store,
    method="rule",
    num_experiences=100
)

# 方法 2：从数据集提取
generate_experiences(
    memory_store=memory_store,
    method="dataset",
    data_file="examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json",
    num_experiences=50
)

# 方法 3：使用本地模型生成
from reasoners.lm import LlamaCppModel
local_model = LlamaCppModel(path="path/to/model.gguf")
with open("examples/CoT/blocksworld/prompts/pool_prompt_v1.json") as f:
    prompt_template = json.load(f)

generate_experiences(
    memory_store=memory_store,
    method="local_model",
    local_model=local_model,
    prompt_template=prompt_template,
    data_file="examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json",
    num_experiences=30
)
```

## 推荐工作流程

### 1. 快速开始（推荐）

```bash
# 步骤 1：使用规则生成一些基础经验
python generate_experiences.py --method rule --num_experiences 100

# 步骤 2：从数据集提取更多经验
python generate_experiences.py --method dataset \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 50

# 步骤 3：现在可以使用只使用本地模型的 WorldModel 了
python test_local_only_worldmodel.py
```

### 2. 高质量经验（如果有本地模型）

```bash
# 步骤 1：使用规则生成基础经验
python generate_experiences.py --method rule --num_experiences 50

# 步骤 2：使用本地模型生成更高质量的经验
python generate_experiences.py --method local_model \
    --model_path path/to/model.gguf \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 30
```

## 查看生成的经验

经验保存在 `world_model_memory.json` 文件中，格式如下：

```json
{
  "memories": [
    {
      "state": "hand is empty, the red block is on the table",
      "action": "pick up the red block",
      "next_state": "is holding the red block, the red block is on the table",
      "prompt": "...",
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "metadata": {
    "count": 1,
    "last_updated": "2024-01-01T12:00:00"
  }
}
```

## 验证经验的准确性

生成经验后，可以使用验证工具检查准确性：

### 方法 1：使用验证脚本（推荐）

```bash
python validate_experiences.py --memory_file world_model_memory.json
```

这会执行三种验证：
1. **动作可执行性检查**：检查动作在当前状态下是否可以执行
2. **逻辑一致性检查**：检查状态转换是否符合 Blocksworld 规则
3. **规则验证**：使用规则生成器验证状态转换是否正确

### 方法 2：生成时自动验证

生成经验时会自动验证（默认启用）：

```bash
python generate_experiences.py --method rule --num_experiences 100
```

如果经验无效，会被自动过滤掉。

### 验证结果示例

```
============================================================
经验验证器
============================================================

加载了 100 条经验

开始验证 100 条经验...
  验证进度: 10/100
  验证进度: 20/100
  ...

============================================================
验证结果
============================================================
总经验数: 100
有效经验: 98
无效经验: 2
准确率: 98.00%

错误详情:

action_applicability 错误 (1 个):
  - pick up the red block: 动作 'pick up the red block' 在当前状态下不可执行

logical_consistency 错误 (1 个):
  - stack the blue block on top of the red block: 执行 'stack' 后，手应该是空的

✓ 经验质量很好！
```

### 验证标准

经验被认为是**有效**的，如果：
- ✅ 动作在当前状态下可执行
- ✅ 状态转换符合 Blocksworld 规则
- ✅ 手的状态转换正确
- ✅ 积木位置变化正确
- ✅ 积木的 clear 状态正确

### 查看详细错误信息

使用 `--verbose` 参数查看详细的错误信息：

```bash
python validate_experiences.py --memory_file world_model_memory.json --verbose
```

## 常见问题

### Q: 哪种方法最好？

**A:** 推荐使用**方法 1（基于规则）**，因为：
- 完全免费
- 速度快
- 准确性高
- 不需要任何额外资源

### Q: 需要生成多少经验？

**A:** 建议：
- **最少**：50-100 条经验（可以覆盖基本场景）
- **推荐**：200-500 条经验（覆盖更多场景）
- **最佳**：1000+ 条经验（覆盖几乎所有场景）

### Q: 可以混合使用多种方法吗？

**A:** 可以！多次运行脚本，经验会累积到同一个内存文件中。

### Q: 经验会覆盖吗？

**A:** 不会。新经验会追加到现有经验中，除非达到 `max_memory_size` 限制。

### Q: 如何清空经验？

**A:** 删除 `world_model_memory.json` 文件，或使用代码：

```python
memory_store = MemoryStore(memory_file="world_model_memory.json")
memory_store.clear()
```

### Q: 如何判断生成的经验是否准确？

**A:** 使用验证工具：

```bash
# 验证所有经验
python validate_experiences.py --memory_file world_model_memory.json

# 查看详细错误信息
python validate_experiences.py --memory_file world_model_memory.json --verbose
```

验证器会检查：
- 动作可执行性
- 逻辑一致性
- 规则正确性

**基于规则生成的经验准确率通常 > 95%**，因为使用的是 Blocksworld 的确定性规则。

### Q: 如果验证发现错误怎么办？

**A:** 
1. **少量错误（< 5%）**：可以忽略，不影响使用
2. **大量错误（> 10%）**：
   - 检查生成代码
   - 重新生成经验
   - 使用 `rule` 方法（准确性最高）
3. **特定类型错误**：查看错误详情，修复生成逻辑

## 两种预测方式对比（ChatGPT vs 本地小模型+内存）

本代码库的 WorldModel 支持两种预测方式，可用同一批测试用例对比预测成功率：

| 方式 | 说明 | 对应实现 |
|------|------|----------|
| **ChatGPT** | 每一步都用 OpenAI 预测下一状态 | `BlocksWorldModel(base_model=chatgpt_lm)` |
| **本地小模型 + 内存经验** | 只用车内模型 + 内存中的相似经验预测（不调 API） | `CachedWorldModel(local_only=True, local_model=..., memory_store=...)` |

**本地小模型一键配置：**

```bash
# 下载 TinyLlama-1.1B（约 700MB）到项目 models/，之后对比脚本会自动使用
python setup_local_model.py
```

**运行对比：**

```bash
# 只跑 ChatGPT 模式（需配置 OPENAI_API_KEY）
python compare_world_model_modes.py --max_cases 20

# 同时跑两种方式并对比（若已运行过 setup_local_model.py，可不写 --local_model_path）
python compare_world_model_modes.py --max_cases 20
python compare_world_model_modes.py --max_cases 20 --local_model_path path/to/model.gguf
```

- 测试用例由规则生成，标准答案为规则给出的下一状态；预测正确即与标准答案一致。
- 脚本会输出两种方式的「正确数/总数」和「预测成功率」，便于对比。

详见：`compare_world_model_modes.py`（项目根目录）。

### 在 Google Colab 中运行对比

项目根目录下提供了 **`colab_compare_world_model.ipynb`**，在 Colab 中打开后按顺序运行各单元格即可完成对比。

1. **克隆仓库**：`!git clone ...` 到 `/content/llm-reasoners` 并 `%cd` 进入目录。
2. **安装依赖**：安装 `huggingface_hub`、`openai`，以及带 CUDA 的 `llama-cpp-python`（约 2–5 分钟）。
3. **下载模型**：运行 `python setup_local_model.py --model llama32_8b`（免费 Colab 用 8B）；Colab Pro A100 可改用 `--model llama31_70b`。
4. **（可选）设置 OpenAI API Key**：若需要跑「模式 1：ChatGPT」，在 Colab 中设置 `os.environ["OPENAI_API_KEY"] = "..."`。
5. **运行对比**：`!python compare_world_model_modes.py --max_cases 15`。

笔记本路径：**`colab_compare_world_model.ipynb`**（可上传到 Colab 或从 GitHub 打开）。

## 相关文件

- `reasoners/world_model/experience_generator.py`: 经验生成器实现
- `reasoners/world_model/cached_world_model.py`: 带内存的 WorldModel（支持 ChatGPT / 本地+内存 两种预测）
- `generate_experiences.py`: 命令行脚本
- `reasoners/world_model/memory_store.py`: 内存存储系统
- `compare_world_model_modes.py`: **两种预测方式对比脚本**（ChatGPT vs 本地小模型+内存）
- `test_local_only_worldmodel.py`: 测试只使用本地模型的脚本
