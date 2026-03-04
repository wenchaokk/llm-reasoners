# 带内存缓存的 WorldModel 使用指南

## 概述

这个系统实现了一个**混合 WorldModel**，它结合了 ChatGPT 和本地小模型的优势：

1. **ChatGPT**：用于新的、未见过的状态转换预测
2. **本地小模型**：用于基于历史经验的预测（节省 API 调用成本）
3. **内存系统**：自动存储和查询历史经验

## 工作原理

```
┌─────────────────────────────────────────────────────────┐
│                    CachedWorldModel                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 收到预测请求 (state, action)                        │
│     ↓                                                    │
│  2. 查询内存：是否有相似的历史经验？                     │
│     ↓                                                    │
│  3a. 有相似经验 (相似度 ≥ 阈值)                         │
│      → 使用本地小模型 + 历史经验进行预测                │
│      → 返回结果（不调用 ChatGPT）                        │
│                                                          │
│  3b. 没有相似经验                                        │
│      → 使用 ChatGPT 进行预测                             │
│      → 将结果存入内存                                    │
│      → 返回结果                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. MemoryStore（内存存储）

存储历史的状态转换经验：

```python
from reasoners.world_model import MemoryStore

memory_store = MemoryStore(
    memory_file="world_model_memory.json",  # 持久化文件
    similarity_threshold=0.7,              # 相似度阈值
    max_memory_size=10000                   # 最大存储数量
)

# 添加经验
memory_store.add(
    state="红色在桌上，蓝色在红色上",
    action="pick up the blue block",
    next_state="红色在桌上，蓝色在手中",
    prompt="..."
)

# 查询相似经验
similar = memory_store.query(
    state="红色在桌上，蓝色在红色上",
    action="pick up the blue block",
    top_k=3
)
```

### 2. CachedWorldModel（带缓存的 WorldModel）

包装原有的 WorldModel，添加缓存功能：

```python
from reasoners.world_model import CachedWorldModel
from reasoners.lm import OpenAIModelWithUsage, LlamaCppModel
from examples.CoT.blocksworld.world_model import BlocksWorldModel

# 1. 创建 ChatGPT 模型（用于新预测）
chatgpt_model = OpenAIModelWithUsage(
    model="gpt-4o-mini",
    max_tokens=512,
    temperature=0.0
)

# 2. 创建本地小模型（用于基于历史经验的预测）
local_model = LlamaCppModel(path="path/to/model.gguf")

# 3. 创建基础 WorldModel
base_world_model = BlocksWorldModel(
    base_model=chatgpt_model,
    prompt=prompt,
    max_steps=6
)

# 4. 创建带缓存的 WorldModel
cached_world_model = CachedWorldModel(
    base_world_model=base_world_model,
    chatgpt_model=chatgpt_model,
    local_model=local_model,
    memory_store=memory_store,
    use_cache=True,
    cache_threshold=0.7
)

# 5. 使用（和普通 WorldModel 一样）
state = cached_world_model.init_state()
new_state, aux = cached_world_model.step(state, action)
```

## 使用示例

### 快速开始

运行演示脚本：

```bash
python examples/CoT/blocksworld/cached_world_model_demo.py
```

### 完整示例

```python
import os
import sys
import json

# 添加项目根目录
_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _root)

from reasoners.lm import OpenAIModelWithUsage, LlamaCppModel
from reasoners.world_model import CachedWorldModel, MemoryStore
from examples.CoT.blocksworld.world_model import BlocksWorldModel
from load_api_key import load_api_key

# 1. 加载配置
load_api_key()

# 加载示例和提示词
with open("examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json") as f:
    examples = json.load(f)
example = examples[0]

with open("examples/CoT/blocksworld/prompts/pool_prompt_v1.json") as f:
    prompt = json.load(f)

# 2. 初始化模型
chatgpt_model = OpenAIModelWithUsage(
    model="gpt-4o-mini",
    max_tokens=512,
    temperature=0.0
)

local_model = LlamaCppModel(path="path/to/model.gguf")  # 可选

# 3. 创建 WorldModel
base_world_model = BlocksWorldModel(
    base_model=chatgpt_model,
    prompt=prompt,
    max_steps=6
)
base_world_model.update_example(example)

# 4. 创建内存存储
memory_store = MemoryStore(
    memory_file="world_model_memory.json",
    similarity_threshold=0.7
)

# 5. 创建带缓存的 WorldModel
cached_world_model = CachedWorldModel(
    base_world_model=base_world_model,
    chatgpt_model=chatgpt_model,
    local_model=local_model,
    memory_store=memory_store,
    use_cache=True
)

# 6. 使用
state = cached_world_model.init_state()
for action in actions:
    new_state, aux = cached_world_model.step(state, action)
    state = new_state
    if cached_world_model.is_terminal(state):
        break

# 7. 查看统计信息
cached_world_model.print_stats()
```

## 配置参数

### MemoryStore 参数

- `memory_file`: 持久化文件路径（默认：`world_model_memory.json`）
- `similarity_threshold`: 相似度阈值（0-1），低于此值不返回（默认：0.7）
- `max_memory_size`: 最大存储数量，超过后删除最旧的（默认：10000）

### CachedWorldModel 参数

- `base_world_model`: 原有的 WorldModel（必需）
- `chatgpt_model`: ChatGPT 模型（必需）
- `local_model`: 本地小模型（可选，如果不提供则只使用 ChatGPT）
- `memory_store`: 内存存储系统（可选，会自动创建）
- `use_cache`: 是否使用缓存（默认：True）
- `cache_threshold`: 缓存相似度阈值（默认：0.7）

## 优势

1. **节省成本**：相似的状态转换使用本地模型，减少 API 调用
2. **提高速度**：本地模型推理速度更快
3. **自动学习**：系统自动积累经验，越用越智能
4. **持久化存储**：经验保存在文件中，下次运行自动加载

## 统计信息

查看缓存效果：

```python
stats = cached_world_model.get_stats()
print(f"总预测次数: {stats['total_predictions']}")
print(f"缓存命中: {stats['cache_hits']}")
print(f"缓存未命中: {stats['cache_misses']}")
print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
print(f"内存中经验数: {stats['memory_stats']['total_memories']}")
```

## 注意事项

1. **本地模型**：需要先下载 LlamaCpp 格式的模型文件
2. **相似度阈值**：根据任务调整，太高可能错过有用经验，太低可能使用不相关的经验
3. **内存大小**：根据任务复杂度调整 `max_memory_size`
4. **持久化文件**：内存文件会不断增长，定期清理或设置合理的 `max_memory_size`

## 扩展

### 使用向量数据库（可选）

当前使用简单的文本相似度匹配。如果需要更精确的相似度计算，可以：

1. 使用向量数据库（如 FAISS、Chroma）
2. 使用嵌入模型（如 sentence-transformers）生成向量
3. 基于向量相似度查询

### 自定义相似度计算

继承 `MemoryStore` 并重写 `_calculate_similarity` 方法：

```python
class CustomMemoryStore(MemoryStore):
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        # 自定义相似度计算逻辑
        return your_custom_similarity(text1, text2)
```

## 故障排除

### 问题：本地模型加载失败

**解决方案**：
1. 确保已安装 `llama-cpp-python`：`pip install llama-cpp-python`
2. 检查模型文件路径是否正确
3. 如果不需要本地模型，可以设置为 `None`（只使用 ChatGPT）

### 问题：缓存命中率很低

**解决方案**：
1. 降低 `similarity_threshold`（如从 0.7 降到 0.5）
2. 增加内存大小，积累更多经验
3. 检查状态和动作的表示是否一致

### 问题：内存文件过大

**解决方案**：
1. 设置合理的 `max_memory_size`
2. 定期清理旧经验：`memory_store.clear()`
3. 使用更紧凑的存储格式

## 相关文件

- `reasoners/world_model/memory_store.py`: 内存存储实现
- `reasoners/world_model/cached_world_model.py`: 带缓存的 WorldModel 实现
- `examples/CoT/blocksworld/cached_world_model_demo.py`: 演示脚本
