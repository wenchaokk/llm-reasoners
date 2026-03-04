# 快速开始：生成 10000 条经验

## 当前状态

已创建完整的经验生成和验证系统：

1. ✅ **经验生成器** (`reasoners/world_model/experience_generator.py`)
   - 基于规则生成
   - 从数据集提取
   - 使用本地模型生成

2. ✅ **经验验证器** (`reasoners/world_model/experience_validator.py`)
   - 动作可执行性检查
   - 逻辑一致性检查
   - 规则验证

3. ✅ **批量生成脚本** (`generate_10000_simple.py`)
   - 自动从数据集提取经验
   - 自动验证
   - 目标：10000 条

## 使用方法

### 方法 1：使用简化脚本（推荐）

```bash
python generate_10000_simple.py
```

这会：
- 从所有数据集文件中提取经验
- 自动验证每条经验
- 生成 10000 条有效经验
- 保存到 `world_model_memory.json`

### 方法 2：手动生成（逐步）

```bash
# 步骤 1：从数据集提取经验
python generate_experiences.py --method dataset \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 1000

# 步骤 2：验证经验
python validate_experiences.py --memory_file world_model_memory.json

# 步骤 3：重复步骤 1，直到达到 10000 条
```

### 方法 3：检查进度

```bash
python check_progress.py
```

## 验证经验

生成完成后，验证所有经验：

```bash
python validate_experiences.py --memory_file world_model_memory.json --verbose
```

## 使用经验

生成的经验会自动保存到 `world_model_memory.json`，可以在以下场景使用：

1. **只使用本地模型的 WorldModel**：
   ```python
   from reasoners.world_model import CachedWorldModel, MemoryStore
   from reasoners.lm import LlamaCppModel
   
   memory_store = MemoryStore(memory_file="world_model_memory.json")
   cached_world_model = CachedWorldModel(
       base_world_model=base_world_model,
       chatgpt_model=None,
       local_model=local_model,
       memory_store=memory_store,
       local_only=True
   )
   ```

2. **混合模式（ChatGPT + 本地模型）**：
   ```python
   cached_world_model = CachedWorldModel(
       base_world_model=base_world_model,
       chatgpt_model=chatgpt_model,
       local_model=local_model,
       memory_store=memory_store,
       use_cache=True
   )
   ```

## 预期结果

- **数据集提取方法**：准确率通常 > 90%
- **生成时间**：取决于数据集大小，可能需要几分钟到几十分钟
- **内存文件大小**：10000 条经验约 5-10 MB

## 故障排除

### 问题：生成的经验数量为 0

**可能原因**：
1. 数据集文件不存在
2. 数据格式不匹配
3. 验证失败

**解决方案**：
```bash
# 检查数据集文件
ls examples/CoT/blocksworld/data/split_v1/*.json

# 测试生成少量经验
python generate_experiences.py --method dataset \
    --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
    --num_experiences 10 \
    --no_validate
```

### 问题：验证准确率低

**解决方案**：
1. 检查错误详情：`python validate_experiences.py --verbose`
2. 重新生成经验
3. 使用数据集提取方法（准确率更高）

## 相关文件

- `generate_10000_simple.py` - 批量生成脚本
- `generate_experiences.py` - 单次生成脚本
- `validate_experiences.py` - 验证脚本
- `check_progress.py` - 进度检查脚本
- `reasoners/world_model/experience_generator.py` - 生成器实现
- `reasoners/world_model/experience_validator.py` - 验证器实现
- `GENERATE_EXPERIENCES_README.md` - 详细文档
