# 复现 Baseline 方法（Blocksworld）

Baseline 指 **Chain-of-Thought (CoT)**，即直接让 LLM 生成完整动作序列，不使用搜索算法。

## 快速清单（首次运行前检查）

| 步骤 | 命令/操作 | 状态 |
|------|-----------|------|
| 1. 安装依赖 | `pip install -e .` 和 `pip install pddl==0.2.0` | □ |
| 2. 初始化子模块 | `git submodule update --init` | □ |
| 3. 配置 API Key | 编辑 `api_key.txt` 填入 OpenAI Key | □ |
| 4. 开通计费 | 在 platform.openai.com 添加付款方式 | □ |

**run_baseline_with_monitor.py 会自动设置 PLANBENCH_PATH 和 VAL**，无需手动配置。

---

## 一、环境准备

### 1. 安装依赖

```bash
cd llm-reasoners
pip install -e .
pip install pddl==0.2.0   # Blocksworld 评估所需
```

### 2. 初始化 LLMs-Planning 子模块

```bash
git submodule update --init
```

### 3. 配置 VAL 验证工具

Blocksworld 评估需要 VAL 验证生成的计划。子模块已包含 VAL：

```powershell
# Windows PowerShell（在项目根目录执行）
$env:PLANBENCH_PATH = "C:\Users\19933\llm-reasoners\LLMs-Planning"
$env:VAL = "C:\Users\19933\llm-reasoners\LLMs-Planning\planner_tools\VAL\bin\validate"
```

```bash
# Linux/Mac
export PLANBENCH_PATH="$(pwd)/LLMs-Planning"
export VAL="$(pwd)/LLMs-Planning/planner_tools/VAL"
```

注意：Windows 下 VAL 的 `validate` 可能需在 WSL 中运行；若报错可尝试使用 Linux 环境。

### 4. 配置 API Key（使用 OpenAI 时）

编辑项目根目录的 `api_key.txt`，填入你的 OpenAI API Key。

---

## 二、运行 CoT Baseline

### 方式 A：带 API 用量监控（推荐）

```bash
cd llm-reasoners

# 运行 CoT baseline，每次 API 调用后实时打印 Token 和费用
python run_baseline_with_monitor.py

# 仅跑前 N 个样本（快速测试）
python run_baseline_with_monitor.py --max_samples 2
```

脚本会自动：从 `api_key.txt` 加载 Key、设置 PLANBENCH_PATH/VAL、使用 `OpenAIModelWithUsage` 实时监控用量。

### 方式 B：命令行（原 cot_inference.py）

```bash
cd llm-reasoners

# 单子集（step_4，约 84 样本）
python examples/CoT/blocksworld/cot_inference.py \
  --model_dir openai \
  --data_path examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
  --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json \
  --log_dir logs/blocksworld_cot_openai_step4/ \
  --temperature 0.0

# 完整复现（所有 step：2,4,6,8,10,12）
# 参考 examples/CoT/blocksworld/test_cot_mistral.sh
```

### 方式 C：在 demo.ipynb 中运行

1. 运行「方案 1：使用 OpenAI API」单元格
2. 运行「加载积木世界评估器和数据」及后续单元格
3. 最后运行「在数据集上评估推理器」单元格（需将 `reasoner_tot` 改为 CoT 的 reasoner）

---

## 三、数据集说明

| 设置 | 数据路径 | 样本数 |
|------|----------|--------|
| Hard (v1) step 2 | split_v1/split_v1_step_2_data.json | ~84 |
| Hard (v1) step 4 | split_v1/split_v1_step_4_data.json | ~84 |
| Hard (v1) step 6 | split_v1/split_v1_step_6_data.json | ~84 |
| ... | ... | ... |

论文 Table 3/4 使用 Hard 设置 (split_v1)。

---

## 四、其他 Baseline 方法

| 方法 | 脚本 | 说明 |
|------|------|------|
| **ToT** (Tree-of-Thought) | examples/ToT/blocksworld/tot_inference.py | Beam Search |
| **RAP** (Reasoning-via-Planning) | examples/RAP/blocksworld/rap_inference.py | MCTS |

ToT 和 RAP 需要 `get_loglikelihood`，**OpenAI API 不支持**，需使用 HuggingFace 等本地模型。

---

## 五、聚合结果

运行完所有 step 后，可用 `aggregate.py` 计算总体准确率：

```bash
python examples/CoT/blocksworld/aggregate.py logs/blocksworld_cot_openai_*/
```
