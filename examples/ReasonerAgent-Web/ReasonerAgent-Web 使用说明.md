# ReasonerAgent-Web 使用说明

基于 [llm-reasoners ReasonerAgent-Web](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/ReasonerAgent-Web)：一个**开源、可直接运行**的智能体，通过控制 Chromium 浏览器在网页上执行任务并回答你的问题。

---

## 一、环境要求

- **Python 3.10+**
- **Node.js / npm**（用于安装 Playwright）
- 建议 **16GB+ 内存**
- **OpenAI 或 DeepSeek 等 API Key**

---

## 二、安装步骤

### 1. 安装 Playwright 与浏览器

```powershell
# 若未安装 Node.js，请先安装：https://nodejs.org/

# 安装 Playwright（通过 npm）
npx playwright install

# 安装系统依赖（Windows 一般可跳过；Linux 需执行）
# npx playwright install-deps
```

### 2. 创建 Python 环境并安装依赖

在 **PowerShell** 中执行（在仓库根目录 `llm-reasoners` 下）：

```powershell
cd C:\Users\19933\llm-reasoners

# 使用 conda（推荐）
conda create -n reasoners python=3.10 -y
conda activate reasoners

# 安装主库（开发模式）
pip install -e .

# 进入示例目录并安装示例依赖
cd examples\ReasonerAgent-Web
pip install -r requirements.txt
```

若安装 `browsergym==0.3.6.dev0` 失败，可尝试：

```powershell
pip install browsergym
```

然后根据 [BrowserGym 官方说明](https://github.com/ServiceNow/BrowserGym) 处理版本兼容。

### 3. 配置 API Key（推荐）

在 `examples/ReasonerAgent-Web/` 下创建文件 `default_api_key.txt`，写入你的 API Key（一行，不要多余空格）：

```
sk-xxxxxxxxxxxxxxxx
```

这样运行时不必每次加 `--api_key` 参数。

---

## 三、快速运行

### 单次问答（推荐先试这个）

在 `examples/ReasonerAgent-Web` 目录下：

```powershell
python main.py test_job --query "Who is the current president of USA?" --api_key "sk-你的API密钥"
```

若已配置 `default_api_key.txt`，可省略 `--api_key`：

```powershell
python main.py test_job --query "美国现任总统是谁？"
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query` | 要回答的问题（单次运行必填） | - |
| `--api_key` | API 密钥 | 从 `default_api_key.txt` 读取 |
| `--agent` | 智能体类型 | `reasoner`（可选 `openhands`） |
| `--model` | 模型 | `gpt-4o`（可选 `o1`, `o3-mini`, `deepseek-chat`, `deepseek-reasoner`） |
| `--max_steps` | 单任务最大步数 | 30 |
| `--output_dir` | 浏览记录输出目录 | `./browsing_data` |

### 使用评估数据集

支持 **FanOutQA**、**FlightQA**、**WebArena**。示例（FanOutQA）：

```powershell
python main.py my_fanout_job --dataset fanout --start_idx 0 --end_idx 10
```

---

## 四、查看浏览历史（可视化）

```powershell
cd C:\Users\19933\llm-reasoners\examples\ReasonerAgent-Web
python log_visualizer/main.py
```

在终端里会打印一个本地链接，用浏览器打开即可查看每次任务的浏览与操作历史。

---

## 五、目录结构简要

```
ReasonerAgent-Web/
├── main.py              # 主入口
├── requirements.txt     # Python 依赖
├── default_api_key.txt  # API Key（需自行创建）
├── configs/             # 模型配置（如 o1、deepseek-reasoner）
├── data/                # 评估数据集
├── evaluation/          # FanOutQA、FlightQA、WebArena 评估脚本
└── log_visualizer/      # 历史日志可视化
```

---

## 六、常见问题

1. **没有 API Key**  
   需准备 OpenAI 或 DeepSeek 等支持的 API Key，并设置到 `default_api_key.txt` 或 `--api_key`。

2. **Playwright 报错**  
   确保执行过 `npx playwright install`，且本机可运行 Chromium。

3. **browsergym 安装失败**  
   参考 [BrowserGym 官方仓库](https://github.com/ServiceNow/BrowserGym) 的安装与版本说明。

4. **想用本地/自建模型**  
   项目通过 [LiteLLM](https://docs.litellm.ai/docs/) 调用模型，可在 `main.py` 中增加 `base_url` 和 provider 配置以对接本地服务。

---

## 七、参考链接

- 仓库：<https://github.com/maitrix-org/llm-reasoners>
- ReasonerAgent-Web 示例：<https://github.com/maitrix-org/llm-reasoners/tree/main/examples/ReasonerAgent-Web>
- 在线 Demo：<https://easyweb.maitrix.org>
- BrowserGym：<https://github.com/ServiceNow/BrowserGym>
