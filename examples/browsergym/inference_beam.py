"""
BrowserGym 环境中的 Beam Search 推理脚本

这个脚本展示了如何在 BrowserGym 环境中使用 Beam Search 算法执行任务。
它使用 LLM Reasoners 框架将浏览器自动化任务转化为搜索问题。

主要流程：
1. 解析命令行参数
2. 创建 BrowserGym 环境
3. 初始化 LLM 模型
4. 配置世界模型、搜索配置和搜索算法
5. 运行推理器执行任务
6. 保存结果并报告成功/失败
"""

import argparse
import os
import pickle
import time

from reasoners import Reasoner
from reasoners.algorithm import MCTS, BeamSearch, DFS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from gym_env import EnvironmentGym
from search_config import SearchConfigBrowsergym
from utils.misc import obs_preprocessor
from utils.parse import parse_common_arguments


def parse_arguments():
    """解析命令行参数
    
    这个函数解析所有命令行参数，包括：
    1. 通用参数（通过 parse_common_arguments 添加）
    2. Beam Search 特定参数（beam_size, max_depth）
    
    Returns:
        解析后的参数对象，包含所有配置信息
    """
    parser = argparse.ArgumentParser(
        description="使用指定参数运行任务。"
    )
    # 添加通用参数（任务、模型、环境等）
    parse_common_arguments(parser)

    # ========================================================================
    # Beam Search 算法特定参数
    # ========================================================================
    parser.add_argument(
        "--beam_size", 
        type=int, 
        default=2, 
        help="Beam Search 的束宽度。控制每一步保留多少个最佳候选。默认值：2"
    )
    parser.add_argument(
        "--max_depth", 
        type=int, 
        default=10, 
        help="Beam Search 的最大搜索深度。超过此深度将停止搜索。默认值：10"
    )

    return parser.parse_args()


def run_task(args):
    """运行浏览器任务
    
    这个函数是主执行流程，它：
    1. 创建浏览器动作集
    2. 创建 BrowserGym 环境
    3. 初始化 LLM 模型
    4. 配置世界模型、搜索配置和搜索算法
    5. 运行推理器执行任务
    6. 保存结果并返回是否成功
    
    Args:
        args: 命令行参数对象，包含所有配置信息
        
    Returns:
        bool: 如果任务成功完成返回 True，否则返回 False
    """
    
    # ========================================================================
    # 1. 创建浏览器动作集
    # ========================================================================
    browser_action_set = HighLevelActionSet(
        subsets=[args.action_set],  # 动作集类型（如 "webarena"）
        strict=False,               # 是否严格模式
        multiaction=True,            # 是否允许多动作
        demo_mode="off",             # 演示模式关闭
    )

    # ========================================================================
    # 2. 创建环境参数
    # ========================================================================
    env_args = EnvArgs(
        task_name=args.task_name,    # 任务名称（如 "webarena.xxx"）
        task_seed=args.task_seed,    # 任务随机种子
        max_steps=args.max_steps,    # 最大步数
        headless=True,               # 无头模式（不显示浏览器窗口）
        record_video=True,           # 录制视频（用于调试和分析）
    )

    # ========================================================================
    # 3. 创建实验目录
    # ========================================================================
    exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ========================================================================
    # 4. 创建 BrowserGym 环境
    # ========================================================================
    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,  # 动作映射函数
        exp_dir=exp_dir,                                    # 实验目录（用于保存视频等）
    )

    # ========================================================================
    # 5. 初始化 LLM 模型
    # ========================================================================
    llm = OpenAIModel(
        model=args.model,           # 模型名称（如 "gpt-4o-mini"）
        temperature=args.temperature, # 采样温度
        max_tokens=args.max_tokens,   # 最大生成 token 数
        backend=args.backend,         # 后端（"openai" 或 "sglang"）
    )

    # ========================================================================
    # 6. 创建世界模型（环境适配器）
    # ========================================================================
    # EnvironmentGym 将 Gymnasium 环境适配到 LLM Reasoners 框架
    # obs_preprocessor 用于预处理观察数据（如将截图转换为 base64 URL）
    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor)
    
    # ========================================================================
    # 7. 创建搜索配置
    # ========================================================================
    # SearchConfigBrowsergym 定义了：
    # - 如何生成动作（使用 LLM 生成动作提案）
    # - 如何评估动作（使用 LLM 评估动作的好坏）
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set,  # 浏览器动作集
        n_proposals=10,                  # 每次生成的动作提案数量
        llm=llm,                         # LLM 模型（用于生成和评估动作）
        use_axtree=True,                 # 使用可访问性树（推荐）
        use_html=False,                  # 不使用 HTML（减少 token 消耗）
        use_screenshot=False,           # 不使用截图（需要视觉模型）
    )
    
    # ========================================================================
    # 8. 创建搜索算法
    # ========================================================================
    # Beam Search 算法：每一步保留 top-k 个最佳候选
    algorithm = BeamSearch(
        beam_size=args.beam_size,  # 束宽度（每一步保留的候选数量）
        max_depth=args.max_depth,   # 最大搜索深度
    )

    # ========================================================================
    # 9. 创建推理器并运行
    # ========================================================================
    # Reasoner 组合三个组件：世界模型、搜索配置、搜索算法
    reasoner = Reasoner(world_model, search_config, algorithm)

    # 运行推理器执行任务
    # 这会启动搜索过程，使用 Beam Search 找到完成任务的最佳动作序列
    plan_result = reasoner()

    # ========================================================================
    # 10. 保存结果
    # ========================================================================
    # 将结果保存为 pickle 文件，可以用于后续分析和可视化
    with open(f"{exp_dir}/result.pkl", "wb") as f:
        pickle.dump(plan_result, f)

    # 关闭环境（释放资源）
    env.close()

    # 检查任务是否成功完成
    # 成功条件：达到终止状态且奖励为 1.0
    return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0


if __name__ == "__main__":
    """主入口函数
    
    执行流程：
    1. 解析命令行参数
    2. 运行任务并计时
    3. 报告成功/失败
    4. 显示执行时间
    """
    # 解析命令行参数
    args = parse_arguments()

    # 记录开始时间
    start_time = time.time()
    
    # 运行任务
    success = run_task(args)

    # 报告结果
    if success:
        print("Task completed successfully.")
    else:
        print(
            "Task didn't reach the goal. Please check the detailed result w/ visualization (python visualize.py --task_name <task_name>).",
        )

    # 计算并显示执行时间
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
