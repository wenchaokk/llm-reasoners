"""
BrowserGym 命令行参数解析工具

这个模块提供了 BrowserGym 环境中使用的通用命令行参数解析函数。
用于配置任务、模型、环境等参数。
"""

import argparse


def parse_common_arguments(parser: argparse.ArgumentParser):
    """解析 BrowserGym 环境的通用命令行参数
    
    这个函数为 argparse.ArgumentParser 添加所有 BrowserGym 需要的命令行参数。
    参数分为以下几类：
    1. 任务参数：任务名称、随机种子
    2. 路径参数：结果保存目录
    3. 模型参数：模型名称、温度、最大 token 数、后端
    4. 环境参数：最大步数、动作集、观察模式、视频录制
    
    Args:
        parser: argparse.ArgumentParser 对象，用于添加参数
        
    使用示例:
        >>> parser = argparse.ArgumentParser()
        >>> parse_common_arguments(parser)
        >>> args = parser.parse_args()
    """
    
    # ========================================================================
    # 任务参数
    # ========================================================================
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="任务名称。例如：webarena.<task_id>。注意：你需要先部署任务网站。",
    )
    parser.add_argument(
        "--task_seed", 
        type=int,
        default=42, 
        help="任务的随机种子，用于确保任务的可重复性。默认值：42"
    )
    
    # ========================================================================
    # 路径参数
    # ========================================================================
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/tree-search",
        help="保存实验结果的目录路径。默认值：results/tree-search",
    )
    
    # ========================================================================
    # 模型参数
    # ========================================================================
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "要使用的 OpenAI 模型名称。默认值：gpt-4o-mini。"
            "注意：你也可以适配 `reasoners.lm` 中的任何其他模型。"
        ),
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help=(
            "模型的采样温度。控制输出的随机性："
            "- 0.0：完全确定性（贪心解码）"
            "- 0.7：平衡随机性和确定性（推荐）"
            "- 1.0：完全随机"
            "默认值：0.7"
        )
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2048, 
        help="模型生成的最大 token 数。默认值：2048"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "sglang"],
        help=(
            "模型后端。当前支持："
            "- openai: OpenAI API 后端"
            "- sglang: SGLang 高性能推理框架后端（需要本地部署）"
            "默认值：openai"
        ),
    )
    
    # ========================================================================
    # 环境参数
    # ========================================================================
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help=(
            "环境允许的最大步数。如果达到此步数仍未完成任务，"
            "任务将被视为失败。默认值：15"
        ),
    )
    parser.add_argument(
        "--action_set", 
        type=str, 
        default="webarena", 
        help=(
            "要使用的动作集。当前支持："
            "- webarena: WebArena 动作集（默认）"
            "默认值：webarena"
        )
    )
    parser.add_argument(
        "--use_axtree",
        type=bool,
        default=True,
        help=(
            "是否在观察数据中包含可访问性树（a11y tree）信息。"
            "可访问性树提供了页面的结构化表示，有助于 LLM 理解页面元素。"
            "默认值：True（推荐开启）"
        ),
    )
    parser.add_argument(
        "--use_html",
        type=bool,
        default=False,
        help=(
            "是否在观察数据中包含 HTML 信息。"
            "HTML 提供了页面的完整结构，但可能包含大量冗余信息。"
            "默认值：False（通常与 use_axtree 一起使用）"
        ),
    )
    parser.add_argument(
        "--use_screenshot",
        type=bool,
        default=False,
        help=(
            "是否在观察数据中包含截图信息。"
            "截图提供了页面的视觉信息，但需要模型支持图像处理（如 GPT-4V）。"
            "注意：如果设置为 True，请确保使用的模型可以处理图像。"
            "默认值：False"
        ),
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help=(
            "是否录制浏览器操作的视频。"
            "视频可以用于调试和分析智能体的行为。"
            "默认值：False"
        ),
    )
