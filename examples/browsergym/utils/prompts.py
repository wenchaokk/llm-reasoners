"""
BrowserGym 提示词构建工具

这个模块提供了构建 BrowserGym 环境中 LLM 提示词的函数。
主要用于：
1. 构建动作提案提示词（用于生成下一步动作）
2. 构建动作评估提示词（用于评估动作的好坏）

参考实现：
https://github.com/ServiceNow/BrowserGym/blob/main/demo_agent/agent.py
"""

import logging
import os
import re
import json
import argparse
import base64
import io
import numpy as np
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.chat import Chat
from browsergym.core.env import BrowserEnv
from browsergym.experiments import EnvArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.core.action.parsers import highlevel_action_parser

from reasoners import SearchConfig, WorldModel, LanguageModel
from .misc import image_to_jpg_base64_url


def get_user_messages_for_current_state(
    obs: dict,
    action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False
) -> list[dict]:
    """获取当前状态的用户消息列表
    
    这个函数将浏览器环境的观察数据转换为适合 LLM 处理的消息列表。
    消息列表包含：
    1. 任务目标
    2. 打开的标签页信息
    3. 页面结构信息（可访问性树、HTML、截图）
    4. 动作空间描述
    5. 历史动作记录
    
    Args:
        obs: 浏览器环境的观察数据字典
        action_set: 高级动作集对象
        action_history: 历史动作列表
        use_axtree: 是否包含可访问性树信息（默认：True）
        use_html: 是否包含 HTML 信息（默认：False）
        use_screenshot: 是否包含截图信息（默认：False）
        
    Returns:
        用户消息列表，每个消息是一个字典，包含 "type" 和 "text"/"image_url" 字段
        
    Raises:
        AssertionError: 如果观察数据中缺少目标信息
    """
    assert obs["goal_object"], "The goal is missing."
    user_msgs = []
    
    # 任务目标直接作为 OpenAI 风格的消息列表呈现
    user_msgs.extend(obs["goal_object"])
    # 添加所有打开的标签页信息
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Currently open tabs
    """,
        }
    )
    # 遍历所有打开的标签页，添加每个标签页的标题和 URL
    for page_index, (page_url, page_title) in enumerate(
        zip(obs["open_pages_urls"], obs["open_pages_titles"])
    ):
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
Title: {page_title}
URL: {page_url}
    """,
            }
        )

    # 添加页面可访问性树（如果启用）
    # 可访问性树提供了页面的结构化表示，有助于 LLM 理解页面元素
    if use_axtree:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current page Accessibility Tree

{obs["axtree_txt"]}
""",
            }
        )
    
    # 添加页面 HTML（如果启用）
    # HTML 提供了页面的完整 DOM 结构，但可能包含大量冗余信息
    if use_html:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
            }
        )

    # 添加页面截图（如果启用）
    # 截图提供了页面的视觉信息，需要模型支持图像处理（如 GPT-4V）
    if use_screenshot:
        user_msgs.append(
            {
                "type": "text",
                "text": """\
# Current page Screenshot
    """,
            }
        )
        # 将截图转换为 base64 编码的 data URL
        user_msgs.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(obs["screenshot"]),
                    "detail": "auto",  # 图像细节级别："low", "high", "auto"
                },
            }
        )

    # 添加动作空间描述
    # 这告诉 LLM 有哪些可用的动作以及如何使用它们
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Action Space

{action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
        }
    )

    # 添加历史动作记录（如果有）和最后的错误信息
    # 这帮助 LLM 了解之前的操作历史，避免重复错误
    if action_history:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# History of past actions
""",
            }
        )
        # 将每个历史动作添加为单独的消息
        user_msgs.extend(
            [
                {
                    "type": "text",
                    "text": f"""\

{action}
""",
                }
                for action in action_history
            ]
        )

        # 如果上一个动作有错误，添加错误信息
        # 这帮助 LLM 理解为什么上一个动作失败，并避免重复同样的错误
        if obs["last_action_error"]:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                }
            )

    return user_msgs


def build_propose_prompt(
    obs: dict,
    action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False,
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    """构建动作提案提示词
    
    这个函数构建用于生成下一步动作的提示词。它包含：
    1. 系统指令：告诉 LLM 如何生成动作
    2. 当前状态信息：目标、页面结构、历史动作等
    3. 动作请求：要求 LLM 生成下一步动作
    
    用于搜索算法中的动作生成阶段（如 MCTS 的扩展阶段）。
    
    Args:
        obs: 浏览器环境的观察数据字典
        action_set: 高级动作集对象
        action_history: 历史动作列表
        use_axtree: 是否包含可访问性树信息（默认：True）
        use_html: 是否包含 HTML 信息（默认：False）
        use_screenshot: 是否包含截图信息（默认：False）
        
    Returns:
        (system_msgs, user_msgs, full_prompt_txt) 元组：
        - system_msgs: 系统消息列表
        - user_msgs: 用户消息列表
        - full_prompt_txt: 完整的提示词文本（用于日志记录）
        
    Raises:
        AssertionError: 如果观察数据中缺少目标信息
        ValueError: 如果消息类型未知
    """
    system_msgs = []
    user_msgs = []

    assert obs["goal_object"], "The goal is missing."
    
    # 添加系统指令：告诉 LLM 如何生成动作
    system_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
        }
    )

    # 添加当前状态的所有信息（目标、标签页、页面结构、动作空间、历史动作等）
    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, use_axtree, use_html, use_screenshot))

    # 请求生成下一步动作
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. Make sure to fill in ALL PARAMETERS of the action. 
""",
        }
    )

    # 将消息列表转换为文本字符串（用于日志记录和调试）
    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                # 文本消息直接添加
                prompt_text_strings.append(message["text"])
            case "image_url":
                # 图像消息转换为文本描述（因为日志中不能直接显示图像）
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    # base64 编码的图像 URL 会被截断，只显示前 30 个字符
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )
    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt


def build_evaluation_prompt(
    obs: dict,
    action: str, action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False,
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    """构建动作评估提示词
    
    这个函数构建用于评估动作好坏的提示词。它包含：
    1. 系统指令：告诉 LLM 如何评估动作（返回 0-10 的分数和推理过程）
    2. 当前状态信息：目标、页面结构、历史动作等
    3. 待评估的动作
    4. 评估请求：要求 LLM 评估动作并返回 JSON 格式的结果
    
    用于搜索算法中的奖励计算阶段（如 MCTS 的 fast_reward 或 reward）。
    
    Args:
        obs: 浏览器环境的观察数据字典
        action: 待评估的动作字符串
        action_set: 高级动作集对象
        action_history: 历史动作列表
        use_axtree: 是否包含可访问性树信息（默认：True）
        use_html: 是否包含 HTML 信息（默认：False）
        use_screenshot: 是否包含截图信息（默认：False）
        
    Returns:
        (system_msgs, user_msgs, full_prompt_txt) 元组：
        - system_msgs: 系统消息列表
        - user_msgs: 用户消息列表
        - full_prompt_txt: 完整的提示词文本（用于日志记录）
        
    Raises:
        AssertionError: 如果观察数据中缺少目标信息
        ValueError: 如果消息类型未知
        
    注意:
        LLM 应该返回 JSON 格式的评估结果：
        {
            "reasoning": "评估推理过程",
            "score": 0-10 之间的分数
        }
    """
    system_msgs = []
    user_msgs = []

    assert obs["goal_object"], "The goal is missing."
    
    # 添加系统指令：告诉 LLM 如何评估动作
    # 要求返回 0-10 的分数和推理过程，格式为 JSON
    system_msgs.append(
        {
            "type": "text",
            "text": """\
# Instructions

Review the current state of the page along with a proposed action and determine how promising it is towards completing the goal. Provide a score between 0 and 10 along with your reasoning in a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    # 添加当前状态的所有信息（目标、标签页、页面结构、动作空间、历史动作等）
    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, use_axtree, use_html, use_screenshot))

    # 添加待评估的动作
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Proposed action

{action}
""",
        }
    )

    # 请求评估动作
    user_msgs.append(
        {
            "type": "text",
            "text": """\
# Evaluation of Proposed Action

As mentioned before, considering all the information above in the context of the goal, evaluate the proposed action by providing a score from 0 to 10 along with your reasoning. Use a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    # 将消息列表转换为文本字符串（用于日志记录和调试）
    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                # 文本消息直接添加
                prompt_text_strings.append(message["text"])
            case "image_url":
                # 图像消息转换为文本描述（因为日志中不能直接显示图像）
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    # base64 编码的图像 URL 会被截断，只显示前 30 个字符
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )

    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt
