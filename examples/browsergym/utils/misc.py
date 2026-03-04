"""
BrowserGym 工具函数模块

这个模块提供了 BrowserGym 环境中使用的工具函数，包括：
1. 图像处理：将图像转换为 base64 编码的 URL（用于可视化）
2. 观察预处理：将浏览器环境的观察数据转换为适合 LLM 处理的格式
3. 动作验证：检查动作提案是否有效
"""

import base64
import io
import numpy as np
from PIL import Image

from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.core.action.parsers import highlevel_action_parser


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """将图像转换为 base64 编码的 JPEG URL
    
    这个函数将图像（numpy 数组或 PIL Image）转换为 base64 编码的 data URL，
    可以直接嵌入到 HTML 或 JSON 中，用于可视化或传递给 LLM。
    
    处理流程：
    1. 如果是 numpy 数组，转换为 PIL Image
    2. 如果是 RGBA 或 LA 模式，转换为 RGB（JPEG 不支持透明度）
    3. 保存为 JPEG 格式到内存缓冲区
    4. 编码为 base64 字符串
    5. 返回 data URL 格式
    
    Args:
        image: 输入图像，可以是 numpy 数组或 PIL Image 对象
        
    Returns:
        base64 编码的 data URL 字符串，格式为 "data:image/jpeg;base64,{base64_string}"
        
    示例:
        >>> img = np.array([...])  # 截图数组
        >>> url = image_to_jpg_base64_url(img)
        >>> # url = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    """
    # 如果是 numpy 数组，转换为 PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # JPEG 格式不支持透明度，需要将 RGBA 或 LA 模式转换为 RGB
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    # 将图像保存到内存缓冲区，格式为 JPEG
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        # 将缓冲区内容编码为 base64 字符串
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # 返回 data URL 格式的字符串
    return f"data:image/jpeg;base64,{image_base64}"


def obs_preprocessor(obs: dict) -> dict:
    """预处理浏览器环境的观察数据
    
    这个函数将 BrowserGym 环境的原始观察数据转换为适合 LLM 处理的格式。
    主要处理包括：
    1. 将截图转换为 base64 URL（用于可视化）
    2. 将 DOM 树和可访问性树转换为文本格式（用于 LLM 理解页面结构）
    3. 保留其他重要信息（聊天消息、目标、动作历史等）
    
    观察数据包含的信息：
    - chat_messages: 聊天消息历史
    - screenshot: 当前页面截图（转换为 base64 URL）
    - goal_object: 任务目标
    - last_action: 上一个执行的动作
    - last_action_error: 上一个动作的错误信息（如果有）
    - open_pages_urls: 打开的页面 URL 列表
    - open_pages_titles: 打开的页面标题列表
    - active_page_index: 当前活动页面的索引
    - axtree_txt: 可访问性树文本（用于理解页面结构）
    - pruned_html: 修剪后的 HTML 文本（移除不必要的内容）
    
    Args:
        obs: 浏览器环境的原始观察数据字典
        
    Returns:
        预处理后的观察数据字典，格式适合传递给 LLM
        
    注意:
        - axtree_txt 和 pruned_html 是文本格式，便于 LLM 理解页面结构
        - screenshot 转换为 base64 URL，可以直接嵌入到提示词中用于多模态理解
    """
    return {
        "chat_messages": obs["chat_messages"],  # 聊天消息历史
        "screenshot": image_to_jpg_base64_url(obs["screenshot"]),  # 截图转换为 base64 URL
        "goal_object": obs["goal_object"],  # 任务目标
        "last_action": obs["last_action"],  # 上一个执行的动作
        "last_action_error": obs["last_action_error"],  # 上一个动作的错误信息
        "open_pages_urls": obs["open_pages_urls"],  # 打开的页面 URL 列表
        "open_pages_titles": obs["open_pages_titles"],  # 打开的页面标题列表
        "active_page_index": obs["active_page_index"],  # 当前活动页面的索引
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),  # 可访问性树文本（扁平化）
        "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),  # 修剪后的 HTML 文本
    }


# ============================================================================
# WebArena 动作集定义
# 这些是浏览器环境中支持的动作类型
# 注意：当前硬编码为 WebArena 动作集，如果需要支持其他动作集，需要修改此列表
# ============================================================================
valid_action_types = [
    "noop",              # 无操作（等待）
    "scroll",            # 滚动页面
    "keyboard_press",    # 键盘按键
    "click",             # 点击元素
    "fill",              # 填写输入框
    "hover",             # 鼠标悬停
    "tab_focus",         # 切换标签页焦点
    "new_tab",           # 打开新标签页
    "go_back",           # 浏览器后退
    "go_forward",        # 浏览器前进
    "goto",              # 导航到 URL
    "tab_close",         # 关闭标签页
    "select_option",     # 选择下拉选项
    "send_msg_to_user",  # 向用户发送消息
    "report_infeasible"  # 报告任务不可行
]


def check_validity_of_action_proposal(action_proposal: str):
    """检查动作提案的有效性
    
    这个函数验证 LLM 生成的动作提案是否包含有效的动作类型。
    它会解析动作提案字符串，提取所有函数调用，然后检查每个函数名
    是否在预定义的有效动作类型列表中。
    
    工作流程：
    1. 使用高级动作解析器从字符串中提取函数调用
    2. 将解析结果展平为列表
    3. 检查是否至少有一个函数调用
    4. 检查每个函数名是否在有效动作类型列表中
    
    Args:
        action_proposal: LLM 生成的动作提案字符串
                        例如："click('button_id')" 或 "fill('input_id', 'text')"
        
    Returns:
        bool: 如果所有动作都是有效的返回 True，否则返回 False
        
    示例:
        >>> check_validity_of_action_proposal("click('submit_btn')")
        True
        >>> check_validity_of_action_proposal("invalid_action('arg')")
        False
        >>> check_validity_of_action_proposal("click('btn1') fill('input1', 'text')")
        True
        
    注意:
        - 如果动作提案为空或没有解析出任何函数调用，返回 False
        - 只要有一个动作无效，整个提案就被视为无效
    """
    # 使用高级动作解析器从字符串中提取函数调用
    # 这会识别类似 "click('id')" 或 "fill('id', 'value')" 这样的函数调用
    function_calls = highlevel_action_parser.search_string(action_proposal)
    
    # 将解析结果展平为列表格式：[(function_name, function_args), ...]
    function_calls = sum(function_calls.as_list(), [])

    # 如果没有解析出任何函数调用，动作提案无效
    if len(function_calls) == 0:
        return False

    # 检查每个函数调用是否使用有效的动作类型
    for function_name, function_args in function_calls:
        if function_name not in valid_action_types:
            # 如果发现无效的动作类型，整个提案无效
            return False

    # 所有动作都有效
    return True
