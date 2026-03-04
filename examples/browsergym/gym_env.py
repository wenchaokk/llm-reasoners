"""
BrowserGym 环境适配器

这个模块将 Gymnasium 环境适配到 LLM Reasoners 框架中。
它实现了 Environment 接口，使得 Gymnasium 环境（如 BrowserGym）可以
与搜索算法（如 MCTS、Beam Search）无缝配合使用。

主要功能：
1. 将 Gymnasium 环境的状态转换为 LLM Reasoners 框架的状态
2. 处理环境的状态转换和回退（backtracking）
3. 提供观察数据预处理功能
"""

import gymnasium as gym
from typing import NamedTuple, Optional, Callable, Any
from reasoners import Environment

# 动作类型：可以是任何类型（由具体的 Gymnasium 环境定义）
ActionGym = Any


class StateGym(NamedTuple):
    """Gymnasium 环境的状态表示
    
    这个类封装了 Gymnasium 环境的状态信息，包括：
    - 步骤索引：当前步骤数
    - 动作历史：用于重建环境状态（用于回退）
    - 观察数据：当前和上一个观察
    - 奖励和终止标志：环境的反馈信息
    
    Attributes:
        step_idx: 当前步骤索引（从 0 开始）
        action_history: 动作历史列表，用于重建环境状态（用于回退到之前的状态）
        last_obs: 上一个观察数据（字典格式）
        current_obs: 当前观察数据（字典格式，来自 env.step() 的输出）
        reward: 当前步骤的奖励值
        terminated: 是否正常终止（任务完成或失败）
        truncated: 是否被截断（达到最大步数等）
    """
    step_idx: int
    action_history: list[ActionGym]  # 动作历史，用于重建环境状态（回退）
    last_obs: dict                   # 上一个观察数据
    current_obs: dict                # 当前观察数据（来自 env.step() 的输出）
    reward: float                     # 当前步骤的奖励值
    terminated: bool                 # 是否正常终止
    truncated: bool                   # 是否被截断


class EnvironmentGym(Environment):
    """Gymnasium 环境的适配器
    
    这个类将 Gymnasium 环境适配到 LLM Reasoners 框架中。
    与 WorldModel 不同，它不是基于文本示例，而是直接使用 Gymnasium 环境。
    LLM 不会用于生成新状态，而是由 Gymnasium 环境的 step 函数处理状态转换。
    
    主要特点：
    1. 直接使用 Gymnasium 环境的状态转换（不需要 LLM 预测）
    2. 支持观察数据预处理（如将截图转换为 base64 URL）
    3. 支持状态回退（通过重放动作历史）
    
    Attributes:
        env: Gymnasium 环境对象
        env_seed: 环境的随机种子（用于可重复性）
        max_steps: 最大步数（超过此步数会被截断）
        obs_preprocessor: 可选的观察数据预处理函数
        env_current_obs: 环境的当前观察数据（用于检查状态对齐）
    """

    def __init__(self, env: gym.Env, env_seed: int = 42, max_steps=20, obs_preprocessor: Optional[Callable[[dict], dict]] = None):
        """初始化 Gymnasium 环境适配器
        
        Args:
            env: Gymnasium 环境对象（如 BrowserGym 环境）
            env_seed: 环境的随机种子，用于确保可重复性（默认：42）
            max_steps: 最大步数，超过此步数任务会被截断（默认：20）
            obs_preprocessor: 可选的观察数据预处理函数
                            用于在存储到状态之前处理观察数据
                            例如：将截图转换为 base64 URL
        """
        self.env = env
        self.env_seed = env_seed
        self.obs_preprocessor = obs_preprocessor
        self.max_steps = max_steps
        self.env_current_obs: dict = None  # 环境的当前观察数据（用于状态对齐检查）

    def init_state(self) -> StateGym:
        """初始化环境状态
        
        重置 Gymnasium 环境并返回初始状态。
        这是搜索算法的起点。
        
        Returns:
            初始状态对象，包含：
            - step_idx: 0（初始步骤）
            - action_history: 空列表（还没有执行任何动作）
            - last_obs: 空字典（初始状态没有上一个观察）
            - current_obs: 环境的初始观察数据
            - reward: 0（初始奖励）
            - terminated: False（未终止）
            - truncated: False（未截断）
        """
        # 重置环境，使用指定的随机种子
        obs, env_info = self.env.reset(seed=self.env_seed)
        
        # 如果提供了观察数据预处理函数，对观察数据进行预处理
        # 例如：将截图转换为 base64 URL，以便传递给 LLM
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        
        # 更新环境的当前观察数据
        self.env_current_obs = obs

        # 返回初始状态
        return StateGym(
            step_idx=0,           # 初始步骤索引
            last_obs={},          # 没有上一个观察
            current_obs=obs,      # 当前观察（环境的初始观察）
            action_history=[],    # 空的动作历史
            reward=0,             # 初始奖励为 0
            terminated=False,     # 未终止
            truncated=False       # 未截断
        )

    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        """执行动作，转换到下一个状态
        
        这个方法执行动作并返回新状态。需要注意的是，环境可能不与传入的状态对齐。
        如果环境的当前状态（self.env_current_obs）与传入的状态不一致，需要进行回退。
        
        回退机制：
            基本实现比较朴素，它会重置环境并重放状态中的动作历史。
            对于某些环境，可能有更高效的回退方法（如保存检查点）。
            但对于大多数情况，这个实现已经足够。
        
        工作流程：
            1. 检查环境状态是否与传入状态对齐
            2. 如果不对齐，重置环境并重放动作历史（回退）
            3. 执行动作
            4. 预处理观察数据（如果提供了预处理函数）
            5. 创建并返回新状态
        
        Args:
            state: 当前状态（要从此状态执行动作）
            action: 要执行的动作
            
        Returns:
            (next_state, aux) 元组：
            - next_state: 执行动作后的新状态
            - aux: 辅助信息字典，包含环境的奖励值
                   这个奖励会被传递给搜索算法，然后传递给 SearchConfig 的 reward 函数
        """
        
        # 检查环境状态是否与传入状态对齐
        # 如果不对齐，说明搜索算法在探索不同的路径，需要回退到指定状态
        if self.env_current_obs != state.current_obs:
            # 重置环境到初始状态
            self.env.reset(seed=self.env_seed)
            # 重放动作历史，使环境回到指定状态
            # 这是回退机制：通过重放动作历史来重建环境状态
            for action in state.action_history:
                self.env.step(action)

        # 执行动作，获取新观察、奖励和终止信息
        obs, reward, terminated, truncated, step_info = self.env.step(action)
        
        # 如果提供了观察数据预处理函数，对观察数据进行预处理
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        
        # 更新环境的当前观察数据
        self.env_current_obs = obs

        # 创建新状态
        next_state = StateGym(
            step_idx=state.step_idx + 1,                    # 步骤索引加 1
            last_obs=state.current_obs,                     # 上一个观察是当前状态的观察
            current_obs=obs,                                # 当前观察是执行动作后的新观察
            action_history=state.action_history + [action], # 将新动作添加到动作历史
            reward=reward,                                   # 环境的奖励值
            terminated=terminated,                          # 是否正常终止
            truncated=truncated                             # 是否被截断
        )

        # 返回新状态和辅助信息（包含环境奖励）
        return next_state, {"env_reward": reward}

    def is_terminal(self, state: StateGym) -> bool:
        """判断状态是否为终止状态
        
        终止条件：
        1. 环境正常终止（terminated=True）：任务完成或失败
        2. 环境被截断（truncated=True）：达到环境的最大步数等
        3. 达到最大步数（step_idx >= max_steps）：防止无限搜索
        
        Args:
            state: 要判断的状态
            
        Returns:
            如果为终止状态返回 True，否则返回 False
        """
        return state.terminated or state.truncated or state.step_idx >= self.max_steps
