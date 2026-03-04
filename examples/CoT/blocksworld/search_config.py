"""积木世界问题的搜索配置实现

这个文件展示了如何为一个具体的推理任务实现 SearchConfig。
它定义了：
1. 如何生成可用动作（get_actions）
2. 如何快速评估动作（fast_reward）
3. 如何完整评估动作（reward）
"""

import numpy as np

import reasoners.benchmark.bw_utils as utils
from world_model import BWState, BWAction
from reasoners import SearchConfig, LanguageModel

class BWConfig(SearchConfig):
    """积木世界的搜索配置
    
    这个类定义了：
    1. 动作空间：在给定状态下有哪些可用动作
    2. 奖励函数：如何评估一个动作的好坏
    
    奖励设计：
        - 结合"直觉"（LLM 的 log-likelihood）和"目标达成度"
        - fast_reward: 只用直觉（快速）
        - reward: 结合直觉和目标达成度（完整）
    """
    
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size=2,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100) -> None:
        """初始化搜索配置
        
        Args:
            base_model: LLM 模型（用于计算 log-likelihood）
            prompt: 提示词模板字典
            batch_size: 批处理大小
            reward_alpha: 奖励混合系数（直觉 vs 目标达成度）
            goal_reward_default: 默认目标奖励（当目标未达成时）
            goal_reached_reward: 目标达成时的奖励（很大的正数）
        """
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha          # 奖励混合系数
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def get_actions(self, state: BWState) -> list[BWAction]:
        """获取在给定状态下的所有可用动作
        
        这个方法定义了动作空间。对于积木问题，我们使用规则来生成所有合法动作。
        （例如：只能拿起"clear"的积木，只能把积木放在"clear"的位置上）
        
        Args:
            state: 当前状态
            
        Returns:
            可用动作列表（例如：["pick up the red block", "stack the blue block on the red block"]）
        """
        blocks_state = state.blocks_state
        # 使用工具函数根据当前积木状态生成所有合法动作
        return utils.generate_all_actions(blocks_state)

    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        """快速评估动作的奖励
        
        这个方法应该快速执行，因为它会在搜索过程中被频繁调用。
        它使用 LLM 的 log-likelihood 来评估动作的合理性，而不需要实际执行动作。
        
        评估方法：
        1. 直觉（intuition）：动作在给定上下文下的 log-likelihood
        2. 自评估（self_eval）：LLM 认为这个动作是否"好"
        
        Args:
            state: 当前状态
            action: 要评估的动作
            
        Returns:
            (奖励值, 详细信息字典)
            详细信息会被传递给 reward 方法，避免重复计算
        """
        # 处理缓冲机制：如果有缓冲动作，使用上一个状态
        if state.buffered_action == "":
            # 如果没有缓冲动作，使用当前状态
            current_blocks_state = state.blocks_state
        else:
            # 如果有缓冲动作，使用上一个状态（因为当前状态还没更新）
            current_blocks_state = state.last_blocks_state
        
        previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
        
        # 根据步骤索引选择对应的 few-shot 提示词模板
        # 每两步减少一个示例，使步骤长度分布更合理
        icl_template = self.prompt["icl_list"][state.step_idx // 2]
        
        # 构建 in-context learning 提示词
        inputs = icl_template.replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))\
            .replace("<action>", previous_action)
        
        # 计算"直觉"：动作在给定上下文下的 log-likelihood
        # 这表示 LLM 认为这个动作在给定上下文中出现的可能性
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        # 构建自评估提示词：询问 LLM 这个动作是否"好"
        self_eval_prompt = self.prompt["self-eval"]\
            .replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))\
            .replace("<action>", action)
        
        # 计算自评估：LLM 认为这个动作是"good"的可能性
        self_eval = self.base_model.get_loglikelihood(
            self_eval_prompt, 
            [self_eval_prompt + "good"]
        )[0]

        # 计算奖励（此时还不知道目标达成度，使用默认值）
        reward = self.calculate_reward(intuition, self_eval)
        
        # 返回奖励和详细信息（详细信息会被传递给 reward 方法）
        return reward, {'intuition': intuition, "self_eval": self_eval}

    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        """计算奖励的统一接口
        
        奖励公式：
            reward = (intuition + self_eval) * alpha + goal_reward * (1 - alpha)
        
        其中：
            - intuition + self_eval: LLM 的评估（直觉 + 自评估）
            - goal_reward: 目标达成度
            - alpha: 混合系数
        
        Args:
            intuition: 直觉值（log-likelihood）
            self_eval: 自评估值（log-likelihood）
            goal_reached: 目标达成度（None 或 (bool, float) 元组）
            
        Returns:
            奖励值
        """
        # 根据目标达成度计算目标奖励
        if goal_reached is None:
            # 如果不知道目标达成度（fast_reward 时），使用默认值
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            # 如果完全达成目标，给予大奖励
            goal_reward = self.goal_reached_reward
        else:
            # 如果部分达成目标，使用达成度（0-1 之间的浮点数）
            goal_reward = goal_reached[1]
        
        # 混合直觉和目标达成度
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: BWState, action: BWAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> tuple[float, dict]:
        """完整评估动作的奖励
        
        这个方法在动作执行后调用，此时已经知道目标达成度。
        它结合 fast_reward 的结果和目标达成度来计算最终奖励。
        
        设计理念：
            - fast_reward 中已经计算了 intuition 和 self_eval
            - WorldModel.step 中已经计算了 goal_reached
            - 通过 **kwargs 传递，避免重复计算
        
        Args:
            state: 当前状态
            action: 要评估的动作
            intuition: 直觉值（从 fast_reward 传递过来）
            self_eval: 自评估值（从 fast_reward 传递过来）
            goal_reached: 目标达成度（从 WorldModel.step 传递过来）
            
        Returns:
            (奖励值, 详细信息字典)
        """
        # 检查必需参数（这些应该从 fast_reward 和 WorldModel.step 传递过来）
        assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        
        # 计算最终奖励（结合直觉、自评估和目标达成度）
        reward = self.calculate_reward(intuition, self_eval, goal_reached)
        
        return reward, {'intuition': intuition, 'goal_reached': goal_reached}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
