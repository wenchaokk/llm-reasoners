"""积木世界（Blocksworld）问题的世界模型实现

这个文件展示了如何为一个具体的推理任务实现 WorldModel。
积木世界问题：给定初始积木排列和目标排列，找到移动积木的序列。
"""

from typing import NamedTuple
import reasoners.benchmark.bw_utils as utils
from reasoners import WorldModel, LanguageModel
import copy

# 动作类型：字符串，例如 "pick up the red block"
BWAction = str

class BWState(NamedTuple):
    """积木世界的状态表示
    
    状态包含：
        step_idx: 当前步骤索引
        last_blocks_state: 上一个积木状态（用于缓冲机制）
        blocks_state: 当前积木状态（字符串描述）
        buffered_action: 缓冲的动作（用于每两步更新一次状态）
    
    注意：这个实现使用了一个缓冲机制，每两个动作才更新一次积木状态。
    这样可以减少 LLM 调用的次数，提高效率。
    """
    step_idx: int              # 当前步骤索引
    last_blocks_state: str     # 上一个积木状态
    blocks_state: str          # 当前积木状态（字符串描述，如 "红色在桌上，蓝色在红色上"）
    buffered_action: BWAction  # 缓冲的动作


class BlocksWorldModel(WorldModel):
    """积木世界模型
    
    这个类实现了积木世界问题的状态转换逻辑。
    
    状态表示：
        - step_idx: 步骤索引
        - last_blocks_state: 上一个积木状态
        - blocks_state: 当前积木状态（字符串）
        - buffered_action: 缓冲的动作
    
    动作示例：
        - "pick up the red block"（拿起红色积木）
        - "put down the blue block"（放下蓝色积木）
        - "stack the red block on the blue block"（把红色积木叠在蓝色积木上）
    
    设计说明：
        使用缓冲机制：每两个动作才更新一次积木状态。
        这样可以减少 LLM 调用次数，提高效率。
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=2) -> None:
        """初始化积木世界模型
        
        Args:
            base_model: LLM 模型（用于预测状态转换）
            prompt: 提示词模板字典
            max_steps: 最大步数（防止无限搜索）
            batch_size: 批处理大小
        """
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> BWState:
        """初始化状态
        
        从问题中提取初始积木状态，创建初始状态对象。
        
        Returns:
            初始状态（步骤索引为 0，积木状态从问题中提取）
        """
        return BWState(
            step_idx=0, 
            last_blocks_state="", 
            blocks_state=utils.extract_init_state(self.example),  # 从问题中提取初始状态
            buffered_action=""
        )

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """执行动作，转换到下一个状态
        
        这是世界模型的核心方法。它：
        1. 执行动作（更新积木状态）
        2. 管理缓冲机制（每两个动作更新一次）
        3. 检查目标达成度（用于奖励计算）
        
        Args:
            state: 当前状态
            action: 要执行的动作
            
        Returns:
            (新状态, 辅助信息字典)
            辅助信息包含 goal_reached，会被传递给 SearchConfig.reward 方法
        """
        state = copy.deepcopy(state)  # 深拷贝，避免修改原状态
        buffered_action = state.buffered_action
        blocks_state = state.blocks_state
        step_idx = state.step_idx
        
        # 更新积木状态（使用 LLM 预测状态转换）
        blocks_state = self.update_blocks(blocks_state, action)
        
        # 缓冲机制：每两个动作更新一次状态
        if state.buffered_action == "":
            # 如果没有缓冲动作，将当前动作放入缓冲
            new_buffered_action = action
        else:
            # 如果有缓冲动作，清空缓冲（因为已经更新了状态）
            new_buffered_action = ""

        # 创建新状态
        new_state = BWState(
            step_idx=step_idx+1, 
            last_blocks_state=state.blocks_state,
            blocks_state=blocks_state, 
            buffered_action=new_buffered_action
        )
        
        # 检查目标达成度（用于奖励计算）
        goal_reached = utils.goal_check(
            utils.extract_goals(self.example), 
            blocks_state
        )
        
        return new_state, {"goal_reached": goal_reached}

    def update_blocks(self, block_states: str, action: BWAction) -> str:
        """使用动作更新积木状态
        
        这个方法使用 LLM 来预测执行动作后的新状态。
        这是"世界模型"的核心：用 LLM 来模拟物理世界的状态转换。
        
        Args:
            block_states: 当前积木状态（字符串描述）
            action: 要执行的动作
            
        Returns:
            更新后的积木状态（字符串描述）
        """
        # 根据动作类型选择对应的提示词模板
        if "pick" in action:
            key = "world_update_pickup"      # 拿起积木
        elif "unstack" in action:
            key = "world_update_unstack"      # 从积木堆中取出
        elif "put" in action:
            key = "world_update_putdown"      # 放下积木
        elif "stack" in action:
            key = "world_update_stack"       # 叠放积木
        else:
            raise ValueError("Invalid action")
        
        # 构建提示词：告诉 LLM 当前状态和要执行的动作
        world_update_prompt = self.prompt[key].format(
            block_states, 
            action.capitalize() + "."
        )
        
        # 使用 LLM 预测新状态
        world_output = self.base_model.generate(
            [world_update_prompt],
            eos_token_id="\n",      # 遇到换行符停止
            hide_input=True,         # 只返回生成的部分
            temperature=0            # 贪心解码（确定性）
        ).text[0].strip()
        
        # 应用 LLM 的输出到当前状态，得到新状态
        new_state = utils.apply_change(world_output, block_states)
        return new_state

    def is_terminal(self, state: BWState) -> bool:
        """判断状态是否为终止状态
        
        终止条件：
        1. 达到目标状态（所有目标都满足）
        2. 达到最大步数（防止无限搜索）
        
        Args:
            state: 要判断的状态
            
        Returns:
            如果为终止状态返回 True，否则返回 False
        """
        # 检查是否达到目标状态
        goal_reached = utils.goal_check(
            utils.extract_goals(self.example), 
            state.blocks_state
        )[0]  # [0] 是布尔值，表示是否完全达成目标
        
        if goal_reached:
            return True
        # 检查是否达到最大步数（防止无限搜索）
        elif state.step_idx == self.max_steps:
            return True
        return False
