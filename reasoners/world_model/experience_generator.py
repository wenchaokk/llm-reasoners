"""经验生成器：不使用 ChatGPT 生成 WorldModel 经验

提供多种方法生成经验：
1. 基于规则生成（Blocksworld 有明确的规则）
2. 从已有数据集提取
3. 使用本地模型生成
"""

import os
import json
import random
import re
from typing import List, Tuple, Optional
from reasoners.world_model.memory_store import MemoryStore, StateTransition
import reasoners.benchmark.bw_utils as bw_utils


class RuleBasedExperienceGenerator:
    """基于规则的经验生成器（Blocksworld）
    
    使用 Blocksworld 的规则来生成状态转换经验，不需要 LLM。
    """
    
    def __init__(self):
        """初始化规则生成器"""
        pass
    
    def _apply_action_rule(self, state: str, action: str) -> str:
        """使用规则应用动作，返回新状态
        
        使用 bw_utils.apply_change，但需要先生成 LLM 格式的变化描述。
        
        Args:
            state: 当前状态字符串
            action: 动作字符串
            
        Returns:
            新状态字符串
        """
        # 根据动作类型生成 LLM 格式的变化描述
        # 注意：状态格式应该是 "hand is empty" 或 "is holding the X block"
        if "pick up" in action:
            color = re.search(r"pick up the (\w+) block", action)
            if color:
                color = color.group(1)
                # 检查积木是在桌上还是在其他积木上
                if f"the {color} block is on the table" in state:
                    change = f"the hand was empty and is now holding the {color} block, and the {color} block was on the table and is no longer on the table"
                else:
                    # 从其他积木上取下
                    below_match = re.search(f"the {color} block is on top of the (\\w+) block", state)
                    if below_match:
                        below_color = below_match.group(1)
                        change = f"the hand was empty and is now holding the {color} block, and the {color} block was on top of the {below_color} block and is no longer on top of the {below_color} block, and the {below_color} block is now clear"
                    else:
                        return state  # 无法确定位置
            else:
                return state
        
        elif "unstack" in action:
            match = re.search(r"unstack the (\w+) block from on top of the (\w+) block", action)
            if match:
                color, below_color = match.groups()
                change = f"the hand was empty and is now holding the {color} block, and the {color} block was on top of the {below_color} block and is no longer on top of the {below_color} block, and the {below_color} block is now clear"
            else:
                return state
        
        elif "put down" in action:
            color = re.search(r"put down the (\w+) block", action)
            if color:
                color = color.group(1)
                # 检查手是否持有这个积木
                if f"is holding the {color} block" in state:
                    change = f"the hand was holding the {color} block and is now empty, and the {color} block is now on the table"
                else:
                    return state  # 手没有持有这个积木
            else:
                return state
        
        elif "stack" in action:
            match = re.search(r"stack the (\w+) block on top of the (\w+) block", action)
            if match:
                color, below_color = match.groups()
                # 检查手是否持有这个积木
                if f"is holding the {color} block" in state:
                    # 检查积木是否在桌上
                    if f"the {color} block is on the table" in state:
                        change = f"the hand was holding the {color} block and is now empty, and the {color} block was on the table and is no longer on the table, and the {color} block is now on top of the {below_color} block, and the {below_color} block was clear and is no longer clear"
                    else:
                        # 可能已经在其他位置，简化处理
                        change = f"the hand was holding the {color} block and is now empty, and the {color} block is now on top of the {below_color} block, and the {below_color} block was clear and is no longer clear"
                else:
                    return state  # 手没有持有这个积木
            else:
                return state
        else:
            return state
        
        # 使用 bw_utils.apply_change 应用变化
        try:
            next_state = bw_utils.apply_change(change, state)
            # 验证生成的状态是否有效
            if next_state and len(next_state.strip()) > 0:
                return next_state
            else:
                return state
        except Exception as e:
            # 静默失败，返回原状态（会被验证器过滤）
            return None  # 返回 None 表示失败
    
    def generate_from_state(self, state: str, max_depth: int = 5, max_experiences: int = None) -> List[Tuple[str, str, str]]:
        """从给定状态生成经验
        
        Args:
            state: 初始状态
            max_depth: 最大深度（生成多少步经验）
            max_experiences: 最大经验数量（如果设置，会生成多个路径）
            
        Returns:
            List of (state, action, next_state) tuples
        """
        experiences = []
        current_state = state
        visited_states = set()
        
        # 如果设置了 max_experiences，生成多个路径
        if max_experiences:
            for path_idx in range(max_experiences // max_depth + 1):
                if len(experiences) >= max_experiences:
                    break
                
                # 重置状态（从初始状态开始新路径）
                current_state = state
                path_experiences = []
                
                for _ in range(max_depth):
                    if len(experiences) >= max_experiences:
                        break
                    
                    # 生成所有可能的动作
                    actions = bw_utils.generate_all_actions(current_state)
                    
                    if not actions:
                        break
                    
                    # 随机选择一个动作
                    action = random.choice(actions)
                    
                    # 应用动作规则
                    next_state = self._apply_action_rule(current_state, action)
                    
                    # 如果应用失败，跳过
                    if next_state is None or next_state == current_state:
                        break
                    
                    # 避免循环
                    state_key = (current_state, action, path_idx)
                    if state_key in visited_states:
                        break
                    visited_states.add(state_key)
                    
                    # 添加到经验列表
                    path_experiences.append((current_state, action, next_state))
                    current_state = next_state
                
                experiences.extend(path_experiences)
        else:
            # 原始逻辑：单一路径
            for _ in range(max_depth):
                # 生成所有可能的动作
                actions = bw_utils.generate_all_actions(current_state)
                
                if not actions:
                    break
                
                # 随机选择一个动作（或选择第一个）
                action = random.choice(actions)
                
                # 应用动作规则
                next_state = self._apply_action_rule(current_state, action)
                
                # 如果应用失败，跳过
                if next_state is None or next_state == current_state:
                    break
                
                # 避免循环
                state_key = (current_state, action)
                if state_key in visited_states:
                    break
                visited_states.add(state_key)
                
                # 添加到经验列表
                experiences.append((current_state, action, next_state))
                
                current_state = next_state
        
        return experiences
    
    def generate_from_dataset(self, data_file: str, num_examples: int = 10, experiences_per_example: int = 5) -> List[Tuple[str, str, str]]:
        """从数据集中提取经验
        
        Args:
            data_file: 数据集文件路径
            num_examples: 使用多少个示例
            experiences_per_example: 每个示例生成多少条经验
            
        Returns:
            List of (state, action, next_state) tuples
        """
        experiences = []
        
        if not os.path.exists(data_file):
            print(f"警告：找不到数据文件 {data_file}")
            return experiences
        
        # 使用 load_blocksworld 加载数据（需要 config_file 和 domain_file）
        try:
            # 查找 config_file 和 domain_file
            config_file = os.path.join(os.path.dirname(data_file), "..", "..", "data", "bw_config.yaml")
            if not os.path.exists(config_file):
                # 尝试其他路径
                config_file = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "CoT", "blocksworld", "data", "bw_config.yaml")
            
            domain_file = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "CoT", "blocksworld", "data", "generated_domain.pddl")
            
            if not os.path.exists(config_file) or not os.path.exists(domain_file):
                # 如果找不到配置文件，尝试直接使用原始数据格式
                with open(data_file, 'r') as f:
                    raw_data = json.load(f)
                
                # 原始数据格式是列表，每个元素是 [instance_file, plan_code, step_count]
                # 我们需要使用 load_blocksworld 来转换
                print(f"警告：找不到配置文件，跳过数据集提取")
                return experiences
            
            # 使用 load_blocksworld 加载
            examples = bw_utils.load_blocksworld(config_file, domain_file, data_file)
            
            # 限制示例数量
            examples = examples[:num_examples]
            
            for example in examples:
                # 提取初始状态
                init_state = bw_utils.extract_init_state(example)
                
                # 生成经验（可以生成多条路径）
                exp = self.generate_from_state(init_state, max_depth=5, max_experiences=experiences_per_example)
                experiences.extend(exp)
        
        except Exception as e:
            print(f"警告：加载数据集失败: {e}")
            import traceback
            traceback.print_exc()
            return experiences
        
        return experiences


class LocalModelExperienceGenerator:
    """使用本地模型生成经验"""
    
    def __init__(self, local_model):
        """初始化本地模型生成器
        
        Args:
            local_model: 本地语言模型（如 LlamaCppModel）
        """
        self.local_model = local_model
    
    def generate_from_state(self, state: str, prompt_template: dict, max_experiences: int = 5) -> List[Tuple[str, str, str]]:
        """使用本地模型从状态生成经验
        
        Args:
            state: 初始状态
            prompt_template: 提示词模板
            max_experiences: 最多生成多少条经验
            
        Returns:
            List of (state, action, next_state) tuples
        """
        experiences = []
        current_state = state
        
        for _ in range(max_experiences):
            # 生成所有可能的动作
            actions = bw_utils.generate_all_actions(current_state)
            
            if not actions:
                break
            
            # 选择一个动作
            action = actions[0]  # 或随机选择
            
            # 使用本地模型预测新状态
            try:
                # 构建提示词
                if "pick" in action:
                    key = "world_update_pickup"
                elif "unstack" in action:
                    key = "world_update_unstack"
                elif "put" in action:
                    key = "world_update_putdown"
                elif "stack" in action:
                    key = "world_update_stack"
                else:
                    continue
                
                prompt = prompt_template[key].format(
                    current_state,
                    action.capitalize() + "."
                )
                
                # 使用本地模型生成
                output = self.local_model.generate(
                    [prompt],
                    eos_token_id="\n",
                    hide_input=True,
                    temperature=0
                ).text[0].strip()
                
                # 应用变化
                next_state = bw_utils.apply_change(output, current_state)
                
                experiences.append((current_state, action, next_state))
                current_state = next_state
                
            except Exception as e:
                print(f"本地模型生成失败: {e}")
                break
        
        return experiences


def generate_experiences(
    memory_store: MemoryStore,
    method: str = "rule",
    data_file: Optional[str] = None,
    local_model=None,
    prompt_template: Optional[dict] = None,
    num_experiences: int = 50,
    validate: bool = True
):
    """生成经验并存入内存
    
    Args:
        memory_store: 内存存储系统
        method: 生成方法 ("rule", "dataset", "local_model")
        data_file: 数据集文件路径（用于 "dataset" 方法）
        local_model: 本地模型（用于 "local_model" 方法）
        prompt_template: 提示词模板（用于 "local_model" 方法）
        num_experiences: 要生成的经验数量
        validate: 是否验证生成的经验（默认 True）
    """
    print(f"\n开始使用 '{method}' 方法生成经验...")
    
    if method == "rule":
        generator = RuleBasedExperienceGenerator()
        
        # 生成更多样化的初始状态
        base_states = [
            "hand is empty, the red block is on the table, the blue block is on top of the red block, the red block is clear",
            "hand is empty, the red block is on the table, the blue block is on the table, the red block is clear, the blue block is clear",
            "is holding the red block, the blue block is on the table, the blue block is clear",
            "hand is empty, the red block is on the table, the blue block is on the table, the green block is on top of the red block, the red block is clear, the green block is clear",
            "is holding the blue block, the red block is on the table, the green block is on top of the red block, the red block is clear",
        ]
        
        # 生成更多状态变体
        colors = ['red', 'blue', 'green', 'yellow', 'orange']
        additional_states = []
        
        # 生成一些随机状态
        for _ in range(min(20, num_experiences // 10)):
            num_blocks = random.randint(2, 4)
            selected_colors = random.sample(colors, num_blocks)
            
            state_parts = []
            if random.random() < 0.5:
                state_parts.append("hand is empty")
            else:
                held_color = random.choice(selected_colors)
                state_parts.append(f"is holding the {held_color} block")
            
            # 随机放置积木
            for i, color in enumerate(selected_colors):
                if f"is holding the {color} block" not in " ".join(state_parts):
                    if random.random() < 0.5:
                        state_parts.append(f"the {color} block is on the table")
                        state_parts.append(f"the {color} block is clear")
                    elif i > 0:
                        below_color = random.choice(selected_colors[:i])
                        state_parts.append(f"the {color} block is on top of the {below_color} block")
                        state_parts.append(f"the {color} block is clear")
            
            state = ", ".join(state_parts)
            additional_states.append(state)
        
        all_states = base_states + additional_states
        
        experiences = []
        # 计算每个状态生成多少经验
        experiences_per_state = max(5, num_experiences // len(all_states))
        
        for state in all_states:
            exp = generator.generate_from_state(state, max_depth=5, max_experiences=experiences_per_state)
            experiences.extend(exp)
        
        # 如果提供了数据文件，也从数据集生成
        if data_file and os.path.exists(data_file):
            exp = generator.generate_from_dataset(data_file, num_examples=min(10, num_experiences // 20), experiences_per_example=5)
            experiences.extend(exp)
        
    elif method == "dataset":
        if not data_file:
            raise ValueError("dataset 方法需要提供 data_file")
        
        generator = RuleBasedExperienceGenerator()
        # 计算每个示例生成多少经验
        experiences_per_example = max(5, num_experiences // 10)  # 至少每个示例 5 条
        num_examples = min(100, num_experiences // experiences_per_example + 1)  # 最多使用 100 个示例
        experiences = generator.generate_from_dataset(data_file, num_examples=num_examples, experiences_per_example=experiences_per_example)
        
        # 若数据集未产出经验（如缺少 config/domain 文件），回退到规则生成，避免“不动”
        if len(experiences) == 0:
            print("  警告：数据集未产出经验（可能缺少 bw_config.yaml / generated_domain.pddl），改用规则生成本批")
            base_states = [
                "hand is empty, the red block is on the table, the blue block is on top of the red block, the red block is clear",
                "hand is empty, the red block is on the table, the blue block is on the table, the red block is clear, the blue block is clear",
                "is holding the red block, the blue block is on the table, the blue block is clear",
                "hand is empty, the red block is on the table, the blue block is on the table, the green block is on top of the red block, the red block is clear, the green block is clear",
            ]
            for state in base_states:
                exp = generator.generate_from_state(state, max_depth=5, max_experiences=max(5, num_experiences // 5))
                experiences.extend(exp)
                if len(experiences) >= num_experiences:
                    break
            experiences = experiences[:num_experiences]
        
    elif method == "local_model":
        if not local_model:
            raise ValueError("local_model 方法需要提供 local_model")
        if not prompt_template:
            raise ValueError("local_model 方法需要提供 prompt_template")
        
        generator = LocalModelExperienceGenerator(local_model)
        
        # 从数据文件加载初始状态
        if data_file and os.path.exists(data_file):
            with open(data_file, 'r') as f:
                examples = json.load(f)
            
            experiences = []
            for example in examples[:num_experiences // 5]:
                init_state = bw_utils.extract_init_state(example)
                exp = generator.generate_from_state(init_state, prompt_template, max_experiences=5)
                experiences.extend(exp)
        else:
            # 使用默认状态
            states = [
                "hand is empty, the red block is on the table, the blue block is on top of the red block, the red block is clear",
            ]
            experiences = []
            for state in states:
                exp = generator.generate_from_state(state, prompt_template, max_experiences=num_experiences)
                experiences.extend(exp)
    else:
        raise ValueError(f"未知的方法: {method}")
    
    # 存入内存（可选验证）
    count_before = len(memory_store.memories)
    print(f"\n生成 {len(experiences)} 条经验，正在存入内存...")
    
    if validate:
        from reasoners.world_model.experience_validator import ExperienceValidator
        validator = ExperienceValidator()
        valid_count = 0
        invalid_count = 0
    
    n_processed = 0
    for state, action, next_state in experiences:
        # 验证经验（如果启用）
        if validate:
            # 检查动作可执行性
            is_applicable, _ = validator.validate_action_applicability(state, action)
            if not is_applicable:
                invalid_count += 1
                continue
            
            # 检查逻辑一致性
            is_valid, _ = validator.validate_logical_consistency(state, action, next_state)
            if not is_valid:
                invalid_count += 1
                continue
            
            valid_count += 1
        
        # 构建提示词（如果有模板）
        prompt = ""
        if prompt_template:
            if "pick" in action:
                key = "world_update_pickup"
            elif "unstack" in action:
                key = "world_update_unstack"
            elif "put" in action:
                key = "world_update_putdown"
            elif "stack" in action:
                key = "world_update_stack"
            else:
                key = None
            
            if key:
                prompt = prompt_template[key].format(state, action.capitalize() + ".")
        
        memory_store.add(state, action, next_state, prompt)
        n_processed += 1
        if n_processed % 500 == 0 and n_processed < len(experiences):
            print(f"  已处理 {n_processed}/{len(experiences)} 条...")
    
    count_after = len(memory_store.memories)
    new_count = count_after - count_before
    integrated_count = n_processed - new_count
    memory_store.save()
    print(f"[OK] 本批通过验证并写入 {n_processed} 条；新增 {new_count} 条（整合更新 {integrated_count} 条），当前共 {count_after} 条经验")
    if validate:
        print(f"  有效经验: {valid_count}, 无效经验: {invalid_count}")
        if invalid_count > 0 and len(experiences) > 0:
            print(f"  准确率: {valid_count / len(experiences):.2%}")
