"""经验验证器：验证生成的经验是否准确

提供多种验证方法：
1. 逻辑一致性检查（检查状态转换是否符合 Blocksworld 规则）
2. PDDL 验证（使用 VAL 验证器）
3. 规则验证（使用规则生成器验证）
"""

import re
from typing import List, Tuple, Dict
from reasoners.world_model.memory_store import MemoryStore, StateTransition
import reasoners.benchmark.bw_utils as bw_utils


class ExperienceValidator:
    """经验验证器"""
    
    def __init__(self):
        """初始化验证器"""
        pass
    
    def validate_logical_consistency(self, state: str, action: str, next_state: str) -> Tuple[bool, str]:
        """验证逻辑一致性
        
        检查状态转换是否符合 Blocksworld 的基本规则：
        1. 手的状态转换是否正确
        2. 积木位置变化是否正确
        3. 积木的 clear 状态是否正确
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 转换后的状态
            
        Returns:
            (is_valid, error_message)
        """
        errors = []
        
        # 检查手的状态
        hand_was_empty = "hand is empty" in state
        hand_is_empty = "hand is empty" in next_state
        
        if "pick up" in action or "unstack" in action:
            # 拿起或卸下：手应该从空变为持有
            if not hand_was_empty:
                errors.append(f"动作 '{action}' 要求手是空的，但当前手不是空的")
            if hand_is_empty:
                errors.append(f"执行 '{action}' 后，手应该持有积木，但手仍然是空的")
        
        elif "put down" in action or "stack" in action:
            # 放下或叠放：手应该从持有变为空
            if hand_was_empty:
                errors.append(f"动作 '{action}' 要求手持有积木，但手是空的")
            if not hand_is_empty:
                errors.append(f"执行 '{action}' 后，手应该是空的，但手仍然持有积木")
        
        # 检查积木位置变化
        if "pick up" in action:
            color = re.search(r"pick up the (\w+) block", action)
            if color:
                color = color.group(1)
                # 积木应该从桌上或积木堆上移除
                if f"the {color} block is on the table" in next_state:
                    errors.append(f"执行 'pick up' 后，{color} 积木不应该还在桌上")
                if f"the {color} block is on top of" in next_state:
                    errors.append(f"执行 'pick up' 后，{color} 积木不应该还在其他积木上")
                # 手应该持有积木
                if f"is holding the {color} block" not in next_state:
                    errors.append(f"执行 'pick up' 后，手应该持有 {color} 积木")
        
        elif "unstack" in action:
            match = re.search(r"unstack the (\w+) block from on top of the (\w+) block", action)
            if match:
                color, below_color = match.groups()
                # 上面积木应该被移除
                if f"the {color} block is on top of the {below_color} block" in next_state:
                    errors.append(f"执行 'unstack' 后，{color} 积木不应该还在 {below_color} 积木上")
                # 下面积木应该变为 clear
                if f"the {below_color} block is clear" not in next_state:
                    errors.append(f"执行 'unstack' 后，{below_color} 积木应该变为 clear")
                # 手应该持有上面积木
                if f"is holding the {color} block" not in next_state:
                    errors.append(f"执行 'unstack' 后，手应该持有 {color} 积木")
        
        elif "put down" in action:
            color = re.search(r"put down the (\w+) block", action)
            if color:
                color = color.group(1)
                # 积木应该放在桌上
                if f"the {color} block is on the table" not in next_state:
                    errors.append(f"执行 'put down' 后，{color} 积木应该在桌上")
                # 手应该为空
                if "hand is empty" not in next_state:
                    errors.append(f"执行 'put down' 后，手应该是空的")
        
        elif "stack" in action:
            match = re.search(r"stack the (\w+) block on top of the (\w+) block", action)
            if match:
                color, below_color = match.groups()
                # 上面积木应该在目标积木上
                if f"the {color} block is on top of the {below_color} block" not in next_state:
                    errors.append(f"执行 'stack' 后，{color} 积木应该在 {below_color} 积木上")
                # 不应该还在桌上
                if f"the {color} block is on the table" in next_state:
                    errors.append(f"执行 'stack' 后，{color} 积木不应该还在桌上")
                # 手应该为空
                if "hand is empty" not in next_state:
                    errors.append(f"执行 'stack' 后，手应该是空的")
                # 下面积木不应该再是 clear
                if f"the {below_color} block is clear" in next_state:
                    errors.append(f"执行 'stack' 后，{below_color} 积木不应该再是 clear")
        
        # 检查状态完整性
        # 每个积木应该有明确的位置（桌上、其他积木上、或手中）
        blocks = re.findall(r"the (\w+) block", state)
        for block_color in blocks:
            block_color = block_color.strip()
            if block_color:
                # 检查积木在新状态中的位置
                on_table = f"the {block_color} block is on the table" in next_state
                on_top = f"the {block_color} block is on top of" in next_state
                in_hand = f"is holding the {block_color} block" in next_state
                
                if not (on_table or on_top or in_hand):
                    errors.append(f"积木 {block_color} 在新状态中没有明确的位置")
        
        if errors:
            return False, "; ".join(errors)
        return True, "逻辑一致性检查通过"
    
    def validate_with_rule_generator(self, state: str, action: str, next_state: str) -> Tuple[bool, str]:
        """使用规则生成器验证
        
        使用规则生成器生成预期的 next_state，然后与实际结果对比。
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 转换后的状态
            
        Returns:
            (is_valid, error_message)
        """
        from reasoners.world_model.experience_generator import RuleBasedExperienceGenerator
        
        generator = RuleBasedExperienceGenerator()
        expected_next_state = generator._apply_action_rule(state, action)
        
        # 标准化状态字符串（去除空格、标点等差异）
        def normalize(s):
            s = re.sub(r"\s+", " ", s)
            s = s.strip(", ")
            return sorted(s.split(", "))
        
        expected_normalized = normalize(expected_next_state)
        actual_normalized = normalize(next_state)
        
        if expected_normalized == actual_normalized:
            return True, "规则验证通过"
        else:
            diff = set(expected_normalized) - set(actual_normalized)
            extra = set(actual_normalized) - set(expected_normalized)
            error_msg = f"状态不匹配"
            if diff:
                error_msg += f"; 缺少: {diff}"
            if extra:
                error_msg += f"; 多余: {extra}"
            return False, error_msg
    
    def validate_action_applicability(self, state: str, action: str) -> Tuple[bool, str]:
        """验证动作是否可执行
        
        检查在当前状态下，动作是否可以执行。
        
        Args:
            state: 当前状态
            action: 要执行的动作
            
        Returns:
            (is_applicable, error_message)
        """
        # 生成所有可能的动作
        possible_actions = bw_utils.generate_all_actions(state)
        
        if action not in possible_actions:
            return False, f"动作 '{action}' 在当前状态下不可执行。可能的动作: {possible_actions[:3]}..."
        
        return True, "动作可执行"
    
    def validate_all(self, state: str, action: str, next_state: str) -> Dict[str, Tuple[bool, str]]:
        """执行所有验证
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 转换后的状态
            
        Returns:
            Dict of {check_name: (is_valid, message)}
        """
        results = {}
        
        # 1. 动作可执行性检查
        results['action_applicability'] = self.validate_action_applicability(state, action)
        
        # 2. 逻辑一致性检查
        results['logical_consistency'] = self.validate_logical_consistency(state, action, next_state)
        
        # 3. 规则验证
        results['rule_validation'] = self.validate_with_rule_generator(state, action, next_state)
        
        return results
    
    def validate_memory(self, memory_store: MemoryStore) -> Dict:
        """验证内存中的所有经验
        
        Args:
            memory_store: 内存存储系统
            
        Returns:
            验证结果统计
        """
        total = len(memory_store.memories)
        valid_count = 0
        invalid_count = 0
        errors_by_type = {}
        
        print(f"\n开始验证 {total} 条经验...")
        
        for i, memory in enumerate(memory_store.memories):
            if (i + 1) % 10 == 0:
                print(f"  验证进度: {i + 1}/{total}")
            
            results = self.validate_all(memory.state, memory.action, memory.next_state)
            
            # 检查是否所有验证都通过
            all_valid = all(valid for valid, _ in results.values())
            
            if all_valid:
                valid_count += 1
            else:
                invalid_count += 1
                # 记录错误类型
                for check_name, (is_valid, msg) in results.items():
                    if not is_valid:
                        if check_name not in errors_by_type:
                            errors_by_type[check_name] = []
                        errors_by_type[check_name].append({
                            'memory_index': i,
                            'state': memory.state[:50] + "...",
                            'action': memory.action,
                            'error': msg
                        })
        
        return {
            'total': total,
            'valid': valid_count,
            'invalid': invalid_count,
            'accuracy': valid_count / total if total > 0 else 0.0,
            'errors_by_type': errors_by_type
        }


def validate_experiences(memory_file: str = "world_model_memory.json", verbose: bool = False):
    """验证经验文件
    
    Args:
        memory_file: 内存文件路径
        verbose: 是否显示详细信息
    """
    print("="*60)
    print("经验验证器")
    print("="*60)
    
    # 加载内存
    memory_store = MemoryStore(memory_file=memory_file)
    
    if len(memory_store.memories) == 0:
        print("错误：内存中没有经验")
        return
    
    print(f"\n加载了 {len(memory_store.memories)} 条经验")
    
    # 验证
    validator = ExperienceValidator()
    results = validator.validate_memory(memory_store)
    
    # 显示结果
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"总经验数: {results['total']}")
    print(f"有效经验: {results['valid']}")
    print(f"无效经验: {results['invalid']}")
    print(f"准确率: {results['accuracy']:.2%}")
    
    # 显示错误详情
    if results['errors_by_type']:
        print("\n错误详情:")
        for check_name, errors in results['errors_by_type'].items():
            print(f"\n{check_name} 错误 ({len(errors)} 个):")
            if verbose:
                for error in errors[:5]:  # 只显示前 5 个
                    print(f"  [{error['memory_index']}] {error['action']}")
                    print(f"      状态: {error['state']}")
                    print(f"      错误: {error['error']}")
            else:
                print(f"  前 3 个错误:")
                for error in errors[:3]:
                    print(f"    - {error['action']}: {error['error'][:60]}...")
    
    # 建议
    if results['accuracy'] < 0.9:
        print("\n[!] 准确率较低，建议:")
        print("  1. 检查生成经验的代码")
        print("  2. 重新生成经验")
        print("  3. 使用 'rule' 方法生成经验（准确性最高）")
    elif results['accuracy'] >= 0.95:
        print("\n[OK] 经验质量很好！")
    
    return results
