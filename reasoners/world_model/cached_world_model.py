"""带内存缓存的 WorldModel

这个模块实现了一个混合 WorldModel，它：
1. 先查询内存中的历史经验
2. 如果有相似经验，使用本地小模型 + 历史经验进行预测
3. 如果没有相似经验，使用 ChatGPT 进行预测
4. 写入前对预测结果做评估（动作可执行性、逻辑一致性），通过后才写入
5. 新经验与旧经验整合：同 (state, action) 时更新已有记录，否则追加
"""

from typing import Optional, Tuple
from reasoners import WorldModel, LanguageModel
from reasoners.world_model.memory_store import MemoryStore
from reasoners.world_model.experience_validator import ExperienceValidator
import copy


class CachedWorldModel(WorldModel):
    """带内存缓存的 WorldModel 包装器
    
    包装原有的 WorldModel，添加内存缓存功能：
    - 查询历史经验
    - 使用本地小模型进行预测（如果有相似经验）
    - 使用 ChatGPT 进行预测（如果没有相似经验）
    - 自动更新内存：预测通过评估后写入，并与旧经验整合（同 (state, action) 则更新）
    """
    
    def __init__(self,
                 base_world_model: WorldModel,
                 chatgpt_model: Optional[LanguageModel] = None,
                 local_model: Optional[LanguageModel] = None,
                 memory_store: Optional[MemoryStore] = None,
                 use_cache: bool = True,
                 cache_threshold: float = 0.7,
                 local_only: bool = False):
        """初始化带缓存的 WorldModel
        
        Args:
            base_world_model: 原有的 WorldModel（BlocksWorldModel）
            chatgpt_model: ChatGPT 模型（用于新预测，可选）
            local_model: 本地小模型（用于基于历史经验的预测，必需如果 local_only=True）
            memory_store: 内存存储系统（可选，会自动创建）
            use_cache: 是否使用缓存
            cache_threshold: 缓存相似度阈值
            local_only: 如果为 True，只使用本地模型，不使用 ChatGPT
        """
        super().__init__()
        self.base_world_model = base_world_model
        self.chatgpt_model = chatgpt_model
        self.local_model = local_model
        self.use_cache = use_cache
        self.cache_threshold = cache_threshold
        self.local_only = local_only
        
        # 如果 local_only=True，必须提供 local_model
        if local_only and local_model is None:
            raise ValueError("local_only=True 时必须提供 local_model")
        
        # 初始化内存存储
        if memory_store is None:
            memory_store = MemoryStore(similarity_threshold=cache_threshold)
        self.memory_store = memory_store
        
        # 统计信息
        self.stats = {
            'cache_hits': 0,      # 缓存命中次数
            'cache_misses': 0,    # 缓存未命中次数
            'total_predictions': 0  # 总预测次数
        }
    
    def init_state(self):
        """初始化状态（委托给 base_world_model）"""
        return self.base_world_model.init_state()
    
    def step(self, state, action):
        """执行动作，转换到下一个状态（带缓存）
        
        流程：
        1. 如果 local_only=True，只使用本地模型
        2. 查询内存中是否有相似经验
        3. 如果有相似经验，使用本地模型 + 历史经验预测
        4. 如果没有相似经验：
           - 如果 local_only=True，使用本地模型（无历史经验）
           - 否则使用 ChatGPT 预测，并将结果存入内存
        """
        self.stats['total_predictions'] += 1
        
        # 如果只使用本地模型：先查内存再预测，预测结果也会写入内存
        if self.local_only:
            return self._predict_with_local_model_only(state, action)
        
        # 如果禁用缓存或没有本地模型，使用 ChatGPT
        if not self.use_cache or self.local_model is None:
            if self.chatgpt_model is None:
                raise ValueError("未提供 chatgpt_model 或 local_model")
            return self._predict_with_chatgpt(state, action)
        
        # 获取状态和动作的字符串表示
        state_str = self._state_to_string(state)
        action_str = self._action_to_string(action)
        
        # 查询相似的历史经验
        similar_memories = self.memory_store.query(state_str, action_str, top_k=3)
        
        if similar_memories and similar_memories[0][1] >= self.cache_threshold:
            # 缓存命中：使用本地模型 + 历史经验
            self.stats['cache_hits'] += 1
            return self._predict_with_local_model(state, action, similar_memories)
        else:
            # 缓存未命中：使用 ChatGPT
            self.stats['cache_misses'] += 1
            if self.chatgpt_model is None:
                raise ValueError("缓存未命中但未提供 chatgpt_model")
            return self._predict_with_chatgpt(state, action)
    
    def _state_to_string(self, state) -> str:
        """将状态转换为字符串（用于查询）"""
        # 对于 BlocksWorldModel，状态是 BWState，包含 blocks_state
        if hasattr(state, 'blocks_state'):
            return state.blocks_state
        return str(state)
    
    def _action_to_string(self, action) -> str:
        """将动作转换为字符串（用于查询）"""
        if isinstance(action, str):
            return action
        return str(action)
    
    def _add_experience_if_valid(self, state_str: str, action_str: str, next_state_str: str, prompt: str) -> bool:
        """经预测评估通过后写入内存，并与旧经验整合。返回是否已写入。"""
        is_applicable, _ = self._experience_validator.validate_action_applicability(state_str, action_str)
        if not is_applicable:
            return False
        is_consistent, _ = self._experience_validator.validate_logical_consistency(
            state_str, action_str, next_state_str
        )
        if not is_consistent:
            return False
        self.memory_store.add(state_str, action_str, next_state_str, prompt)
        return True
    
    def _predict_with_local_model_only(self, state, action):
        """仅用本地模型预测（local_only 模式）：先查内存再预测，预测结果会写入内存"""
        state_str = self._state_to_string(state)
        action_str = self._action_to_string(action)
        similar_memories = self.memory_store.query(state_str, action_str, top_k=3)
        return self._predict_with_local_model(state, action, similar_memories)
    
    def _predict_with_local_model(self, state, action, similar_memories):
        """使用本地模型 + 历史经验进行预测
        
        Args:
            state: 当前状态
            action: 要执行的动作
            similar_memories: 相似的历史经验列表（可以为空）
            
        Returns:
            (新状态, 辅助信息)
        """
        # 构建包含历史经验的提示词
        if similar_memories:
            # 使用最相似的经验作为示例
            best_memory, similarity = similar_memories[0]
        else:
            best_memory = None
            similarity = 0.0
        
        # 构建提示词：包含历史经验和当前情况（如果有历史经验）
        if best_memory:
            prompt = self._build_prompt_with_memory(state, action, best_memory)
        else:
            prompt = self._build_prompt_without_memory(state, action)
        
        # 使用本地模型生成
        try:
            # 获取 base_world_model 的 prompt 模板
            if hasattr(self.base_world_model, 'prompt'):
                # 使用原有的 prompt 格式
                world_update_prompt = self._get_prompt_template(state, action)
                
                # 在提示词前添加历史经验（如果有）
                if best_memory:
                    enhanced_prompt = f"""以下是相似的历史经验（相似度: {similarity:.2f}）：
状态: {best_memory.state}
动作: {best_memory.action}
结果: {best_memory.next_state}

现在请预测以下情况：
{world_update_prompt}"""
                else:
                    # 没有历史经验，直接使用原始提示词
                    enhanced_prompt = world_update_prompt
            else:
                enhanced_prompt = prompt
            
            # 使用本地模型生成
            world_output = self.local_model.generate(
                [enhanced_prompt],
                eos_token_id="\n",
                hide_input=True,
                temperature=0
            ).text[0].strip()
            
            # 应用输出到状态（使用 base_world_model 的逻辑）
            if hasattr(self.base_world_model, 'update_blocks'):
                # BlocksWorldModel 的特殊处理
                import reasoners.benchmark.bw_utils as utils
                new_state_str = utils.apply_change(world_output, self._state_to_string(state))
                # 创建新状态
                new_state = self._create_new_state(state, new_state_str, action)
            else:
                # 通用处理：直接调用 base_world_model
                return self.base_world_model.step(state, action)
            
            # 检查目标达成度
            if hasattr(self.base_world_model, 'example'):
                import reasoners.benchmark.bw_utils as utils
                goal_reached = utils.goal_check(
                    utils.extract_goals(self.base_world_model.example),
                    new_state_str
                )
            else:
                goal_reached = (False, 0.0)
            
            # 通过预测评估后才写入内存，并与旧经验整合
            if self.use_cache:
                state_str = self._state_to_string(state)
                action_str = self._action_to_string(action)
                prompt = self._get_prompt_template(state, action) if hasattr(self.base_world_model, 'prompt') else ""
                self._add_experience_if_valid(state_str, action_str, new_state_str, prompt)
            
            return new_state, {"goal_reached": goal_reached}
            
        except Exception as e:
            # 如果本地模型失败，回退到 ChatGPT
            print(f"[CachedWorldModel] 本地模型预测失败，回退到 ChatGPT: {e}")
            return self._predict_with_chatgpt(state, action)
    
    def _predict_with_chatgpt(self, state, action):
        """使用 ChatGPT 进行预测，并将结果存入内存
        
        Args:
            state: 当前状态
            action: 要执行的动作
            
        Returns:
            (新状态, 辅助信息)
        """
        if self.chatgpt_model is None:
            raise ValueError("未提供 chatgpt_model")
        
        # 临时替换 base_world_model 的模型为 chatgpt_model
        original_model = self.base_world_model.base_model
        self.base_world_model.base_model = self.chatgpt_model
        
        try:
            # 使用 base_world_model 进行预测
            new_state, aux = self.base_world_model.step(state, action)
            
            # 通过预测评估后才写入内存，并与旧经验整合
            if self.use_cache:
                state_str = self._state_to_string(state)
                action_str = self._action_to_string(action)
                next_state_str = self._state_to_string(new_state)
                prompt = self._get_prompt_template(state, action) if hasattr(self.base_world_model, 'prompt') else ""
                self._add_experience_if_valid(state_str, action_str, next_state_str, prompt)
            
            return new_state, aux
        finally:
            # 恢复原始模型
            self.base_world_model.base_model = original_model
    
    def _get_prompt_template(self, state, action):
        """获取提示词模板（从 base_world_model）"""
        if not hasattr(self.base_world_model, 'prompt'):
            return ""
        
        # 根据动作类型选择模板
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            return ""
        
        state_str = self._state_to_string(state)
        return self.base_world_model.prompt[key].format(
            state_str,
            action.capitalize() + "."
        )
    
    def _build_prompt_with_memory(self, state, action, memory):
        """构建包含历史经验的提示词"""
        if memory is None:
            return self._build_prompt_without_memory(state, action)
        return f"""历史经验：
状态: {memory.state}
动作: {memory.action}
结果: {memory.next_state}

当前情况：
状态: {self._state_to_string(state)}
动作: {action}

请预测结果："""
    
    def _build_prompt_without_memory(self, state, action):
        """构建不包含历史经验的提示词（仅使用本地模型时）"""
        return f"""当前情况：
状态: {self._state_to_string(state)}
动作: {action}

请预测结果："""
    
    def _create_new_state(self, old_state, new_state_str, action):
        """创建新状态对象（BlocksWorldModel 专用）"""
        # 深拷贝原状态
        new_state = copy.deepcopy(old_state)
        
        # 更新 blocks_state
        if hasattr(new_state, 'blocks_state'):
            # 创建新的 BWState
            # 动态导入以避免循环依赖
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
            from examples.CoT.blocksworld.world_model import BWState
            
            # 处理 buffered_action：如果原来有缓冲动作，清空；否则放入当前动作
            if new_state.buffered_action:
                new_buffered_action = ""
            else:
                new_buffered_action = action
            
            return BWState(
                step_idx=new_state.step_idx + 1,
                last_blocks_state=new_state.blocks_state,
                blocks_state=new_state_str,
                buffered_action=new_buffered_action
            )
        
        return new_state
    
    def is_terminal(self, state):
        """判断是否为终止状态（委托给 base_world_model）"""
        return self.base_world_model.is_terminal(state)
    
    def update_example(self, example, **kwargs):
        """更新示例（委托给 base_world_model）"""
        self.base_world_model.update_example(example, **kwargs)
        super().update_example(example, **kwargs)
    
    def get_stats(self):
        """获取统计信息"""
        stats = self.stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / stats['total_predictions'] 
            if stats['total_predictions'] > 0 else 0.0
        )
        stats['memory_stats'] = self.memory_store.get_stats()
        return stats
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("CachedWorldModel 统计信息:")
        print(f"  总预测次数: {stats['total_predictions']}")
        print(f"  缓存命中: {stats['cache_hits']}")
        print(f"  缓存未命中: {stats['cache_misses']}")
        print(f"  缓存命中率: {stats['cache_hit_rate']:.2%}")
        print(f"  内存中经验数: {stats['memory_stats']['total_memories']}")
        print("="*50 + "\n")
