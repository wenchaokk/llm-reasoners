"""内存存储系统：存储和查询历史状态转换经验

这个模块实现了一个内存系统，用于存储 WorldModel 的历史预测经验。
当进行新的状态转换预测时，可以查询相似的历史经验，然后用本地小模型进行预测。

检索优化：精确匹配索引、按动作类型过滤、可选 RapidFuzz 加速、top-k 堆减少排序。
"""

import json
import os
import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib

# 可选：用 RapidFuzz 加速相似度计算（pip install rapidfuzz）
try:
    from rapidfuzz import fuzz
    _USE_RAPIDFUZZ = True
except ImportError:
    _USE_RAPIDFUZZ = False


@dataclass
class StateTransition:
    """状态转换经验
    
    存储一次状态转换的完整信息：
    - state: 当前状态
    - action: 执行的动作
    - next_state: 转换后的状态
    - prompt: 使用的提示词（用于小模型预测）
    - timestamp: 记录时间
    """
    state: str
    action: str
    next_state: str
    prompt: str
    timestamp: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class MemoryStore:
    """内存存储系统
    
    存储历史的状态转换经验，支持：
    1. 添加或整合经验（同 (state, action) 时更新已有记录，否则追加）
    2. 查询相似的经验（基于状态和动作的文本相似度）
    3. 持久化存储（保存到文件）
    4. 从文件加载历史经验
    """
    
    def __init__(self, 
                 memory_file: Optional[str] = None,
                 similarity_threshold: float = 0.7,
                 max_memory_size: int = 10000,
                 save_every: int = 1):
        """初始化内存存储
        
        Args:
            memory_file: 持久化文件路径（可选）
            similarity_threshold: 相似度阈值（0-1），低于此值不返回
            max_memory_size: 最大存储数量，超过后删除最旧的
            save_every: 每追加多少条才写盘一次（1=每次都写；设大可加速大批量生成，结束时需主动 save()）
        """
        self.memories: List[StateTransition] = []
        self.memory_file = memory_file or "world_model_memory.json"
        self.similarity_threshold = similarity_threshold
        self.max_memory_size = max_memory_size
        self.save_every = save_every
        self._append_count = 0
        # 精确匹配索引：(state, action) -> 在 memories 中的下标
        self._exact_index: Dict[Tuple[str, str], int] = {}
        # 按动作类型索引：action_type -> 该类型经验在 memories 中的下标集合（用于检索时缩小范围）
        self._action_type_index: Dict[str, List[int]] = {}
        
        # 如果文件存在，加载历史经验
        if os.path.exists(self.memory_file):
            self.load()
    
    def _action_type(self, action: str) -> str:
        """从动作字符串解析动作类型，用于检索过滤。"""
        a = action.lower()
        if "pick up" in a or "pick-up" in a:
            return "pick"
        if "put down" in a or "put-down" in a:
            return "put"
        if "stack" in a:
            return "stack"
        if "unstack" in a:
            return "unstack"
        return "other"
    
    def _rebuild_indexes(self):
        """在 load 或 pop 后重建 _exact_index 和 _action_type_index。"""
        self._exact_index.clear()
        self._action_type_index.clear()
        for i, m in enumerate(self.memories):
            self._exact_index[(m.state, m.action)] = i
            t = self._action_type(m.action)
            self._action_type_index.setdefault(t, []).append(i)
    
    def add(self, state: str, action: str, next_state: str, prompt: str):
        """添加或整合状态转换经验。
        
        若已存在相同 (state, action) 的经验，则用新的 next_state/prompt 更新该条（整合）；
        否则追加新经验。
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 转换后的状态
            prompt: 使用的提示词
        """
        memory = StateTransition(
            state=state,
            action=action,
            next_state=next_state,
            prompt=prompt,
            timestamp=datetime.now().isoformat()
        )
        
        key = (state, action)
        if key in self._exact_index:
            i = self._exact_index[key]
            self.memories[i] = memory
            self.save()
            return
        
        self.memories.append(memory)
        self._append_count += 1
        idx = len(self.memories) - 1
        self._exact_index[key] = idx
        t = self._action_type(action)
        self._action_type_index.setdefault(t, []).append(idx)
        
        # 如果超过最大数量，删除最旧的
        if len(self.memories) > self.max_memory_size:
            old_state, old_action = self.memories[0].state, self.memories[0].action
            self.memories.pop(0)
            del self._exact_index[(old_state, old_action)]
            for t_key in self._action_type_index:
                self._action_type_index[t_key] = [i - 1 for i in self._action_type_index[t_key] if i > 0]
        
        # 自动保存（每 save_every 次追加保存一次，减少大批量写入次数）
        if self._append_count % self.save_every == 0:
            self.save()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（RapidFuzz 若可用则更快，否则用 SequenceMatcher）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数（0-1）
        """
        if _USE_RAPIDFUZZ:
            return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def query(self, state: str, action: str, top_k: int = 3) -> List[Tuple[StateTransition, float]]:
        """查询相似的历史经验（优化：精确匹配优先、按动作类型过滤、top-k 堆）
        
        Args:
            state: 当前状态
            action: 要执行的动作
            top_k: 返回最相似的 k 条经验
            
        Returns:
            List of (StateTransition, similarity_score) tuples，按相似度降序排列
        """
        if not self.memories:
            return []
        
        # 1. 精确匹配优先：若存在相同 (state, action)，直接返回该条，相似度 1.0
        key = (state, action)
        if key in self._exact_index:
            idx = self._exact_index[key]
            return [(self.memories[idx], 1.0)]
        
        # 2. 只在与当前动作类型相同的经验中做相似度计算，缩小检索范围
        at = self._action_type(action)
        candidate_indices: List[int] = self._action_type_index.get(at, [])
        if not candidate_indices:
            candidate_indices = list(range(len(self.memories)))
        
        # 3. 使用最小堆维护 top-k，避免全量排序；相似度 = (状态相似度 + 动作相似度) / 2
        heap: List[Tuple[float, StateTransition]] = []
        for i in candidate_indices:
            memory = self.memories[i]
            state_sim = self._calculate_similarity(state, memory.state)
            action_sim = self._calculate_similarity(action, memory.action)
            combined_sim = (state_sim + action_sim) / 2.0
            
            if combined_sim < self.similarity_threshold:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (combined_sim, memory))
            elif combined_sim > heap[0][0]:
                heapq.heapreplace(heap, (combined_sim, memory))
        
        # 按相似度降序返回
        result = [(m, s) for s, m in heap]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def save(self):
        """保存内存到文件"""
        if not self.memory_file:
            return
        
        data = {
            'memories': [m.to_dict() for m in self.memories],
            'metadata': {
                'count': len(self.memories),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.memory_file) if os.path.dirname(self.memory_file) else '.', exist_ok=True)
        
        # 先尝试临时文件再替换；若临时文件无权限则直接写目标文件
        temp_file = self.memory_file + '.tmp'
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                if os.path.exists(self.memory_file):
                    os.replace(temp_file, self.memory_file)
                else:
                    os.rename(temp_file, self.memory_file)
            except (PermissionError, OSError):
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
        except (PermissionError, OSError):
            # 临时文件无法创建/写入时（如目录只读或 .tmp 被占用），直接写目标文件
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            raise e
    
    def load(self):
        """从文件加载内存"""
        if not os.path.exists(self.memory_file):
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.memories = [StateTransition.from_dict(m) for m in data.get('memories', [])]
            self._rebuild_indexes()
            print(f"[MemoryStore] 加载了 {len(self.memories)} 条历史经验")
        except Exception as e:
            print(f"[MemoryStore] 加载失败: {e}")
            self.memories = []
            self._exact_index = {}
            self._action_type_index = {}
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_memories': len(self.memories),
            'memory_file': self.memory_file,
            'similarity_threshold': self.similarity_threshold,
            'max_size': self.max_memory_size
        }
    
    def clear(self):
        """清空所有内存"""
        self.memories = []
        self._exact_index.clear()
        self._action_type_index.clear()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        print("[MemoryStore] 已清空所有内存")
