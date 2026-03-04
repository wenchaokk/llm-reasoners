"""测试生成少量经验（用于调试）"""

import os
import sys

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences

# 创建内存存储
memory_store = MemoryStore(memory_file="test_memory.json", max_memory_size=1000)

print("测试生成 10 条经验...")
generate_experiences(
    memory_store=memory_store,
    method="rule",
    num_experiences=10,
    validate=True
)

print(f"\n成功生成 {len(memory_store.memories)} 条经验")
