"""快速检查进度（单次）"""

import os
import sys

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore

memory_file = os.path.join(_root, "world_model_memory.json")
target = 10000

if os.path.exists(memory_file):
    memory_store = MemoryStore(memory_file=memory_file)
    current = len(memory_store.memories)
    progress = current / target * 100
    remaining = target - current
    
    print(f"当前: {current:,} / {target:,} ({progress:.1f}%)")
    print(f"剩余: {remaining:,} 条")
else:
    print("内存文件不存在")
