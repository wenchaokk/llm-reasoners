"""检查经验生成进度"""

import os
import sys

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_validator import ExperienceValidator

memory_file = os.path.join(_root, "world_model_memory.json")

if os.path.exists(memory_file):
    memory_store = MemoryStore(memory_file=memory_file)
    count = len(memory_store.memories)
    
    print("="*60)
    print("经验生成进度")
    print("="*60)
    print(f"当前经验数: {count}")
    print(f"目标: 10000 条")
    print(f"进度: {count/10000*100:.1f}%")
    
    if count > 0:
        print(f"\n最新 3 条经验:")
        for i, mem in enumerate(memory_store.memories[-3:], 1):
            print(f"\n{i}. 状态: {mem.state[:60]}...")
            print(f"   动作: {mem.action}")
            print(f"   结果: {mem.next_state[:60]}...")
        
        # 快速验证
        print("\n快速验证（前 10 条）...")
        validator = ExperienceValidator()
        valid = 0
        for mem in memory_store.memories[:10]:
            results = validator.validate_all(mem.state, mem.action, mem.next_state)
            if all(v for v, _ in results.values()):
                valid += 1
        print(f"前 10 条经验中，{valid}/10 条有效 ({valid/10*100:.0f}%)")
else:
    print("内存文件不存在，经验生成可能还未开始")
