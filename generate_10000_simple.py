"""简化版：从数据集提取 10000 条经验

只使用数据集提取方法，避免规则生成的复杂性。
"""

import os
import sys
import glob

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences
from reasoners.world_model.experience_validator import validate_experiences


def main():
    print("="*60)
    print("从数据集提取 10000 条经验")
    print("="*60)
    
    memory_file = os.path.join(_root, "world_model_memory.json")
    memory_store = MemoryStore(memory_file=memory_file, max_memory_size=20000, save_every=200)
    
    initial_count = len(memory_store.memories)
    print(f"\n初始内存中有 {initial_count} 条经验")
    
    if initial_count > 0:
        # 自动继续累积（非交互模式）
        print(f"内存中已有 {initial_count} 条经验，将继续累积...")
    
    # 查找所有数据集文件
    data_dir = os.path.join(_root, "examples/CoT/blocksworld/data/split_v1")
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    if not data_files:
        print(f"错误：找不到数据集文件在 {data_dir}")
        return
    
    print(f"\n找到 {len(data_files)} 个数据集文件")
    
    target_total = 10000
    batch_size = 200  # 每批处理
    consecutive_empty = 0  # 连续 0 增长的批次数
    
    total_generated = 0
    
    # 遍历所有数据集文件
    for data_file in data_files:
        if len(memory_store.memories) >= target_total:
            break
        
        print(f"\n处理文件: {os.path.basename(data_file)}")
        print(f"当前进度: {len(memory_store.memories)} / {target_total}")
        
        remaining = target_total - len(memory_store.memories)
        batch_target = min(batch_size, remaining)
        
        try:
            initial_count = len(memory_store.memories)
            generate_experiences(
                memory_store=memory_store,
                method="dataset",
                data_file=data_file,
                num_experiences=batch_target,
                validate=True
            )
            final_count = len(memory_store.memories)
            batch_valid = final_count - initial_count
            total_generated += batch_valid
            
            if batch_valid > 0:
                consecutive_empty = 0
                print(f"  本文件新增 {batch_valid} 条有效经验")
            else:
                consecutive_empty += 1
                print(f"  本文件未新增经验（可能缺少 config/domain，或均为整合更新）")
                # 若连续 2 批都是 0，用规则生成一批，避免一直“不动”
                if consecutive_empty >= 2:
                    print("  改用规则生成本批，以继续推进...")
                    generate_experiences(
                        memory_store=memory_store,
                        method="rule",
                        num_experiences=batch_target,
                        validate=True
                    )
                    consecutive_empty = 0
            
        except Exception as e:
            print(f"  警告：处理文件失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 若未满 10000，用规则方法循环生成直至达标
    while len(memory_store.memories) < target_total:
        remaining = target_total - len(memory_store.memories)
        batch_target = min(1000, remaining)
        print(f"\n规则生成补足: 当前 {len(memory_store.memories)} / {target_total}，本批目标 {batch_target}")
        try:
            n_before = len(memory_store.memories)
            generate_experiences(
                memory_store=memory_store,
                method="rule",
                num_experiences=batch_target,
                validate=True
            )
            n_after = len(memory_store.memories)
            print(f"  本批新增 {n_after - n_before} 条，当前共 {n_after} 条")
        except Exception as e:
            print(f"  警告：规则生成失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 最终验证
    print("\n" + "="*60)
    print("最终验证")
    print("="*60)
    
    validate_experiences(memory_file=memory_file, verbose=False)
    
    print("\n" + "="*60)
    print("[OK] 完成！")
    print("="*60)
    print(f"\n内存文件: {memory_file}")
    print(f"总经验数: {len(memory_store.memories)}")
    print("\n可以使用以下命令验证经验：")
    print(f"  python validate_experiences.py --memory_file {memory_file}")


if __name__ == "__main__":
    main()
