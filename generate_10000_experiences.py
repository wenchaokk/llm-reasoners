"""批量生成 10000 条经验并验证

使用两种方法生成经验：
1. 基于规则生成（目标：5000 条）
2. 从数据集提取（目标：5000 条）

所有经验都会通过验证器验证。
"""

import os
import sys
import json
import glob

# 添加项目根目录到路径
_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences
from reasoners.world_model.experience_validator import ExperienceValidator


def find_data_files():
    """查找所有可用的数据集文件"""
    data_dir = os.path.join(_root, "examples/CoT/blocksworld/data/split_v1")
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, "*.json"))
        return sorted(files)
    return []


def generate_with_rule(memory_store: MemoryStore, target_count: int):
    """使用规则方法生成经验"""
    print("\n" + "="*60)
    print("方法 1: 基于规则生成经验")
    print("="*60)
    
    validator = ExperienceValidator()
    current_count = len(memory_store.memories)
    target_total = current_count + target_count
    
    print(f"当前内存中有 {current_count} 条经验")
    print(f"目标：生成 {target_count} 条有效经验（总计 {target_total} 条）")
    
    batch_size = 200  # 每批生成 200 条
    total_generated = 0
    total_valid = 0
    
    # 生成多个批次，直到达到目标数量
    while len(memory_store.memories) < target_total:
        remaining = target_total - len(memory_store.memories)
        batch_target = min(batch_size, remaining)
        
        print(f"\n生成批次（目标 {batch_target} 条有效经验）...")
        
        # 生成经验（会自动验证）
        initial_count = len(memory_store.memories)
        generate_experiences(
            memory_store=memory_store,
            method="rule",
            num_experiences=batch_target * 2,  # 生成更多，因为会过滤无效的
            validate=True
        )
        
        final_count = len(memory_store.memories)
        batch_valid = final_count - initial_count
        total_generated += batch_target * 2
        total_valid += batch_valid
        
        print(f"本批次：生成了 {batch_valid} 条有效经验")
        print(f"累计：{final_count} / {target_total} 条经验")
        
        # 如果连续几批都没有新经验，停止
        if batch_valid == 0:
            print("警告：本批次没有生成新经验，可能已达到上限")
            break
        
        # 如果已经达到目标，停止
        if final_count >= target_total:
            break
    
    print(f"\n[OK] 规则生成完成：共生成 {total_valid} 条有效经验")
    return total_valid


def generate_from_dataset(memory_store: MemoryStore, target_count: int):
    """从数据集提取经验"""
    print("\n" + "="*60)
    print("方法 2: 从数据集提取经验")
    print("="*60)
    
    # 查找数据集文件
    data_files = find_data_files()
    
    if not data_files:
        print("错误：找不到数据集文件")
        print("请确保存在: examples/CoT/blocksworld/data/split_v1/*.json")
        return 0
    
    print(f"找到 {len(data_files)} 个数据集文件")
    
    current_count = len(memory_store.memories)
    target_total = current_count + target_count
    
    print(f"当前内存中有 {current_count} 条经验")
    print(f"目标：生成 {target_count} 条有效经验（总计 {target_total} 条）")
    
    total_valid = 0
    files_processed = 0
    
    # 遍历所有数据集文件
    for data_file in data_files:
        if len(memory_store.memories) >= target_total:
            break
        
        print(f"\n处理文件: {os.path.basename(data_file)}")
        
        # 计算需要从这个文件生成多少经验
        remaining = target_total - len(memory_store.memories)
        batch_size = min(500, remaining)  # 每个文件最多生成 500 条
        
        initial_count = len(memory_store.memories)
        
        try:
            generate_experiences(
                memory_store=memory_store,
                method="dataset",
                data_file=data_file,
                num_experiences=batch_size,
                validate=True
            )
            
            final_count = len(memory_store.memories)
            batch_valid = final_count - initial_count
            total_valid += batch_valid
            files_processed += 1
            
            print(f"  生成了 {batch_valid} 条有效经验")
            print(f"  累计：{final_count} / {target_total} 条经验")
            
        except Exception as e:
            print(f"  警告：处理文件失败: {e}")
            continue
    
    print(f"\n[OK] 数据集提取完成：处理了 {files_processed} 个文件，共生成 {total_valid} 条有效经验")
    return total_valid


def validate_all_experiences(memory_store: MemoryStore):
    """验证所有经验"""
    print("\n" + "="*60)
    print("最终验证")
    print("="*60)
    
    validator = ExperienceValidator()
    results = validator.validate_memory(memory_store)
    
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"总经验数: {results['total']}")
    print(f"有效经验: {results['valid']}")
    print(f"无效经验: {results['invalid']}")
    print(f"准确率: {results['accuracy']:.2%}")
    
    # 如果有无效经验，显示错误
    if results['errors_by_type']:
        print("\n错误详情:")
        for check_name, errors in results['errors_by_type'].items():
            print(f"  {check_name}: {len(errors)} 个错误")
            if len(errors) <= 5:
                for error in errors:
                    print(f"    - {error['action']}: {error['error'][:60]}...")
    
    # 如果准确率低于 95%，建议清理无效经验
    if results['accuracy'] < 0.95:
        print("\n⚠️  准确率低于 95%，建议清理无效经验")
        response = input("是否清理无效经验？(y/n): ").strip().lower()
        if response == 'y':
            print("清理无效经验...")
            # 重新生成内存，只保留有效的
            valid_memories = []
            for memory in memory_store.memories:
                results = validator.validate_all(memory.state, memory.action, memory.next_state)
                if all(valid for valid, _ in results.values()):
                    valid_memories.append(memory)
            
            memory_store.memories = valid_memories
            memory_store.save()
            print(f"[OK] 清理完成，保留 {len(valid_memories)} 条有效经验")
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("批量生成 10000 条经验并验证")
    print("="*60)
    
    # 配置
    total_target = 10000
    rule_target = 5000  # 规则生成 5000 条
    dataset_target = 5000  # 数据集提取 5000 条
    
    memory_file = os.path.join(_root, "world_model_memory.json")
    
    # 创建或加载内存存储
    memory_store = MemoryStore(
        memory_file=memory_file,
        similarity_threshold=0.7,
        max_memory_size=20000  # 增加最大存储量
    )
    
    initial_count = len(memory_store.memories)
    print(f"\n初始内存中有 {initial_count} 条经验")
    
    if initial_count > 0:
        response = input(f"内存中已有 {initial_count} 条经验，是否继续累积？(y/n): ").strip().lower()
        if response != 'y':
            print("清空内存...")
            memory_store.clear()
            initial_count = 0
    
    # 方法 1：基于规则生成
    rule_count = generate_with_rule(memory_store, rule_target)
    
    # 方法 2：从数据集提取
    dataset_count = generate_from_dataset(memory_store, dataset_target)
    
    # 最终验证
    final_count = len(memory_store.memories)
    print("\n" + "="*60)
    print("生成总结")
    print("="*60)
    print(f"初始经验: {initial_count}")
    print(f"规则生成: {rule_count} 条")
    print(f"数据集提取: {dataset_count} 条")
    print(f"总计: {final_count} 条经验")
    
    # 验证所有经验
    validate_all_experiences(memory_store)
    
    print("\n" + "="*60)
    print("[OK] 完成！")
    print("="*60)
    print(f"\n内存文件: {memory_file}")
    print(f"总经验数: {len(memory_store.memories)}")
    print("\n可以使用以下命令验证经验：")
    print(f"  python validate_experiences.py --memory_file {memory_file}")


if __name__ == "__main__":
    main()
