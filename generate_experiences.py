"""生成经验脚本：不使用 ChatGPT 生成 WorldModel 经验

使用方法：
1. 基于规则生成（推荐）：
   python generate_experiences.py --method rule

2. 从数据集提取：
   python generate_experiences.py --method dataset --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json

3. 使用本地模型生成：
   python generate_experiences.py --method local_model --model_path path/to/model.gguf --data_file examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json
"""

import os
import sys
import argparse
import json

# 添加项目根目录到路径
_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences


def main():
    parser = argparse.ArgumentParser(description="生成 WorldModel 经验（不使用 ChatGPT）")
    parser.add_argument(
        "--method",
        type=str,
        default="rule",
        choices=["rule", "dataset", "local_model"],
        help="生成方法：rule（基于规则），dataset（从数据集），local_model（使用本地模型）"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="数据集文件路径（用于 dataset 或 local_model 方法）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="本地模型路径（用于 local_model 方法）"
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="world_model_memory.json",
        help="内存文件路径"
    )
    parser.add_argument(
        "--num_experiences",
        type=int,
        default=50,
        help="要生成的经验数量"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="禁用自动验证（默认启用验证）"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("WorldModel 经验生成器")
    print("="*60)
    print(f"方法: {args.method}")
    print(f"内存文件: {args.memory_file}")
    print(f"目标经验数: {args.num_experiences}")
    
    # 创建或加载内存存储
    memory_store = MemoryStore(
        memory_file=args.memory_file,
        similarity_threshold=0.7,
        max_memory_size=10000
    )
    
    print(f"\n当前内存中有 {len(memory_store.memories)} 条经验")
    
    # 加载提示词模板（如果需要）
    prompt_template = None
    if args.method == "local_model":
        prompt_path = os.path.join(_root, "examples/CoT/blocksworld/prompts/pool_prompt_v1.json")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                prompt_template = json.load(f)
        else:
            print(f"警告：找不到提示词文件 {prompt_path}")
    
    # 加载本地模型（如果需要）
    local_model = None
    if args.method == "local_model":
        if not args.model_path:
            print("错误：local_model 方法需要提供 --model_path")
            return
        
        if not os.path.exists(args.model_path):
            print(f"错误：找不到模型文件 {args.model_path}")
            return
        
        try:
            from reasoners.lm import LlamaCppModel
            print(f"\n加载本地模型: {args.model_path}...")
            local_model = LlamaCppModel(path=args.model_path)
            print("✓ 本地模型已加载")
        except Exception as e:
            print(f"错误：无法加载本地模型: {e}")
            return
    
    # 生成经验
    try:
        generate_experiences(
            memory_store=memory_store,
            method=args.method,
            data_file=args.data_file,
            local_model=local_model,
            prompt_template=prompt_template,
            num_experiences=args.num_experiences,
            validate=not args.no_validate
        )
        
        print("\n" + "="*60)
        print("[OK] 经验生成完成！")
        print("="*60)
        print(f"\n内存文件: {args.memory_file}")
        print(f"总经验数: {len(memory_store.memories)}")
        
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
