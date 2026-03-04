"""测试只使用本地小模型的 WorldModel

这个脚本演示如何只使用本地小模型进行预测，完全不使用 ChatGPT。
"""

import os
import sys
import json

# 添加项目根目录到路径
_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.lm import LlamaCppModel
from reasoners.world_model import CachedWorldModel, MemoryStore
from examples.CoT.blocksworld.world_model import BlocksWorldModel


def main():
    """测试只使用本地模型的 WorldModel"""
    
    print("="*60)
    print("测试只使用本地小模型的 WorldModel")
    print("="*60)
    
    # 1. 获取本地模型路径
    local_model_path = input("\n请输入本地模型路径（LlamaCpp 格式，必需）: ").strip()
    
    if not local_model_path or not os.path.exists(local_model_path):
        print(f"错误：找不到模型文件 {local_model_path}")
        print("\n提示：你需要先下载一个 LlamaCpp 格式的模型文件")
        print("例如：llama-2-7b-chat.gguf 或 llama-3-8b-instruct.gguf")
        return
    
    # 2. 加载本地模型
    print(f"\n加载本地模型: {local_model_path}...")
    try:
        local_model = LlamaCppModel(path=local_model_path)
        print("✓ 本地模型已加载")
    except Exception as e:
        print(f"✗ 本地模型加载失败: {e}")
        print("\n提示：请确保已安装 llama-cpp-python:")
        print("  pip install llama-cpp-python")
        return
    
    # 3. 加载示例数据
    data_path = os.path.join(_root, "examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json")
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return
    
    with open(data_path, 'r') as f:
        examples = json.load(f)
    
    if not examples:
        print("错误：无法加载示例数据")
        return
    
    example = examples[0]
    print(f"\n使用示例 ID: {example.get('id', 'unknown')}")
    
    # 4. 加载提示词
    prompt_path = os.path.join(_root, "examples/CoT/blocksworld/prompts/pool_prompt_v1.json")
    if not os.path.exists(prompt_path):
        print(f"错误：找不到提示词文件 {prompt_path}")
        return
    
    with open(prompt_path, 'r') as f:
        prompt = json.load(f)
    
    # 5. 创建基础 WorldModel（使用本地模型）
    print("\n创建基础 WorldModel（使用本地模型）...")
    base_world_model = BlocksWorldModel(
        base_model=local_model,  # 使用本地模型
        prompt=prompt,
        max_steps=6
    )
    base_world_model.update_example(example)
    print("✓ 基础 WorldModel 已创建")
    
    # 6. 创建内存存储
    memory_file = os.path.join(_root, "world_model_memory.json")
    memory_store = MemoryStore(
        memory_file=memory_file,
        similarity_threshold=0.6,  # 降低阈值，更容易匹配
        max_memory_size=1000
    )
    print(f"\n✓ 内存存储已初始化: {memory_file}")
    print(f"  当前内存中有 {len(memory_store.memories)} 条经验")
    
    if len(memory_store.memories) == 0:
        print("  注意：内存为空，将完全依赖本地模型进行预测")
        print("  建议：先用 ChatGPT 建立一些经验，或手动添加经验")
    
    # 7. 创建只使用本地模型的 CachedWorldModel
    print("\n创建只使用本地模型的 CachedWorldModel...")
    cached_world_model = CachedWorldModel(
        base_world_model=base_world_model,
        chatgpt_model=None,  # 不使用 ChatGPT
        local_model=local_model,
        memory_store=memory_store,
        use_cache=True,
        cache_threshold=0.6,
        local_only=True  # 关键：只使用本地模型
    )
    print("✓ 只使用本地模型的 WorldModel 已创建")
    
    # 8. 测试：执行一些动作
    print("\n" + "="*60)
    print("开始测试状态转换（只使用本地模型）...")
    print("="*60)
    
    # 初始化状态
    state = cached_world_model.init_state()
    print(f"\n初始状态: {state.blocks_state}")
    
    # 获取可能的动作
    from examples.CoT.blocksworld.search_config import BlocksWorldConfig
    search_config = BlocksWorldConfig(base_model=local_model, prompt=prompt)
    search_config.update_example(example)
    
    actions = search_config.get_actions(state)
    print(f"\n可能的动作数量: {len(actions)}")
    
    # 执行几个动作（限制数量）
    test_actions = actions[:2]  # 只测试前 2 个动作
    
    for i, action in enumerate(test_actions, 1):
        print(f"\n--- 动作 {i}: {action} ---")
        print(f"当前状态: {state.blocks_state[:100]}...")
        
        try:
            # 执行动作（只使用本地模型）
            new_state, aux = cached_world_model.step(state, action)
            
            print(f"新状态: {new_state.blocks_state[:100]}...")
            goal_reached, goal_score = aux.get('goal_reached', (False, 0.0))
            print(f"目标达成: {goal_reached} (分数: {goal_score:.2f})")
            
            state = new_state
            
            # 检查是否终止
            if cached_world_model.is_terminal(state):
                print("✓ 已达到终止状态")
                break
        except Exception as e:
            print(f"✗ 预测失败: {e}")
            break
    
    # 9. 打印统计信息
    print("\n" + "="*60)
    cached_world_model.print_stats()
    
    # 10. 显示内存中的经验
    print("\n内存中的经验（最新 3 条）:")
    for i, memory in enumerate(memory_store.memories[-3:], 1):
        print(f"\n经验 {i}:")
        print(f"  状态: {memory.state[:60]}...")
        print(f"  动作: {memory.action}")
        print(f"  结果: {memory.next_state[:60]}...")
        print(f"  时间: {memory.timestamp}")
    
    print("\n" + "="*60)
    print("✓ 测试完成！")
    print("="*60)
    print(f"\n内存文件保存在: {memory_file}")
    print("注意：所有预测都只使用了本地模型，没有调用 ChatGPT API")


if __name__ == "__main__":
    main()
