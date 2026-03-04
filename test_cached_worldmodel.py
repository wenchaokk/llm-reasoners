"""测试带缓存的 WorldModel（简化版）

这个脚本可以在没有本地模型的情况下运行，只使用 ChatGPT 建立内存。
"""

import os
import sys
import json

# 添加项目根目录到路径
_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.lm import OpenAIModelWithUsage
from reasoners.world_model import CachedWorldModel, MemoryStore
from examples.CoT.blocksworld.world_model import BlocksWorldModel
from load_api_key import load_api_key


def main():
    """测试带缓存的 WorldModel"""
    
    print("="*60)
    print("测试带缓存的 WorldModel")
    print("="*60)
    
    # 1. 加载 API Key
    if not load_api_key():
        print("错误：无法加载 API Key")
        return
    
    # 2. 加载示例数据
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
    
    # 3. 加载提示词
    prompt_path = os.path.join(_root, "examples/CoT/blocksworld/prompts/pool_prompt_v1.json")
    if not os.path.exists(prompt_path):
        print(f"错误：找不到提示词文件 {prompt_path}")
        return
    
    with open(prompt_path, 'r') as f:
        prompt = json.load(f)
    
    # 4. 初始化 ChatGPT 模型
    print("\n初始化 ChatGPT 模型...")
    chatgpt_model = OpenAIModelWithUsage(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.0,
        additional_prompt="CONTINUE"
    )
    print("✓ ChatGPT 模型已加载")
    
    # 5. 创建基础 WorldModel
    base_world_model = BlocksWorldModel(
        base_model=chatgpt_model,
        prompt=prompt,
        max_steps=6
    )
    base_world_model.update_example(example)
    
    # 6. 创建内存存储
    memory_file = os.path.join(_root, "world_model_memory.json")
    memory_store = MemoryStore(
        memory_file=memory_file,
        similarity_threshold=0.7,
        max_memory_size=1000
    )
    print(f"\n✓ 内存存储已初始化: {memory_file}")
    print(f"  当前内存中有 {len(memory_store.memories)} 条经验")
    
    # 7. 创建带缓存的 WorldModel（不使用本地模型）
    print("\n创建带缓存的 WorldModel（仅使用 ChatGPT，建立内存）...")
    cached_world_model = CachedWorldModel(
        base_world_model=base_world_model,
        chatgpt_model=chatgpt_model,
        local_model=None,  # 不使用本地模型
        memory_store=memory_store,
        use_cache=False,  # 禁用缓存（因为没有本地模型）
        cache_threshold=0.7
    )
    print("✓ 带缓存的 WorldModel 已创建")
    print("  注意：当前只使用 ChatGPT，但会建立内存供以后使用")
    
    # 8. 测试：执行一些动作
    print("\n" + "="*60)
    print("开始测试状态转换...")
    print("="*60)
    
    # 初始化状态
    state = cached_world_model.init_state()
    print(f"\n初始状态: {state.blocks_state}")
    
    # 获取可能的动作
    from examples.CoT.blocksworld.search_config import BlocksWorldConfig
    search_config = BlocksWorldConfig(base_model=chatgpt_model, prompt=prompt)
    search_config.update_example(example)
    
    actions = search_config.get_actions(state)
    print(f"\n可能的动作数量: {len(actions)}")
    
    # 执行几个动作（限制数量以节省 API 调用）
    test_actions = actions[:2]  # 只测试前 2 个动作
    
    for i, action in enumerate(test_actions, 1):
        print(f"\n--- 动作 {i}: {action} ---")
        print(f"当前状态: {state.blocks_state[:100]}...")
        
        # 执行动作
        new_state, aux = cached_world_model.step(state, action)
        
        print(f"新状态: {new_state.blocks_state[:100]}...")
        goal_reached, goal_score = aux.get('goal_reached', (False, 0.0))
        print(f"目标达成: {goal_reached} (分数: {goal_score:.2f})")
        
        state = new_state
        
        # 检查是否终止
        if cached_world_model.is_terminal(state):
            print("✓ 已达到终止状态")
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
    
    # 11. 显示 API 使用情况
    usage = chatgpt_model.get_usage_summary()
    print("\n" + "="*60)
    print("API 使用情况:")
    print(f"  总调用次数: {usage['call_count']}")
    print(f"  总 tokens: {usage['total_tokens']}")
    print(f"  总费用: ${usage['total_cost_usd']:.4f}")
    print("="*60)
    
    print("\n✓ 测试完成！")
    print(f"\n内存文件保存在: {memory_file}")
    print("下次运行时，历史经验会自动加载")
    print("\n提示：如果要使用本地模型进行缓存预测，请运行:")
    print("  python examples/CoT/blocksworld/cached_world_model_demo.py")


if __name__ == "__main__":
    main()
