"""演示如何使用带内存缓存的 WorldModel

这个脚本展示了如何：
1. 使用 ChatGPT 作为主模型（用于新预测）
2. 使用本地小模型（LlamaCpp）作为缓存模型（用于基于历史经验的预测）
3. 建立内存系统存储历史经验
4. 自动查询和使用历史经验
"""

import os
import sys
import json

# 添加项目根目录到路径
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, _root)

from reasoners.lm import OpenAIModelWithUsage, LlamaCppModel
from reasoners.world_model import CachedWorldModel, MemoryStore
from examples.CoT.blocksworld.world_model import BlocksWorldModel
from load_api_key import load_api_key


def main():
    """主函数：演示带缓存的 WorldModel"""
    
    print("="*60)
    print("带内存缓存的 WorldModel 演示")
    print("="*60)
    
    # 1. 加载 API Key
    if not load_api_key():
        print("错误：无法加载 API Key")
        return
    
    # 2. 加载示例数据
    data_path = os.path.join(_root, "examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json")
    with open(data_path, 'r') as f:
        examples = json.load(f)
    
    if not examples:
        print("错误：无法加载示例数据")
        return
    
    example = examples[0]  # 使用第一个示例
    print(f"\n使用示例: {example.get('id', 'unknown')}")
    
    # 3. 加载提示词
    prompt_path = os.path.join(_root, "examples/CoT/blocksworld/prompts/pool_prompt_v1.json")
    with open(prompt_path, 'r') as f:
        prompt = json.load(f)
    
    # 4. 初始化模型
    print("\n初始化模型...")
    
    # ChatGPT 模型（用于新预测）
    chatgpt_model = OpenAIModelWithUsage(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.0,
        additional_prompt="CONTINUE"
    )
    print("✓ ChatGPT 模型已加载")
    
    # 本地小模型（用于基于历史经验的预测）
    # 注意：需要先下载模型文件
    local_model_path = input("\n请输入本地模型路径（LlamaCpp 格式，回车跳过使用本地模型）: ").strip()
    local_model = None
    
    if local_model_path and os.path.exists(local_model_path):
        try:
            local_model = LlamaCppModel(path=local_model_path)
            print(f"✓ 本地模型已加载: {local_model_path}")
        except Exception as e:
            print(f"✗ 本地模型加载失败: {e}")
            print("  将只使用 ChatGPT（无缓存功能）")
    else:
        print("⚠ 未配置本地模型，将只使用 ChatGPT（无缓存功能）")
    
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
    
    # 7. 创建带缓存的 WorldModel
    cached_world_model = CachedWorldModel(
        base_world_model=base_world_model,
        chatgpt_model=chatgpt_model,
        local_model=local_model,
        memory_store=memory_store,
        use_cache=(local_model is not None),
        cache_threshold=0.7
    )
    print("✓ 带缓存的 WorldModel 已创建")
    
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
    print(f"\n可能的动作: {actions[:3]}... (共 {len(actions)} 个)")
    
    # 执行几个动作
    test_actions = actions[:3]  # 测试前 3 个动作
    
    for i, action in enumerate(test_actions, 1):
        print(f"\n--- 动作 {i}: {action} ---")
        print(f"当前状态: {state.blocks_state}")
        
        # 执行动作
        new_state, aux = cached_world_model.step(state, action)
        
        print(f"新状态: {new_state.blocks_state}")
        print(f"目标达成: {aux.get('goal_reached', (False, 0.0))[0]}")
        
        state = new_state
        
        # 检查是否终止
        if cached_world_model.is_terminal(state):
            print("✓ 已达到终止状态")
            break
    
    # 9. 打印统计信息
    cached_world_model.print_stats()
    
    # 10. 显示内存中的一些经验
    print("\n内存中的经验示例（前 3 条）:")
    for i, memory in enumerate(memory_store.memories[-3:], 1):
        print(f"\n经验 {i}:")
        print(f"  状态: {memory.state[:50]}...")
        print(f"  动作: {memory.action}")
        print(f"  结果: {memory.next_state[:50]}...")
        print(f"  时间: {memory.timestamp}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print(f"\n内存文件保存在: {memory_file}")
    print("下次运行时，历史经验会自动加载")


if __name__ == "__main__":
    main()
