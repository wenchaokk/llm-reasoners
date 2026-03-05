"""两种预测方式对比（代码库正式入口）

本脚本是「ChatGPT 预测」与「本地小模型 + 内存经验预测」的官方对比入口：
- 模式 1：BlocksWorldModel + ChatGPT（每一步都调 API）
- 模式 2：CachedWorldModel(local_only=True) + 本地模型 + 内存经验（不调 API）

在同一批测试用例上运行两种模式，用规则给出的正确下一状态作为标准，统计并对比预测成功率。
用法见 GENERATE_EXPERIENCES_README.md「两种预测方式对比」一节。
"""

import os
import sys
import re
import json
import argparse

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

import reasoners.benchmark.bw_utils as bw_utils
from examples.CoT.blocksworld.world_model import BlocksWorldModel, BWState


def normalize_state(s: str) -> str:
    """标准化状态字符串便于比较（排序子句、统一空格）"""
    if not s or not s.strip():
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(sorted(parts))


def build_test_cases(max_cases: int = 50):
    """用规则生成器从多种初始状态生成 (state, action, ground_truth_next_state) 测试用例"""
    from reasoners.world_model.experience_generator import RuleBasedExperienceGenerator

    generator = RuleBasedExperienceGenerator()
    base_states = [
        "hand is empty, the red block is on the table, the blue block is on top of the red block, the red block is clear",
        "hand is empty, the red block is on the table, the blue block is on the table, the red block is clear, the blue block is clear",
        "is holding the red block, the blue block is on the table, the blue block is clear",
        "hand is empty, the red block is on the table, the blue block is on the table, the green block is on top of the red block, the red block is clear, the green block is clear",
        "is holding the blue block, the red block is on the table, the green block is on top of the red block, the red block is clear",
    ]
    test_cases = []
    for init in base_states:
        exp_list = generator.generate_from_state(
            init, max_depth=5, max_experiences=max(3, max_cases // 5)
        )
        for state, action, next_state in exp_list:
            if next_state and state and action:
                test_cases.append((state, action, next_state))
            if len(test_cases) >= max_cases:
                break
        if len(test_cases) >= max_cases:
            break
    return test_cases[:max_cases]


def evaluate_world_model(world_model, test_cases, example_for_goal):
    """在测试用例上评估 WorldModel，返回 (正确数, 总数)"""
    world_model.update_example(example_for_goal)
    correct, total = 0, 0
    for state_str, action_str, expected_next in test_cases:
        state = BWState(
            step_idx=0,
            last_blocks_state=state_str,
            blocks_state=state_str,
            buffered_action="",
        )
        try:
            new_state, _ = world_model.step(state, action_str)
            pred = new_state.blocks_state
            if normalize_state(pred) == normalize_state(expected_next):
                correct += 1
            total += 1
        except Exception as e:
            # 预测异常算错
            total += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="对比 ChatGPT 与 本地模型+内存 的预测成功率")
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.path.join(_root, "examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json"),
        help="仅用于取 example 做 goal（可选）",
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default=os.path.join(_root, "world_model_memory.json"),
        help="内存经验文件（本地+内存模式使用）",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="",
        help="本地模型路径（.gguf）。不填则自动查找 models/ 下已下载模型（需先运行 setup_local_model.py）",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=30,
        help="最多使用的测试用例数",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI 模型名",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=1024,
        help="本地模型 context 长度，70B 显存紧时可设为 512",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="放到 GPU 的层数，-1=全部；OOM 时可设为 40 把部分层放 CPU",
    )
    args = parser.parse_args()
    if not args.local_model_path:
        models_dir = os.path.join(_root, "models")
        for name in ("llama-3.1-70b-instruct.q4_k_m.gguf", "llama-3.2-8b-instruct.q4_k_m.gguf", "llama-3.2-3b-instruct.q4_k_m.gguf", "smollm2-1.7b-instruct.q4_k_m.gguf", "reasoning-llama-3.2-1b.q4_k_m.gguf", "tinyllama-1.1b-chat.q4_k_m.gguf"):
            default_local = os.path.join(models_dir, name)
            if os.path.exists(default_local):
                args.local_model_path = default_local
                break

    # 加载 API Key（ChatGPT 模式需要）
    try:
        from load_api_key import load_api_key

        load_api_key()
    except Exception:
        pass

    # 提示词
    prompt_path = os.path.join(_root, "examples/CoT/blocksworld/prompts/pool_prompt_v1.json")
    if not os.path.exists(prompt_path):
        print(f"错误：找不到提示词 {prompt_path}")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = json.load(f)

    # 测试用例（由规则生成，不依赖数据集格式）
    print("正在生成测试用例（规则生成）...")
    test_cases = build_test_cases(args.max_cases)
    if not test_cases:
        print("错误：未能生成测试用例")
        return
    print(f"  共 {len(test_cases)} 条 (state, action, expected_next) 测试用例\n")

    # 用于 goal 的 example（step 内会调 extract_goals，需含 question 键）
    example_for_goal = {
        "question": "[STATEMENT]\nAs initial conditions I have that, hand is empty, the red block is on the table. My goal is to the red block is on the table. My plan is as follows"
    }
    if os.path.exists(args.data_file):
        try:
            with open(args.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "question" in data[0]:
                example_for_goal = data[0]
            elif isinstance(data, dict) and "question" in data:
                example_for_goal = data
        except Exception:
            pass

    results = {}

    # ---------- 模式 1：仅 ChatGPT ----------
    print("=" * 60)
    print("模式 1：仅 ChatGPT 预测")
    print("=" * 60)
    try:
        from reasoners.lm import OpenAIModelWithUsage

        chatgpt_lm = OpenAIModelWithUsage(
            model=args.openai_model,
            max_tokens=512,
            temperature=0.0,
            additional_prompt="CONTINUE",
        )
        base_wm_chatgpt = BlocksWorldModel(
            base_model=chatgpt_lm,
            prompt=prompt,
            max_steps=6,
        )
        c, t = evaluate_world_model(base_wm_chatgpt, test_cases, example_for_goal)
        results["chatgpt"] = {"correct": c, "total": t, "rate": c / t if t else 0.0}
        print(f"  正确: {c} / {t}")
        print(f"  预测成功率: {results['chatgpt']['rate']:.2%}\n")
    except Exception as e:
        print(f"  ChatGPT 模式运行失败: {e}\n")
        results["chatgpt"] = None

    # ---------- 模式 2：仅本地小模型 + 内存经验 ----------
    print("=" * 60)
    print("模式 2：仅本地小模型 + 内存经验")
    print("=" * 60)
    if not args.local_model_path or not os.path.exists(args.local_model_path):
        print("  未提供或找不到本地模型路径，跳过本地+内存模式")
        print("  使用: --local_model_path path/to/model.gguf\n")
        results["local_memory"] = None
    else:
        try:
            from reasoners.lm import LlamaCppModel
            from reasoners.world_model import CachedWorldModel, MemoryStore

            local_lm = LlamaCppModel(
                path=args.local_model_path,
                n_ctx=getattr(args, "n_ctx", 1024),
                n_gpu_layers=getattr(args, "n_gpu_layers", -1),
            )
            base_wm_local = BlocksWorldModel(
                base_model=local_lm,
                prompt=prompt,
                max_steps=6,
            )
            memory_store = MemoryStore(
                memory_file=args.memory_file,
                similarity_threshold=0.7,
                max_memory_size=10000,
            )
            cached_wm = CachedWorldModel(
                base_world_model=base_wm_local,
                chatgpt_model=None,
                local_model=local_lm,
                memory_store=memory_store,
                use_cache=True,
                cache_threshold=0.7,
                local_only=True,
            )
            c, t = evaluate_world_model(cached_wm, test_cases, example_for_goal)
            results["local_memory"] = {"correct": c, "total": t, "rate": c / t if t else 0.0}
            print(f"  正确: {c} / {t}")
            print(f"  预测成功率: {results['local_memory']['rate']:.2%}")
            print(f"  内存经验数: {len(memory_store.memories)}\n")
        except Exception as e:
            print(f"  本地+内存模式运行失败: {e}\n")
            import traceback

            traceback.print_exc()
            results["local_memory"] = None

    # ---------- 对比汇总 ----------
    print("=" * 60)
    print("预测成功率对比")
    print("=" * 60)
    print(f"  测试用例数: {len(test_cases)}")
    if results.get("chatgpt"):
        r = results["chatgpt"]
        print(f"  ChatGPT:               {r['correct']}/{r['total']} = {r['rate']:.2%}")
    else:
        print("  ChatGPT:               未运行或失败")
    if results.get("local_memory"):
        r = results["local_memory"]
        print(f"  本地小模型+内存经验:   {r['correct']}/{r['total']} = {r['rate']:.2%}")
    else:
        print("  本地小模型+内存经验:   未运行或失败")
    print("=" * 60)


if __name__ == "__main__":
    main()
