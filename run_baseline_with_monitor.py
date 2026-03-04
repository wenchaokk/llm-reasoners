"""运行 CoT Baseline，并实时监控 API Token/费用"""
import os
import sys
import json
from datetime import datetime

# 确保项目根目录在 path 中
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# 加载 API Key
from load_api_key import load_api_key
if not load_api_key():
    print("ERROR: 请编辑 api_key.txt 填入你的 OpenAI API Key")
    sys.exit(1)

# 设置环境变量（VAL 指向包含 validate 可执行文件的目录）
os.environ.setdefault("PLANBENCH_PATH", os.path.join(_root, "LLMs-Planning"))
os.environ.setdefault("VAL", os.path.join(_root, "LLMs-Planning", "planner_tools", "VAL", "bin", "validate"))

from reasoners.benchmark import BWEvaluator
from reasoners.lm import OpenAIModelWithUsage

# 动态导入 CoTReasoner
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cot_inference",
    os.path.join(_root, "examples", "CoT", "blocksworld", "cot_inference.py")
)
_cot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cot)
CoTReasoner = _cot.CoTReasoner

def main(max_samples=None):
    """运行 CoT baseline，max_samples 限制样本数（None=全部）"""
    with open("examples/CoT/blocksworld/prompts/pool_prompt_v1.json") as f:
        prompt = json.load(f)

    config_file = "examples/CoT/blocksworld/data/bw_config.yaml"
    domain_file = "examples/CoT/blocksworld/data/generated_domain.pddl"
    data_path = "examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json"

    model = OpenAIModelWithUsage(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.0,
        additional_prompt="CONTINUE"
    )
    reasoner = CoTReasoner(model, temperature=0.0, model_type="chat")

    evaluator = BWEvaluator(
        config_file=config_file,
        domain_file=domain_file,
        data_path=data_path,
        init_prompt=prompt,
        disable_log=False,
        output_extractor=lambda x: x,
        sample_prompt_type="rap"
    )

    # 限制样本数
    if max_samples is not None:
        evaluator.full_dataset = evaluator.full_dataset[:max_samples]
        print(f"[限制] 仅运行前 {max_samples} 个样本\n")

    print("=" * 50)
    print("开始运行 CoT Baseline（每次调用后显示 Token 统计）")
    print("=" * 50)

    log_dir = f"logs/blocksworld_cot_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    accuracy = evaluator.evaluate(
        reasoner,
        shuffle_prompt=True,
        num_shot=4,
        resume=0,
        log_dir=log_dir
    )

    print("\n" + "=" * 50)
    print(f"准确率: {accuracy:.4f}")
    s = model.get_usage_summary()
    print(f"[累计统计] total_tokens={s['total_tokens']}, cost≈${s['cost_usd']:.4f}, calls={s['call_count']}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=None, help="限制样本数，默认全部")
    args = p.parse_args()
    main(max_samples=args.max_samples)
