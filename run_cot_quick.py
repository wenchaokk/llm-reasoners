"""快速运行 Chain-of-Thought demo（仅 CoT 部分）"""
import os
import sys

from load_api_key import load_api_key

if not load_api_key():
    print("ERROR: Please set OPENAI_API_KEY.")
    print("  Option 1: Edit api_key.txt, replace sk-your-key-here with your key")
    print("  Option 2: $env:OPENAI_API_KEY = \"sk-your-key\"")
    sys.exit(1)

from reasoners.benchmark import BWEvaluator
from reasoners.lm import OpenAIModelWithUsage

with open("examples/CoT/blocksworld/prompts/pool_prompt_v1.json") as f:
    import json
    prompt = json.load(f)

evaluator = BWEvaluator(
    config_file="examples/CoT/blocksworld/data/bw_config.yaml",
    domain_file="examples/CoT/blocksworld/data/generated_domain.pddl",
    data_path="examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json",
    init_prompt=prompt,
)
prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=4)
example = evaluator.full_dataset[1]

model = OpenAIModelWithUsage(
    model="gpt-4o-mini", max_tokens=512, temperature=0.7, additional_prompt="CONTINUE"
)
cot_inputs = (
    prompt["icl"]
    .replace("<init_state>", example["init"])
    .replace("<goals>", example["goal"])
    .replace("<action>", "")
)
output = model.generate([cot_inputs], hide_input=True, eos_token_id="\n[").text[0][
    :-1
].strip()
print("Generated:", output)
print("Usage:", model.get_usage_summary())
