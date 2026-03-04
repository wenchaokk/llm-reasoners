"""
AutoRace 评估系统

这个模块实现了 AutoRace 评估方法，用于自动评估 LLM 生成的推理链的正确性。
主要功能：
1. 生成评估标准（criterion）：通过对比正确答案和错误答案生成评估标准
2. 评估推理链：使用生成的评估标准评估推理链的正确性
3. 计算评估准确率：与人工标注对比，计算评估系统的准确率

参考论文：AutoRace: Automatically Generating Evaluation Criteria for Reasoning Chains
"""

import os
from typing import Optional, Literal
import time
import pandas as pd
import json
import fire
import jsonlines
from tqdm import tqdm
from openai import OpenAI

# ============================================================================
# 默认配置：OpenAI API 调用参数
# ============================================================================
MAX_TOKENS = 4096                    # 最大生成 token 数
OPENAI_MODEL = 'gpt-4-1106-preview' # 使用的 OpenAI 模型
TEMPERATURE = 0.7                    # 采样温度
TOP_P: float = 1.0                   # Top-p 采样参数
NUM_RETURN_SEQUENCES: int = 1        # 返回的序列数量
RATE_LIMIT_PER_MIN: Optional[int] = None  # 每分钟请求限制（None 表示无限制）
STOP: Optional[str] = None           # 停止词
LOGPROBS: Optional[int] = 0          # 是否返回 log 概率

# ============================================================================
# 数据集到提示词类型的映射
# 每个数据集都有对应的评估提示词模板（存储在 prompt.json 中）
# ============================================================================
PROMPT_TYPE_DICT = {
    'gsm8k': 'gsm8k_auto',              # 数学问题数据集
    'strategyqa': 'sq_auto',             # 策略问答数据集
    'aqua': 'aqua_auto',                 # 逻辑推理数据集
    'cosmos': 'cosmos_auto',              # 常识推理数据集
    'multistep_arithmetic': 'arith_auto', # 多步算术数据集
    'word_sorting': 'sort_auto',          # 单词排序数据集
    'logical_deduction': 'logic_auto'      # 逻辑推理数据集
}

# ============================================================================
# 初始化 OpenAI 客户端
# ============================================================================
OPENAI_KEY = os.getenv('OPENAI_API_KEY', input('Please input your OpenAI API key: '))
client = OpenAI(
    api_key = OPENAI_KEY
)

def generate(prompt):
    """使用 OpenAI API 生成文本
    
    这是一个带重试机制的生成函数，会自动处理 API 限流和错误。
    
    Args:
        prompt: 输入提示词
        
    Returns:
        生成的文本列表（每个选择对应一个文本）
        
    Raises:
        ValueError: 如果模型名称未知
    """
    while(True):
        try:
            # 如果设置了速率限制，在请求之间休眠以避免超过限制
            if RATE_LIMIT_PER_MIN is not None:
                time.sleep(60 / RATE_LIMIT_PER_MIN)
            
            # 检查是否为聊天模型（GPT-3.5 或 GPT-4）
            if ('gpt-3.5-turbo' in OPENAI_MODEL) or ('gpt-4' in OPENAI_MODEL):
                messages = [{'role': 'user', 'content': prompt}]
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    n=NUM_RETURN_SEQUENCES,
                    stop=STOP,
                )
                # 提取生成的文本
                text=[choice.message.content for choice in response.choices]
                return text
            else:
                raise ValueError(f'Unknown OPENAI MODEL {OPENAI_MODEL}')
        except Exception as e:
            # 如果出错，打印错误信息并等待 5 秒后重试
            print(f'An Error Occured: {e}, sleeping for 5 seconds')
            time.sleep(5)

def autorace_criterion(dataset:str = 'aqua', example_wrong_chains:str = 'EXAMPLE_WRONG_CHAINS_AQUA.txt'):
    """生成评估标准（criterion）
    
    这个函数通过对比正确答案和错误答案来生成评估标准（对应论文中的 Fig 2）。
    
    工作流程：
    1. 读取包含错误推理链的示例文件
    2. 使用 LLM 分析这些错误，生成评估标准
    3. 将评估标准保存到 prompt.json 中
    
    输入格式要求（参考 EXAMPLE_WRONG_CHAINS_AQUA.txt）：
    -----------------------------------------------------------------------------
    Question:
    [问题描述]
    
    Reference answer:
    [正确答案的推理过程]
    
    Student answer:
    [错误答案的推理过程]
    -----------------------------------------------------------------------------
    
    Args:
        dataset: 数据集名称（如 'aqua', 'gsm8k' 等）
        example_wrong_chains: 包含错误推理链示例的文件路径
        
    Raises:
        AssertionError: 如果示例文件不存在
    """
    
    # 检查示例文件是否存在
    assert os.path.exists(example_wrong_chains), f'example_wrong_chains: {example_wrong_chains} does not exist!'
    
    # 读取错误推理链示例
    with open(example_wrong_chains) as f:
        EXAMPLE_WRONG_CHAINS = f.read()

    # 读取提示词模板
    with open('prompt.json') as f:
        prompt = json.load(f)
        
    # 检查是否已存在该数据集的评估标准
    if f"{dataset}_auto" in prompt:
        print(f'Warning: dataset {dataset} already exists in prompt.json, please check whether you want to overwrite it.')
        input('Press any key to continue...')
        
    # 使用 'criterion' 提示词模板生成评估标准
    # 这个提示词会要求 LLM 分析错误答案，总结出评估标准
    criterion_prompt = prompt['criterion'].format(EXAMPLE_WRONG_CHAINS)
    criterion_text = generate(criterion_prompt)
    print(criterion_text)
    
    # 清理生成的评估标准文本
    # 提取从 "1. **" 开始的部分
    criterion = '1. **' + criterion_text[0].split('1. **')[-1]
    
    # 移除编号（如 "1. ", "2. " 等），因为评估标准应该是一个整体
    import re
    criterion = re.sub(r'\d\. ', '', criterion)
    
    # 构建评估提示词模板
    # 这个模板会在评估时使用，要求 LLM 根据评估标准检查推理链
    evaluation_prompt = (
        'Below is a question and an answer from a student. '
        'You are required to check the correctness of the reasoning chains step by step. '
        'The criterions are as follows:\n\n{}\n\n'
        'Question:\n{{}}\n\n'
        'Student answer:\n{{}}\n\n'
        'Please check the answer through each criterion, and make sure you carefully examine each reasoning step. '
        'Finally, if there is any step that fails the verification, output a INCORRECT, else output a CORRECT.'
    ).format(criterion)
    
    # 将评估提示词保存到 prompt.json
    prompt[dataset + '_auto'] = evaluation_prompt

    with open('prompt.json', 'w') as f:
        json.dump(prompt, f)

def autorace_score(output_log_path:str):
    """计算并报告 AutoRace 评估分数
    
    AutoRace 分数 = 被评估为正确的推理链数量 / 总数量
    
    Args:
        output_log_path: 评估结果日志文件路径（JSONL 格式）
    """
    # 加载评估结果
    with jsonlines.open(output_log_path, mode='r') as reader:
        autorace = list(reader)

    # 计算分数
    total = len(autorace)
    incorrect = 0
    for i in range(total):
        # 如果评估结果包含 "INCORRECT"，则认为是错误的
        if 'INCORRECT' in autorace[i]['evaluation_result'][0]:
            incorrect += 1

    # 计算正确率（AutoRace 分数）
    score = (total - incorrect) / total
    print(f'autorace score: {score:.2f}')

def autorace_evaluation(
    dataset: str = "gsm8k", 
    reasoning_model: str = "eval_model",
    output_log_dir:str = 'logs/auto_race'
):
    """执行 AutoRace 评估
    
    这个函数使用生成的评估标准来评估推理链的正确性。
    
    使用场景：
    - 论文 Table 1：先用这个函数生成评估结果，然后用 test_evaluation_accuracy 
      与人工标注对比，得到 Table 1 的结果
    - 论文 Table 9：用这个函数计算 gsm8k、aqua、strategyqa 的 AutoRace 分数
    
    工作流程：
    1. 加载推理模型生成的推理链
    2. 对每个推理链，使用评估标准进行评估
    3. 保存评估结果并计算 AutoRace 分数
    
    Args:
        dataset: 数据集名称（如 'gsm8k', 'aqua' 等）
        reasoning_model: 推理模型名称（用于定位数据文件）
        output_log_dir: 输出日志目录
        
    Raises:
        ValueError: 如果数据集不在 PROMPT_TYPE_DICT 中
        AssertionError: 如果数据文件不存在
    """
    
    # 预定义的数据集列表
    predefined_datasets = ['gsm8k', 'strategyqa', 'aqua', 'cosmos', 'multistep_arithmetic', 'word_sorting', 'logical_deduction']
    
    # 检查数据集是否在预定义列表中
    if dataset not in predefined_datasets:
        print(f"Warning: The dataset '{dataset}' is not a predefined dataset.")
    
    # 检查数据集是否有对应的提示词类型
    if dataset not in PROMPT_TYPE_DICT:
        raise ValueError(f"dataset '{dataset}' is not in PROMPT_TYPE_DICT! Please add the prompt type to PROMPT_TYPE_DICT.")
    
    
    # 构建数据文件路径（推理模型生成的推理链）
    data_path = f'./data/{reasoning_model}/{dataset}.jsonl'
    assert os.path.exists(data_path), f'the output from {reasoning_model}: {data_path} does not exist!'
    
    # 创建输出目录
    assert output_log_dir is not None, 'output_log_dir should not be None'
    output_log_dir = os.path.join(output_log_dir, reasoning_model, dataset)
    os.makedirs(output_log_dir, exist_ok=True)
    output_log_path = f'{output_log_dir}/autorace_eval.jsonl'
    
    print("evaluating reasoning model: ", reasoning_model, " on dataset: ", dataset, "output log path: ", output_log_path)
    
    # 加载推理链数据
    import pandas as pd
    data = pd.read_json(data_path, lines=True)
    results = []
    
    # 对每个推理链进行评估
    for index in tqdm(range(len(data))):
        # 获取推理链
        reasoning_chain = data.loc[index, 'reasoning_chain']
        
        # 格式化清理：确保推理链以换行符开头
        if not reasoning_chain.startswith('\n'):
            reasoning_chain = '\n' + reasoning_chain
        # 移除末尾的多余换行符和句号
        reasoning_chain = reasoning_chain.rstrip('\n\n.')
        
        # 获取并清理问题
        raw_question = data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')  # 移除 "Q:" 前缀
        raw_question = raw_question.lstrip(' ')      # 移除开头的空格
        
        # 加载评估提示词模板
        with open('prompt.json') as f:
            prompts = json.load(f)
        
        # 使用对应数据集的评估提示词模板
        prompt = prompts[PROMPT_TYPE_DICT[dataset]].format(raw_question, reasoning_chain)
        # 清理格式问题（如 ".." -> "."）
        prompt = prompt.replace('..', '.')
        
        # 使用 LLM 生成评估结果
        evaluation_result = generate(prompt)
        
        # 保存评估结果
        tmp = {
            'index': index, 
            'evaluation_result': evaluation_result, 
            'question': raw_question, 
            'reasoning_chain': reasoning_chain, 
            'answer': data.loc[index, 'answer'], 
            'prompt': prompt
        }
        results.append(tmp)
        
        # 实时保存结果（每处理一个样本就保存一次，避免数据丢失）
        with jsonlines.open(output_log_path, mode='w') as writer:
            writer.write_all(results)
    
    # 计算并报告 AutoRace 分数
    autorace_score(output_log_path)
    

def test_evaluation_accuracy(output_name: str = time.strftime('%Y-%m-%d-%H-%M-%S')):
    """测试评估系统的准确率
    
    这个函数用于测试 AutoRace 评估系统的准确率，使用人工标注作为真实标签。
    用于复现论文中的 Table 1。
    
    工作流程：
    1. 对每个数据集执行 AutoRace 评估（如果还没有评估结果）
    2. 加载人工标注和评估结果
    3. 对比两者，计算对齐分数（align score）
    4. 保存不一致的案例用于错误分析
    
    对齐分数（align score）= 评估结果与人工标注一致的数量 / 总数量
    
    Args:
        output_name: 输出目录名称（默认为当前时间戳）
    """
    
    print("Start testing evaluation accuracy...")
    
    # 要测试的数据集列表
    datasets = ['gsm8k','strategyqa','cosmos', 'multistep_arithmetic','word_sorting','logical_deduction']

    model = "eval_model"
    eval_dir = "./logs/auto_race"
    human_label_dir = "./data/eval_model"
    
    # 对每个数据集进行测试
    for dataset in datasets:
        # 如果还没有评估结果，先执行评估
        if os.path.exists(f'{eval_dir}/{model}/{dataset}'):
            print(f'{eval_dir}/{model}/{dataset} exists, pass.')
        else:
            print(f'{eval_dir}/{model}/{dataset} does not exist, start autorace evaluation...')
            autorace_evaluation(dataset, model, eval_dir)
        
        # 加载人工标注和评估结果
        human_label_path = os.path.join(human_label_dir, f'{dataset}.jsonl')
        evaluator_label_path = os.path.join(eval_dir, f'{model}/{dataset}/autorace_eval.jsonl')
        
        with jsonlines.open(human_label_path, mode='r') as reader:
            human_labels = list(reader)
        
        with jsonlines.open(evaluator_label_path, mode='r') as reader:
            evaluator_labels = list(reader)
            
        # 确保人工标注数量不少于评估结果数量
        assert len(human_labels) >= len(evaluator_labels), f'there are unlabelled samples in {human_label_path} compared to {evaluator_label_path}!'

        # 统计对齐情况
        total = len(evaluator_labels)
        score = 0  # 对齐的数量
        correct_align_list = []      # 评估为正确且人工也标注为正确的索引
        incorrect_align_list = []    # 评估为错误且人工也标注为错误的索引
        incorrect_disagreement = []  # 评估为错误但人工标注为正确（不一致）
        correct_disagreement = []    # 评估为正确但人工标注为错误（不一致）
        
        # 对比每个样本的评估结果和人工标注
        for i in range(len(evaluator_labels)):
            output = evaluator_labels[i]['evaluation_result'][0]
            
            if 'INCORRECT' in output:
                # 评估结果为错误
                if int(human_labels[i]['human_label']) == 0:
                    # 人工也标注为错误：一致
                    incorrect_align_list.append(i)
                    score += 1
                else:
                    # 人工标注为正确：不一致
                    incorrect_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })
            else:
                # 评估结果为正确
                if int(human_labels[i]['human_label']) == 1:
                    # 人工也标注为正确：一致
                    correct_align_list.append(i)
                    score += 1
                else:
                    # 人工标注为错误：不一致
                    correct_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })

        # 创建输出目录
        output_dir = f'logs/error_analysis/{output_name}/{dataset}'
        correct_path = os.path.join(output_dir, 'correct_disagree')
        incorrect_path = os.path.join(output_dir, 'incorrect_disagree')
        align_score_log = os.path.join(output_dir, 'align_score.txt')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(correct_path, exist_ok=True)
        os.makedirs(incorrect_path, exist_ok=True)

        # 保存不一致的案例用于错误分析
        # 评估为错误但人工标注为正确的案例
        for sample in incorrect_disagreement:
            with open(os.path.join(incorrect_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')
        
        # 评估为正确但人工标注为错误的案例
        for sample in correct_disagreement:
            with open(os.path.join(correct_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')

        # 计算并保存对齐分数
        align_score = score / total
        print(f'align score for {dataset}: {align_score:.2f}')
        with open(align_score_log, 'w') as f:
            f.write(f'Align score: {align_score:.2f}\n')
            f.write(f'Total: {total}\n')
            f.write(f'Correct: {score}\n')
            f.write(f'Incorrect: {total - score}\n')
            f.write(f'Correct align list: {correct_align_list}\n')
            f.write(f'Incorrect align list: {incorrect_align_list}\n')
            f.write(f'Correct disagreement list: {correct_disagreement}\n')
            f.write(f'Incorrect disagreement list: {incorrect_disagreement}\n')    
    

def main(gen_criteria: bool = False, dataset: str = 'gsm8k', example_wrong_chains: str = 'EXAMPLE_WRONG_CHAINS_AQUA.txt',  reproduce_tab1: bool = False, reasoning_model: str = "eval_model", output_log: str = 'logs/auto_race'):
    """主函数：AutoRace 评估系统的入口
    
    根据参数选择执行不同的功能：
    1. 生成评估标准（gen_criteria=True）
    2. 执行评估（默认）
    3. 测试评估准确率（reproduce_tab1=True）
    
    Args:
        gen_criteria: 是否生成评估标准
        dataset: 数据集名称
        example_wrong_chains: 错误推理链示例文件路径
        reproduce_tab1: 是否复现 Table 1（测试评估准确率）
        reasoning_model: 推理模型名称
        output_log: 输出日志目录
    """
    if reproduce_tab1:
        # 测试评估准确率（复现 Table 1）
        test_evaluation_accuracy()
    elif gen_criteria:
        # 生成评估标准
        autorace_criterion(dataset, example_wrong_chains)
    else:
        # 执行 AutoRace 评估
        autorace_evaluation(dataset, reasoning_model, output_log)
    

if __name__ == '__main__':
    # 使用 fire 库提供命令行接口
    fire.Fire(main)
    

