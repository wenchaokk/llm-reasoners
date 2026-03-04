"""
积木世界问题的 Chain-of-Thought (CoT) 推理脚本

这个脚本展示了如何使用简单的 Chain-of-Thought 方法解决积木世界问题。
CoT 是最简单的推理方法：直接让 LLM 生成完整的动作序列，不使用搜索算法。

与使用搜索算法的方法（如 RAP、ToT）不同，CoT 只生成一个推理链，
不进行多路径探索和评估。
"""

from reasoners.lm import ExLlamaModel, HFModel
import json
from reasoners.benchmark import BWEvaluator
import fire
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm import OpenAIModelWithUsage
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm import  Llama2Model, Llama3Model

class CoTReasoner():
    """Chain-of-Thought 推理器
    
    这是一个简单的推理器，不使用搜索算法。
    它直接让 LLM 生成完整的动作序列（推理链）。
    
    特点：
    - 简单快速：只生成一个推理链
    - 不需要搜索：不探索多个路径
    - 适合简单任务：对于复杂任务可能效果不佳
    """
    
    def __init__(self, base_model, temperature=0.8, model_type="completion"):
        """初始化 CoT 推理器
        
        Args:
            base_model: LLM 模型对象
            temperature: 采样温度（控制输出的随机性）
            model_type: 模型类型
                - "completion": 完成式模型（如 LLaMA、GPT-3）
                - "chat": 对话式模型（如 GPT-4、Claude）
        """
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type

    def __call__(self, example, prompt=None):
        """执行 CoT 推理
        
        这个方法构建提示词，让 LLM 生成完整的动作序列。
        与搜索算法不同，这里只生成一次，不进行多路径探索。
        
        Args:
            example: 问题示例，包含初始状态和目标
                - example["init"]: 初始积木状态
                - example["goal"]: 目标积木状态
            prompt: 提示词模板字典
                - prompt["icl"]: in-context learning 提示词模板
                
        Returns:
            str: LLM 生成的完整动作序列（推理链）
        """
        # 构建提示词：替换模板中的占位符
        # <init_state>: 初始积木状态
        # <goals>: 目标积木状态
        # <action>: 动作序列（初始为空，让 LLM 生成）
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")
        
        # 根据模型类型选择不同的生成方式
        if self.model_type == "completion":
            # 完成式模型：使用 eos_token_id 控制生成停止
            # '\n[' 表示遇到换行符后跟 '[' 时停止（这是动作序列的结束标记）
            output = self.base_model.generate([inputs],
                                          hide_input=True,          # 只返回生成的部分
                                          do_sample=True,           # 使用采样（非贪心）
                                          temperature=self.temperature,
                                          eos_token_id='\n[').text[0][:-1].strip()  # 移除最后的 '[' 字符
        elif self.model_type == "chat":
            # 对话式模型：直接生成，然后移除结束标记
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature).text[0].replace("[PLAN END]", "").strip()
        
        return output

def main(model_dir, data_path, prompt_path, disable_log=False, batch_size=1, config_file: str = "examples/CoT/blocksworld/data/bw_config.yaml", domain_file: str = "examples/CoT/blocksworld/data/generated_domain.pddl", resume=0, log_dir=None, temperature=0.8, exllama_mem_map: str = None, quantized="int8", llama_path=None, llama_size=None):
    """主函数：运行 CoT 推理评估
    
    这个函数是脚本的入口点，它：
    1. 加载并初始化 LLM 模型
    2. 创建 CoT 推理器
    3. 在数据集上评估推理器
    4. 输出准确率
    
    Args:
        model_dir: 模型路径或模型类型标识符
            - "google": 使用 Google Gemini API
            - "openai": 使用 OpenAI API
            - "anthropic": 使用 Anthropic Claude API
            - "llama2": 使用 LLaMA-2 模型
            - "llama3": 使用 LLaMA-3 模型
            - 其他: HuggingFace 模型路径
        data_path: 数据集文件路径（JSON 格式）
        prompt_path: 提示词模板文件路径（JSON 格式）
        disable_log: 是否禁用日志记录
        batch_size: 批处理大小（当前未使用，CoT 是单样本生成）
        config_file: 积木世界配置文件路径（YAML 格式）
        domain_file: 积木世界域文件路径（PDDL 格式）
        resume: 从哪个样本开始（用于断点续跑）
        log_dir: 日志目录（如果为 None 会自动生成）
        temperature: 采样温度（控制输出的随机性）
        exllama_mem_map: ExLlama 模型的内存映射配置（用于多 GPU）
        quantized: 量化方式（"int8", "nf4", "fp4" 等）
        llama_path: LLaMA 模型路径（当 model_dir 为 "llama2" 或 "llama3" 时使用）
        llama_size: LLaMA 模型大小（如 "7B", "13B" 等）
        
    Returns:
        int: 0 表示成功
    """
    
    # ========================================================================
    # 1. 初始化 LLM 模型
    # ========================================================================
    # 根据 model_dir 的值选择不同的模型后端
    
    # 注释掉的代码：ExLlama 模型示例
    # base_model = ExLlamaModel(model_dir,
    #                     mem_map=exllama_mem_map, max_batch_size=batch_size,
    #                     max_new_tokens=300, max_seq_length=2048)
    
    if model_dir == "google":
        # Google Gemini API
        base_model = BardCompletionModel("gemini-pro", additional_prompt="CONTINUE")
    elif model_dir == "openai":
        # OpenAI API（从 api_key.txt 或环境变量读取 Key）
        import sys
        import os
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            from load_api_key import load_api_key
            load_api_key()
        except ImportError:
            pass
        base_model = OpenAIModelWithUsage("gpt-4o-mini", additional_prompt="CONTINUE")
    elif model_dir == "anthropic":
        # Anthropic Claude API
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="CONTINUE")
    elif model_dir == 'llama2':
        # LLaMA-2 模型（需要本地模型文件）
        base_model = Llama2Model(llama_path, llama_size, max_batch_size=batch_size)
    elif model_dir == 'llama3':
        # LLaMA-3 模型（需要本地模型文件）
        base_model = Llama3Model(llama_path, llama_size, max_batch_size=batch_size)
    else:
        # HuggingFace 模型（支持量化）
        base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
    
    # ========================================================================
    # 2. 加载提示词模板
    # ========================================================================
    with open(prompt_path) as f:
        prompt = json.load(f)

    # ========================================================================
    # 3. 创建 CoT 推理器
    # ========================================================================
    # 根据模型类型选择推理器类型
    # 对话式模型（OpenAI、Google、Anthropic）使用 "chat" 模式
    # 完成式模型（LLaMA、HuggingFace）使用 "completion" 模式
    reasoner = CoTReasoner(
        base_model, 
        temperature=temperature, 
        model_type="chat" if model_dir in ["openai", "google", "claude"] else "completion"
    )
    
    # ========================================================================
    # 4. 创建评估器并运行评估
    # ========================================================================
    # BWEvaluator 是积木世界问题的专用评估器
    evaluator = BWEvaluator(
        config_file=config_file,        # 积木世界配置
        domain_file=domain_file,        # 积木世界域定义
        data_path=data_path,            # 数据集路径
        init_prompt=prompt,            # 初始提示词模板
        disable_log=disable_log,        # 是否禁用日志
        output_extractor=lambda x: x,  # 输出提取器（CoT 直接返回生成的文本）
        sample_prompt_type="rap"       # 提示词类型（"rap" 包含 CoT 格式）
    )
    
    # 在数据集上评估推理器
    # shuffle_prompt=True: 打乱 few-shot 示例的顺序（提高鲁棒性）
    # num_shot=4: 使用 4 个 few-shot 示例
    accuracy = evaluator.evaluate(
        reasoner, 
        shuffle_prompt=True, 
        num_shot=4, 
        resume=resume, 
        log_dir=log_dir
    )
    
    # 输出最终准确率
    print(f'accuracy: {accuracy:.4f}')
    # 若使用 OpenAI，打印累计 Token 和费用
    if model_dir == "openai" and hasattr(base_model, "get_usage_summary"):
        s = base_model.get_usage_summary()
        print(f"\n[API 使用统计] total_tokens={s['total_tokens']}, cost≈${s['cost_usd']:.4f}, calls={s['call_count']}")
    return 0

if __name__ == '__main__':
    """脚本入口
    
    使用 fire 库提供命令行接口，可以直接通过命令行参数调用 main 函数。
    
    使用示例:
        python cot_inference.py \
            --model_dir openai \
            --data_path examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json \
            --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json \
            --temperature 0.8
    """
    fire.Fire(main)