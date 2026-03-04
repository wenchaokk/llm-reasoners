# ============================================================================
# LLM Reasoners 核心抽象层
# 这个文件定义了整个代码库的核心接口和抽象类
# ============================================================================

from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList
import inspect
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch

from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch

# 类型变量定义：用于泛型编程
State = TypeVar("State")      # 状态类型：表示推理过程中的中间状态
Action = TypeVar("Action")     # 动作类型：表示从一个状态到另一个状态的操作
Example = TypeVar("Example")  # 示例类型：表示输入的问题/任务
Trace = tuple[list[State], list[Action]]  # 轨迹类型：状态序列和动作序列的元组

def create_directory_if_not_exists(directory):
    """创建目录（如果不存在）
    
    这是一个工具函数，用于确保目录存在。
    如果目录不存在，则创建它（包括所有必要的父目录）。
    
    Args:
        directory: 要创建的目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


class GenerateOutput(NamedTuple):
    """LLM 生成输出的数据结构
    
    Attributes:
        text: 生成的文本列表（每个输入对应一个输出）
        log_prob: 可选的对数概率列表（用于计算奖励）
    """
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):
    """大语言模型接口抽象类
    
    所有 LLM 后端（OpenAI、HuggingFace、Anthropic 等）都需要实现这个接口
    这样搜索算法就可以与不同的 LLM 后端无缝配合
    """
    
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """从提示词列表生成文本
        
        Args:
            inputs: 提示词列表
            max_length: 总输出最大长度（输入+生成）
            max_new_tokens: 生成的最大 token 数（会覆盖 max_length）
            do_sample: 如果为 False，使用贪心解码
            temperature: 采样温度
            top_k: Top-k 采样
            top_p: Top-p 采样
            num_return_sequences: 返回的序列数量
            eos_token_id: 结束符 token ID。传入 str 会被转换为 token_id。
                         传入 list 会被视为多个可能的结束符
            hide_input: 如果为 True，只解码生成的部分
            output_log_probs: 如果为 True，也输出每个生成 token 的对数概率
            stopping_criteria: 停止条件
            
        Returns:
            GenerateOutput: 包含生成文本和对数概率的输出对象
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """获取下一个 token 的 logits（未归一化的概率）
        
        用于评估候选动作的概率，常用于 fast_reward 计算
        
        Args:
            prompt: 提示词（字符串或字符串列表）
            candidates: 候选 token 列表（用于评估每个候选的概率）
            postprocess: 可选的后处理，可以是 'log_softmax' 或 'softmax'
            **kwargs: 其他参数
            
        Returns:
            logits 列表，每个候选对应一个 logits 数组
        """
        ...

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """获取内容在给定前缀下的对数似然
        
        这是计算 fast_reward 的关键方法，用于快速评估一个动作的合理性
        
        Args:
            prefix: 前缀（不包含在对数似然计算中）
            contents: 要评估的内容列表（必须包含前缀）
            **kwargs: 其他参数
            
        Returns:
            对数似然数组，每个 content 对应一个值
        """
        ...

class Dynamics(ABC, Generic[State, Action]):
    """动态系统抽象类：定义状态转换规则
    
    这是 WorldModel 和 Environment 的基类，定义了状态空间和状态转换的基本接口
    """

    @abstractmethod
    def init_state(self) -> State:
        """初始化状态
        
        Returns:
            初始状态
        """
        ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """执行动作，转换到下一个状态
        
        这是核心方法：定义如何从一个状态通过执行动作到达下一个状态
        
        Args:
            state: 当前状态
            action: 要执行的动作
            
        Returns:
            下一个状态，以及可选的辅助数据字典（用于传递给奖励函数）
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """判断状态是否为终止状态
        
        Args:
            state: 要判断的状态
            
        Returns:
            如果为终止状态返回 True，否则返回 False
        """
        ...

class WorldModel(Dynamics, Generic[State, Action, Example]):
    """世界模型：定义推理问题的状态空间和状态转换规则
    
    这是用户需要实现的核心类之一。它定义了：
    1. 状态是什么（State 类型）
    2. 如何初始化状态（init_state）
    3. 如何转换状态（step）
    4. 什么情况下终止（is_terminal）
    
    示例：
        - 数学题：状态 = 已完成的步骤，动作 = 下一步计算
        - 积木问题：状态 = 积木排列，动作 = 移动积木
    """
    
    def __init__(self) -> None:
        self.example = None  # 当前要解决的问题
        self.prompt = None   # 提示词模板

    def update_example(self, example: Example, prompt = None) -> None:
        """更新当前问题和提示词
        
        在推理开始前，Reasoner 会调用这个方法设置当前问题
        
        Args:
            example: 要解决的问题
            prompt: 可选的提示词模板
        """
        if prompt is not None:
            self.prompt = prompt
        self.example = example

class DefaultWorldModel(WorldModel):
    """默认世界模型的实现
    
    这是一个简单的实现，只保存动作序列作为状态。
    适用于不需要复杂状态表示的任务（如简单的文本生成）。
    
    对于复杂任务，用户应该实现自己的 WorldModel。
    """

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def init_state(self):
        """初始化状态：返回空的动作列表"""
        return []

    def step(self, state, action):
        """执行动作：将新动作追加到状态（动作列表）中"""
        return state + [action], {}

    def is_terminal(self, state):
        """默认情况下状态永远不会终止（需要用户自己定义终止条件）"""
        return False

class Environment(Dynamics, Generic[State, Action]):
    """环境抽象类：用于真实环境（如 Gymnasium 环境）
    
    与 WorldModel 不同，Environment 表示真实的环境（如浏览器、游戏等），
    状态转换由环境本身处理，而不是由 LLM 预测。
    
    这是 BrowserGym 等真实环境适配器的基类。
    
    Attributes:
        env: 真实环境对象（如 Gymnasium 环境）
    """
    def __init__(self) -> None:
        self.env = None

class SearchConfig(ABC, Generic[State, Action, Example]):
    """搜索配置：定义动作空间和奖励函数
    
    这是用户需要实现的另一个核心类。它定义了：
    1. 在给定状态下有哪些可用动作（get_actions）
    2. 如何快速评估一个动作（fast_reward）- 可选但推荐
    3. 如何完整评估一个动作（reward）- 必需
    
    关键概念：
        - fast_reward: 快速但可能不准确的评估（如用 log-likelihood）
        - reward: 慢但更准确的评估（如验证答案正确性）
        
    设计理念：
        fast_reward 用于加速搜索（在扩展节点时快速评估），
        reward 用于最终评估（在状态转换后完整评估）。
    """
    
    def __init__(self) -> None:
        self.example = None  # 当前要解决的问题
        self.prompt = None   # 提示词模板

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        """获取在给定状态下的所有可用动作
        
        这是搜索算法的关键：它决定了搜索空间的大小。
        动作太多会导致搜索空间爆炸，动作太少可能找不到解。
        
        Args:
            state: 当前状态
            
        Returns:
            可用动作列表
        """
        ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        """快速评估动作的奖励（可选但推荐实现）
        
        这个方法应该快速执行，因为它会在搜索过程中被频繁调用。
        通常使用 LLM 的 log-likelihood 来评估动作的合理性。
        
        Args:
            state: 当前状态
            action: 要评估的动作
            
        Returns:
            (奖励值, 详细信息字典)
            详细信息会被传递给 reward 方法，避免重复计算
        """
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        """完整评估动作的奖励（必需实现）
        
        这个方法可以较慢，因为它用于最终评估。
        通常会结合 fast_reward 的结果和状态信息来计算。
        
        Args:
            state: 当前状态
            action: 要评估的动作
            **kwargs: 从 fast_reward 和 WorldModel.step 传递过来的信息
            
        Returns:
            (奖励值, 详细信息字典)
        """
        ...

    def update_example(self, example: Example, prompt = None) -> None:
        """更新当前问题和提示词"""
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    """搜索算法的输出协议
    
    所有搜索算法都应该返回符合这个协议的对象
    """
    terminal_state: State  # 终止状态（找到的最终状态）
    trace: Trace           # 轨迹（从初始状态到终止状态的状态和动作序列）


class SearchAlgorithm(ABC):
    """搜索算法抽象类
    
    这是搜索策略的接口。不同的算法（MCTS、Beam Search、DFS 等）都实现这个接口。
    
    设计模式：策略模式 - 可以轻松切换不同的搜索算法而不改变其他代码。
    """
    
    def __init__(self, **kwargs): 
        """初始化搜索算法
        
        Args:
            **kwargs: 算法特定的参数（如 MCTS 的 n_iters、depth_limit 等）
        """
        ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> AlgorithmOutput:
        """执行搜索算法
        
        这是搜索算法的核心方法。它使用 world_model 和 search_config 来搜索最优解。
        
        Args:
            world_model: 世界模型（定义状态转换）
            search_config: 搜索配置（定义动作和奖励）
            **kwargs: 其他参数
            
        Returns:
            AlgorithmOutput: 包含终止状态和轨迹的输出对象
        """
        ...


class Reasoner(ABC, Generic[State, Action, Example]):
    """推理器：组合三个核心组件的统一接口
    
    这是用户使用的主要类。它将 WorldModel、SearchConfig 和 SearchAlgorithm
    组合在一起，提供一个简单的接口来执行推理。
    
    使用示例：
        world_model = MyWorldModel(...)
        search_config = MySearchConfig(...)
        search_algo = MCTS(n_iters=10)
        reasoner = Reasoner(world_model, search_config, search_algo)
        result = reasoner(example)
    """
    
    def __init__(self,
                 world_model: Dynamics[State, Action],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        """初始化推理器
        
        Args:
            world_model: 世界模型（定义状态转换）
            search_config: 搜索配置（定义动作和奖励）
            search_algo: 搜索算法（定义搜索策略）
        """
        self.dynamics = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Optional[Example] = None, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        """执行推理
        
        这是推理器的主要接口。给定一个问题，它会使用配置的搜索算法找到最优解。
        
        Args:
            example: 要解决的问题
            prompt: 可选的提示词模板
            **kwargs: 传递给搜索算法的其他参数
            
        Returns:
            AlgorithmOutput: 包含终止状态和轨迹的输出对象
        """
        if isinstance(self.dynamics, WorldModel):
            if example is None:
                raise ValueError("An example must be provided when using WorldModel")
            # 更新世界模型和搜索配置中的问题和提示词
            self.dynamics.update_example(example, prompt=prompt)
            self.search_config.update_example(example, prompt=prompt)
        # 调用搜索算法执行搜索
        return self.search_algo(self.dynamics, self.search_config, **kwargs)

class Evaluator():
    """评估器：用于在数据集上评估推理器的性能
    
    这个类提供了统一的评估接口，可以：
    1. 在数据集上运行推理器
    2. 计算准确率
    3. 保存结果和日志
    """
    
    @abstractmethod
    def __init__(self) -> None:
        """初始化评估器"""
        pass

    @abstractmethod
    def sample_prompt(self,
                      shuffle_prompt,
                      num_shot,
                      sample_prompt_type):
        """采样提示词（few-shot learning）
        
        Args:
            shuffle_prompt: 是否打乱示例顺序
            num_shot: few-shot 的示例数量
            sample_prompt_type: 提示词类型
        """
        pass
    
    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 log_dir=None):
        """在数据集上评估推理器
        
        Args:
            reasoner: 要评估的推理器
            shuffle_prompt: 是否打乱提示词中的示例顺序
            num_shot: few-shot 的示例数量
            resume: 从哪个样本开始（用于断点续跑）
            log_dir: 日志目录（如果为 None 会自动生成）
            
        Returns:
            准确率（0-1 之间的浮点数）
        """

        # 从指定位置开始的数据集（支持断点续跑）
        self.dataset = list(self.full_dataset)[resume:]
        
        # 获取搜索算法的名称（用于日志目录命名）
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        # 只在主进程中创建日志目录（分布式训练时）
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # 如果没有指定日志目录，自动生成一个（包含数据集名、算法名和时间戳）
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            # 保存命令行参数（用于复现实验）
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0  # 正确答案计数

        # 决定是否禁用进度条（分布式训练时只在主进程显示）
        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        
        # 遍历数据集中的每个样本
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            # 运行推理器生成答案
            # input_processor: 将原始示例转换为推理器需要的格式
            # sample_prompt: 生成 few-shot 提示词
            algo_output = reasoner(self.input_processor(example),
                                    prompt=self.sample_prompt(
                                        shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot))
            
            # 从推理结果中提取输出和正确答案
            output = self.output_extractor(algo_output)  # 从 AlgorithmOutput 中提取答案
            answer = self.answer_extractor(example)       # 从数据集中提取正确答案
            
            # 评估输出是否正确
            correct = self.eval_output(answer, output)
            correct_count += correct
            
            # 计算当前准确率（累积准确率）
            accuracy = correct_count / (i + 1)
            
            # 生成日志字符串
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            # 保存日志和结果（只在主进程中，且启用日志时）
            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                # 追加日志到文件
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
            
                # 保存算法输出（pickle 格式，可用于后续分析和可视化）
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                    pickle.dump(algo_output, f)
        
        return accuracy

    def evaluate_sc(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 n_sc = 10,
                 log_dir=None):
        """在数据集上评估推理器（使用 Self-Consistency）
        
        这个方法使用 Self-Consistency 策略：对每个问题生成多个答案，
        然后选择出现次数最多的答案作为最终答案。
        这可以提高推理的准确性和鲁棒性。
        
        Args:
            reasoner: 要评估的推理器
            shuffle_prompt: 是否打乱提示词中的示例顺序
            num_shot: few-shot 的示例数量
            resume: 从哪个样本开始（用于断点续跑）
            n_sc: Self-Consistency 的采样数量（生成多少个答案）
            log_dir: 日志目录（如果为 None 会自动生成）
            
        Returns:
            准确率（0-1 之间的浮点数）
            
        注意:
            Self-Consistency 策略：
            1. 对每个问题生成 n_sc 个不同的答案
            2. 使用投票机制选择最常见的答案
            3. 这通常比单次生成更准确
        """

        # 从指定位置开始的数据集（支持断点续跑）
        self.dataset = list(self.full_dataset)[resume:]
        
        # 获取搜索算法的名称（用于日志目录命名）
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        # 只在主进程中创建日志目录（分布式训练时）
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # 如果没有指定日志目录，自动生成一个（包含数据集名、算法名和时间戳）
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            # 保存命令行参数（用于复现实验）
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0  # 正确答案计数

        # 决定是否禁用进度条（分布式训练时只在主进程显示）
        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        
        # 遍历数据集中的每个样本
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            # 生成 few-shot 提示词（所有 Self-Consistency 采样使用同一个提示词）
            prompt = self.sample_prompt(
                            shuffle_prompt=shuffle_prompt,
                            num_shot=num_shot)
            
            output_list = []  # 存储所有生成的答案
            save_list = []    # 存储所有推理路径（用于分析）
            
            # Self-Consistency: 对每个问题生成 n_sc 个不同的答案
            for j in range(n_sc):
                # 运行推理器生成答案
                algo_output = reasoner(self.input_processor(example),
                                    prompt=prompt)
                
                # 提取终止状态（用于保存推理路径）
                terminal_state = algo_output.terminal_state
                path = ""
                # 构建推理路径字符串（子问题和子答案的序列）
                for k in range(len(terminal_state)):
                    path += terminal_state[k].sub_question + " " + terminal_state[k].sub_answer + " "
                save_list.append(path)
                
                # 从推理结果中提取输出
                output = self.output_extractor(algo_output)
                output_list.append(output)
                
                # 从数据集中提取正确答案（只需要提取一次）
                answer = self.answer_extractor(example)
            
            # Self-Consistency 投票：选择出现次数最多的答案
            from collections import Counter
            output = Counter(output_list).most_common(1)[0][0]
            
            # 评估最终答案是否正确
            correct = self.eval_output(answer, output)
            correct_count += correct
            
            # 计算当前准确率（累积准确率）
            accuracy = correct_count / (i + 1)
            
            # 生成日志字符串
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            # 保存日志和结果（只在主进程中，且启用日志时）
            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                # 追加日志到文件
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
                
                # 保存所有推理路径（用于分析 Self-Consistency 的效果）
                with open(os.path.join(log_dir, 'algo_output.txt'),'a') as f1:
                    print(save_list, file=f1)
        
        return accuracy
    @abstractmethod
    def eval_output(self, answer, output):
        """评估输出是否正确
        
        这是评估器的核心方法，用于判断推理器的输出是否与正确答案匹配。
        不同的数据集可能有不同的评估标准（如精确匹配、部分匹配等）。
        
        Args:
            answer: 正确答案（从数据集中提取）
            output: 推理器的输出（从 AlgorithmOutput 中提取）
            
        Returns:
            bool: 如果输出正确返回 True，否则返回 False
            
        注意:
            子类必须实现这个方法，定义具体的评估标准。
        """
        pass

class Tool():
    """工具类：封装可调用的函数
    
    用于在推理过程中调用外部工具（如计算器、搜索引擎等）
    """
    
    def __init__(self, func, name, description):
        """初始化工具
        
        Args:
            func: 要封装的函数
            name: 工具名称
            description: 工具描述（用于提示 LLM 何时使用这个工具）
        """
        self.func = func
        self.name = name
        self.description = description

    def __call__(self, **kwargs):
        """调用工具函数"""
        return self.func(**kwargs)