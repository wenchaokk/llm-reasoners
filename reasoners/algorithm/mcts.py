"""
MCTS (蒙特卡洛树搜索) 算法实现

MCTS 是最强大的搜索算法之一，通过四个阶段的迭代来探索搜索空间：
1. Selection（选择）：使用 UCT 公式从根节点选择路径到叶子节点
2. Expansion（扩展）：扩展选中的叶子节点，生成所有可能的动作
3. Simulation（模拟）：从扩展节点开始快速模拟到终止状态
4. Backpropagation（回传）：将模拟得到的奖励回传到路径上的所有节点

MCTS 的优势：
- 平衡探索和利用（通过 UCT 公式）
- 不需要完整的领域知识
- 可以处理大规模搜索空间
"""

import pickle
from os import PathLike
import pickle
from os import PathLike
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange

from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


class MCTSNode(Generic[State, Action, Example]):
    """MCTS 搜索树中的节点
    
    每个节点代表搜索树中的一个状态，包含：
    - 状态信息
    - 动作信息（从父节点到此节点的动作）
    - 奖励信息（fast_reward 和 reward）
    - 访问历史（cum_rewards）
    - 子节点列表
    
    Attributes:
        id: 节点唯一标识符
        state: 当前状态（None 表示未扩展的节点）
        action: 从父节点到此节点的动作
        parent: 父节点（None 表示根节点）
        children: 子节点列表（None 表示未扩展）
        fast_reward: 快速评估的奖励（在扩展前计算）
        reward: 完整评估的奖励（在扩展后计算）
        cum_rewards: 累积奖励历史（用于计算 Q 值）
        is_terminal: 是否为终止状态
        depth: 节点深度（根节点为 0）
    """
    id_iter = itertools.count()  # 节点 ID 生成器

    @classmethod
    def reset_id(cls):
        """重置节点 ID 生成器（用于新的搜索）"""
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[MCTSNode]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.mean):
        """初始化 MCTS 节点
        
        Args:
            state: 当前状态（None 表示节点尚未扩展，状态未知）
            action: 从父节点到此节点的动作（None 表示根节点）
            parent: 父节点（None 表示根节点）
            fast_reward: 快速评估的奖励（在扩展前计算，用于 UCT 选择）
            fast_reward_details: 快速奖励的详细信息（会传递给 reward 函数）
            is_terminal: 是否为终止状态
            calc_q: Q 值计算方法（默认：平均值）
                    用于从累积奖励历史计算 Q 值
        """
        self.id = next(MCTSNode.id_iter)  # 分配唯一 ID
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []  # 累积奖励历史（每次模拟后添加）
        self.fast_reward = self.reward = fast_reward  # 初始时两者相同
        self.fast_reward_details = fast_reward_details  # 快速奖励的详细信息
        self.is_terminal = is_terminal  # 是否为终止状态
        self.action = action  # 从父节点到此节点的动作
        self.state = state  # 当前状态
        self.parent = parent  # 父节点
        self.children: 'Optional[list[MCTSNode]]' = None  # 子节点列表（None 表示未扩展）
        self.calc_q = calc_q  # Q 值计算方法
        if parent is None:
            self.depth = 0  # 根节点深度为 0
        else:
            self.depth = parent.depth + 1  # 子节点深度 = 父节点深度 + 1

    @property
    def Q(self) -> float:
        """计算节点的 Q 值（动作价值）
        
        Q 值表示从父节点执行动作到达此节点的平均奖励。
        用于 UCT 公式中的利用部分。
        
        Returns:
            float: Q 值
                - 如果节点未扩展（state is None），返回 fast_reward
                - 否则，从累积奖励历史计算（默认使用平均值）
        """
        if self.state is None:
            # 节点未扩展，使用快速奖励作为 Q 值的估计
            return self.fast_reward
        else:
            # 节点已扩展，从累积奖励历史计算 Q 值
            return self.calc_q(self.cum_rewards)


class MCTSResult(NamedTuple):
    """MCTS 算法的输出结果
    
    Attributes:
        terminal_state: 终止状态（找到的最终状态）
        cum_reward: 累积奖励（最优路径的总奖励）
        trace: 轨迹（状态序列和动作序列的元组）
        trace_of_nodes: 轨迹中的节点列表（用于可视化）
        tree_state: 搜索树的根节点（包含完整的搜索树）
        trace_in_each_iter: 每次迭代选择的轨迹（用于分析）
        tree_state_after_each_iter: 每次迭代后的树状态（用于可视化）
        aggregated_result: 聚合结果（如果使用了 MCTSAggregation）
    """
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTSAggregation(Generic[State, Action, Example], ABC):
    """MCTS 结果聚合器
    
    这个类用于从 MCTS 搜索树中聚合多个终止节点的答案。
    通过加权投票机制选择最可能的答案。
    
    使用场景：
    - 当搜索树中有多个终止节点时
    - 需要从多个候选答案中选择最可靠的答案
    """
    
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        """初始化聚合器
        
        Args:
            retrieve_answer: 从状态中提取答案的函数
            weight_policy: 权重策略
                - 'edge': 使用边的奖励作为权重
                - 'edge_inverse_depth': 使用边的奖励除以深度作为权重
                - 'uniform': 均匀权重（每个答案一票）
        """
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode[State, Action,Example]) -> Optional[Hashable]:
        """从搜索树中聚合答案
        
        遍历搜索树，收集所有终止节点的答案，并使用加权投票选择最可能的答案。
        
        Args:
            tree_state: 搜索树的根节点
            
        Returns:
            聚合后的答案（出现次数最多或权重最高的答案），如果没有找到答案返回 None
        """
        answer_dict = defaultdict(lambda: 0)  # 答案到权重的映射

        def visit(cur: MCTSNode[State, Action, Example]):
            """递归访问节点，收集答案和权重"""
            if cur.state is None:
                return []
            
            # 如果是终止节点，提取答案并计算权重
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                
                # 根据权重策略计算权重
                if self.weight_policy == 'edge':
                    # 使用边的奖励作为权重
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    # 使用边的奖励除以深度作为权重（深度越浅权重越大）
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    # 均匀权重（每个答案一票）
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            
            # 非终止节点：递归访问子节点
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                # 收集每个答案的深度信息
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            
            # 为每个答案累加当前节点的权重
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        # 从根节点开始遍历
        visit(tree_state)

        # 选择权重最高的答案
        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])

class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    """MCTS (蒙特卡洛树搜索) 算法实现
    
    MCTS 通过四个阶段的迭代来探索搜索空间：
    1. Selection（选择）：使用 UCT 公式从根节点选择路径到叶子节点
    2. Expansion（扩展）：扩展选中的叶子节点，生成所有可能的动作
    3. Simulation（模拟）：从扩展节点开始快速模拟到终止状态
    4. Backpropagation（回传）：将模拟得到的奖励回传到路径上的所有节点
    
    UCT 公式：
        UCT = Q + w_exp * sqrt(ln(N) / n)
        其中：
        - Q: 动作价值（平均奖励）
        - w_exp: 探索权重
        - N: 父节点的访问次数
        - n: 当前节点的访问次数
    """
    
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__):
        """初始化 MCTS 算法
        
        Args:
            output_trace_in_each_iter: 是否在每次迭代中输出轨迹
                                      如果为 True，会保存每次迭代选择的轨迹（深拷贝）
                                      还会输出每次迭代后的树状态（用于可视化）
            w_exp: UCT 公式中的探索权重
                   - 较大的值：更偏向探索（尝试新路径）
                   - 较小的值：更偏向利用（选择已知好的路径）
            depth_limit: 最大搜索深度（防止无限搜索）
            n_iters: 迭代次数（每次迭代执行一次完整的 MCTS 循环）
            cum_reward: 累积奖励计算方法（默认：sum，即求和）
            calc_q: Q 值计算方法（默认：np.mean，即平均值）
            simulate_strategy: 模拟策略
                - 'max': 选择 fast_reward 最大的子节点（贪心）
                - 'sample': 根据 fast_reward 采样（概率与奖励成正比）
                - 'random': 随机选择
                - 自定义函数：接受 fast_rewards 列表，返回选择的索引
            output_strategy: 输出策略（如何从搜索树中选择最终路径）
                - 'max_reward': 在最终树中 DFS 找到累积奖励最大的路径
                - 'follow_max': 从根节点开始，每一步选择奖励最大的子节点
                - 'max_visit': 访问次数最多的终止节点
                - 'max_iter': 所有迭代中累积奖励最大的路径
                - 'last_iter': 最后一次迭代的路径
                - 'last_terminal_iter': 最后一次迭代中到达终止状态的路径
            uct_with_fast_reward: 如果为 True，在 UCT 中对未访问的子节点使用 fast_reward
                                 否则，优先访问 fast_reward 最大的未访问子节点
            aggregator: 可选的答案聚合器（用于从多个终止节点聚合答案）
            disable_tqdm: 是否禁用进度条
            node_visualizer: 节点可视化函数（用于调试和可视化）
        """
        super().__init__()
        self.world_model = None  # 世界模型（在 __call__ 中设置）
        self.search_config = None  # 搜索配置（在 __call__ 中设置）
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp  # 探索权重
        self.depth_limit = depth_limit  # 最大搜索深度
        self.n_iters = n_iters  # 迭代次数
        self.cum_reward = cum_reward  # 累积奖励计算方法
        self.calc_q = calc_q  # Q 值计算方法
        
        # 定义默认的模拟策略
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),  # 选择 fast_reward 最大的
            'sample': lambda x: np.random.choice(len(x), p=x),  # 根据 fast_reward 采样
            'random': lambda x: np.random.choice(len(x)),  # 随机选择
        }
        # 获取模拟策略函数（如果提供了自定义函数则使用自定义函数）
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy  # 输出策略
        self.uct_with_fast_reward = uct_with_fast_reward  # 是否在 UCT 中使用 fast_reward
        self._output_iter: list[MCTSNode] = None  # 输出的轨迹（节点列表）
        self._output_cum_reward = -math.inf  # 输出的累积奖励
        self.trace_in_each_iter: list[list[MCTSNode]] = None  # 每次迭代的轨迹
        self.root: Optional[MCTSNode] = None  # 搜索树的根节点
        self.disable_tqdm = disable_tqdm  # 是否禁用进度条
        self.node_visualizer = node_visualizer  # 节点可视化函数
        self.aggregator = aggregator  # 答案聚合器

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        """执行一次 MCTS 迭代
        
        这是 MCTS 的核心方法，执行一次完整的 MCTS 循环：
        1. Selection: 选择路径
        2. Expansion: 扩展节点（如果需要）
        3. Simulation: 模拟到终止
        4. Backpropagation: 回传奖励
        
        Args:
            node: 搜索树的根节点
            
        Returns:
            本次迭代选择的路径（节点列表）
        """
        # 1. Selection: 使用 UCT 公式选择路径到叶子节点
        path = self._select(node)
        
        # 2. Expansion: 如果选中的节点不是终止节点且未达到深度限制，则扩展
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])  # 扩展节点，生成子节点
            self._simulate(path)     # 从扩展节点开始模拟
        
        # 3. Backpropagation: 将奖励回传到路径上的所有节点
        cum_reward = self._back_propagate(path)
        
        # 根据输出策略更新输出轨迹
        # 'max_iter': 记录累积奖励最大的迭代
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        # 'last_iter': 记录最后一次迭代
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        # 'last_terminal_iter': 记录最后一次到达终止状态的迭代
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        """判断节点是否为终止节点（考虑深度限制）
        
        Args:
            node: 要判断的节点
            
        Returns:
            bool: 如果为终止状态或达到深度限制返回 True
        """
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """Selection 阶段：从根节点选择路径到叶子节点
        
        使用 UCT 公式从根节点开始，每一步选择 UCT 值最大的子节点，
        直到到达叶子节点（未扩展的节点）或终止节点。
        
        Args:
            node: 搜索树的根节点
            
        Returns:
            从根节点到叶子节点的路径（节点列表）
        """
        path = []
        while True:
            path.append(node)
            # 如果节点未扩展、没有子节点或为终止节点，返回路径
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            # 使用 UCT 公式选择下一个节点
            node = self._uct_select(node)

    def _uct(self, node: MCTSNode) -> float:
        """计算节点的 UCT 值
        
        UCT (Upper Confidence Bound for Trees) 公式：
            UCT = Q + w_exp * sqrt(ln(N) / n)
        
        其中：
            - Q: 动作价值（平均奖励）- 利用部分
            - w_exp: 探索权重
            - N: 父节点的访问次数
            - n: 当前节点的访问次数
            - sqrt(ln(N) / n): 探索项（未访问的节点有更大的探索值）
        
        UCT 平衡了探索和利用：
            - 利用：选择已知好的路径（Q 值大）
            - 探索：尝试访问次数少的路径（探索项大）
        
        Args:
            node: 要计算 UCT 值的节点
            
        Returns:
            float: UCT 值（值越大，越应该被选择）
        """
        # 父节点的访问次数
        parent_visits = len(node.parent.cum_rewards)
        # 当前节点的访问次数
        node_visits = max(1, len(node.cum_rewards))
        
        # UCT 公式：Q 值（利用） + 探索项
        return node.Q + self.w_exp * np.sqrt(np.log(parent_visits) / node_visits)

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """使用 UCT 公式选择子节点
        
        如果启用了 uct_with_fast_reward 或所有子节点都已访问，
        使用 UCT 公式选择。否则，优先访问 fast_reward 最大的未访问子节点。
        
        Args:
            node: 当前节点
            
        Returns:
            选择的子节点
        """
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            # 使用 UCT 公式选择（所有子节点都已访问，或使用 fast_reward 计算 UCT）
            return max(node.children, key=self._uct)
        else:
            # 优先访问 fast_reward 最大的未访问子节点
            # 这样可以快速探索有希望的区域
            unvisited_children = filter(lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.fast_reward)

    def _expand(self, node: MCTSNode):
        """Expansion 阶段：扩展节点，生成子节点
        
        如果节点未扩展，先执行状态转换和奖励计算。
        然后为每个可能的动作创建子节点。
        
        工作流程：
        1. 如果节点未扩展，执行状态转换
        2. 计算完整奖励（结合 fast_reward 和状态信息）
        3. 判断是否为终止状态
        4. 如果不是终止状态，为每个动作创建子节点
        
        Args:
            node: 要扩展的节点
        """
        # 如果节点未扩展，先执行状态转换
        if node.state is None:
            # 执行动作，转换到新状态
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            
            # 计算完整奖励
            # 注意：奖励在状态更新后计算，这样可以从世界模型传递信息（通过 **aux）
            # 到奖励函数，避免重复计算
            node.reward, node.reward_details = self.search_config. \
                reward(node.parent.state, node.action, **node.fast_reward_details, **aux)
            
            # 判断是否为终止状态
            node.is_terminal = self.world_model.is_terminal(node.state)

        # 如果是终止状态，不需要扩展（没有子节点）
        if node.is_terminal:
            return

        # 获取所有可能的动作
        children = []
        actions = self.search_config.get_actions(node.state)
        
        # 为每个动作创建子节点
        for action in actions:
            # 计算快速奖励（用于 UCT 选择）
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            
            # 创建子节点（state=None 表示未扩展，将在后续迭代中扩展）
            child = MCTSNode(
                state=None,  # 未扩展，状态未知
                action=action,  # 从当前节点到此子节点的动作
                parent=node,  # 父节点
                fast_reward=fast_reward,  # 快速奖励
                fast_reward_details=fast_reward_details,  # 快速奖励的详细信息
                calc_q=self.calc_q  # Q 值计算方法
            )
            children.append(child)

        # 设置子节点列表
        node.children = children

    def _simulate(self, path: list[MCTSNode]):
        """Simulation 阶段：从扩展节点快速模拟到终止状态
        
        这是 MCTS 的"快速模拟"阶段，用于快速评估一个路径的价值。
        它不进行完整的搜索，而是根据 fast_reward 快速选择动作，
        直到到达终止状态或达到深度限制。
        
        模拟策略：
            - 'max': 选择 fast_reward 最大的动作（贪心）
            - 'sample': 根据 fast_reward 采样（概率与奖励成正比）
            - 'random': 随机选择
        
        Args:
            path: 当前路径（会被修改，添加模拟的节点）
        """
        node = path[-1]  # 从路径的最后一个节点开始模拟
        while True:
            # 如果节点未扩展，先扩展它
            if node.state is None:
                self._expand(node)
            
            # 如果到达终止状态或深度限制，或没有子节点，停止模拟
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            
            # 根据模拟策略选择下一个动作
            fast_rewards = [child.fast_reward for child in node.children]
            chosen_index = self.simulate_choice(fast_rewards)  # 选择子节点索引
            node = node.children[chosen_index]  # 移动到选中的子节点
            path.append(node)  # 将新节点添加到路径

    def _back_propagate(self, path: list[MCTSNode]):
        """Backpropagation 阶段：将奖励回传到路径上的所有节点
        
        从路径的最后一个节点开始，向前回传奖励。
        每个节点都会记录累积奖励，用于后续的 UCT 计算。
        
        工作流程：
        1. 从路径末尾开始，向前遍历
        2. 收集每个节点的奖励
        3. 计算累积奖励（从当前节点到路径末尾）
        4. 将累积奖励添加到节点的 cum_rewards 列表
        
        Args:
            path: 要回传奖励的路径（节点列表）
            
        Returns:
            float: 路径的累积奖励（从根节点到终止节点）
        """
        rewards = []  # 收集路径上所有节点的奖励
        cum_reward = -math.inf
        
        # 从路径末尾向前遍历（回传）
        for node in reversed(path):
            rewards.append(node.reward)  # 收集奖励
            # 计算累积奖励（从当前节点到路径末尾）
            # rewards[::-1] 反转列表，使第一个元素是当前节点的奖励
            cum_reward = self.cum_reward(rewards[::-1])
            # 将累积奖励添加到节点的历史记录中
            # 这个历史记录用于计算 Q 值（平均奖励）
            node.cum_rewards.append(cum_reward)
        
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        """使用深度优先搜索找到累积奖励最大的路径
        
        这个方法用于 'max_reward' 输出策略。
        它在完整的搜索树中 DFS，找到累积奖励最大的终止路径。
        
        Args:
            path: 当前路径（节点列表）
            
        Returns:
            (累积奖励, 路径) 元组
            如果没有找到终止路径，返回 (-inf, path)
        """
        cur = path[-1]  # 当前节点
        
        # 如果到达终止状态，计算路径的累积奖励
        if cur.is_terminal:
            # 计算从根节点（path[1:] 跳过根节点）到终止节点的累积奖励
            return self.cum_reward([node.reward for node in path[1:]]), path
        
        # 如果节点未扩展，无法继续搜索
        if cur.children is None:
            return -math.inf, path
        
        # 只考虑已访问的子节点（已扩展的节点）
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        
        # 递归搜索所有子节点，选择累积奖励最大的路径
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def search(self):
        """执行 MCTS 搜索
        
        这是 MCTS 算法的主循环，执行 n_iters 次迭代。
        每次迭代执行一次完整的 MCTS 循环（Selection、Expansion、Simulation、Backpropagation）。
        
        工作流程：
        1. 初始化根节点
        2. 执行 n_iters 次迭代
        3. 根据输出策略选择最终路径
        """
        # 初始化输出变量
        self._output_cum_reward = -math.inf
        self._output_iter = None
        
        # 创建根节点（初始状态）
        self.root = MCTSNode(
            state=self.world_model.init_state(),  # 初始状态
            action=None,  # 根节点没有动作
            parent=None,  # 根节点没有父节点
            calc_q=self.calc_q
        )
        
        # 如果需要在每次迭代中输出轨迹，初始化列表
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        # 执行 n_iters 次 MCTS 迭代
        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            # 执行一次完整的 MCTS 循环
            path = self.iterate(self.root)
            
            # 如果需要，保存本次迭代的轨迹（深拷贝，避免后续修改影响）
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        # 根据输出策略选择最终路径
        # 'follow_max': 从根节点开始，每一步选择奖励最大的子节点
        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                # 只考虑已访问的子节点
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                # 选择奖励最大的子节点
                cur = max(visited_children, key=lambda x: x.reward)
            # 计算累积奖励（注意：路径是反向的，需要反转）
            self._output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])
        
        # 'max_reward': 在完整树中 DFS 找到累积奖励最大的路径
        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            # 如果没有找到终止路径，设置为 None
            if self._output_cum_reward == -math.inf:
                self._output_iter = None

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 log_file: Optional[str] = None,
                 **kwargs) -> MCTSResult:
        """执行 MCTS 搜索算法
        
        这是 SearchAlgorithm 接口的实现，是 MCTS 算法的主要入口。
        
        Args:
            world_model: 世界模型（定义状态转换）
            search_config: 搜索配置（定义动作和奖励）
            log_file: 可选的日志文件路径（当前未使用）
            **kwargs: 其他参数
            
        Returns:
            MCTSResult: 包含搜索结果的命名元组
        """
        # 重置节点 ID 生成器（开始新的搜索）
        MCTSNode.reset_id()
        
        # 设置世界模型和搜索配置
        self.world_model = world_model
        self.search_config = search_config

        # 执行搜索
        self.search()

        # 构建结果
        if self._output_iter is None:
            # 如果没有找到有效路径（如所有路径都未到达终止状态）
            terminal_state = trace = None
        else:
            # 提取终止状态和轨迹
            terminal_state = self._output_iter[-1].state  # 路径的最后一个节点状态
            # 构建轨迹：状态序列和动作序列
            trace = (
                [node.state for node in self._output_iter],  # 状态序列
                [node.action for node in self._output_iter[1:]]  # 动作序列（跳过根节点）
            )
        
        # 处理每次迭代的轨迹（如果启用）
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        
        # 创建结果对象
        result = MCTSResult(
            terminal_state=terminal_state,  # 终止状态
            cum_reward=self._output_cum_reward,  # 累积奖励
            trace=trace,  # 轨迹
            trace_of_nodes=self._output_iter,  # 轨迹中的节点列表
            tree_state=self.root,  # 搜索树的根节点（包含完整树）
            trace_in_each_iter=trace_in_each_iter,  # 每次迭代的轨迹
            tree_state_after_each_iter=tree_state_after_each_iter  # 每次迭代后的树状态
        )
        
        # 如果提供了聚合器，使用它聚合答案
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),  # 聚合后的答案
            )
        return result
