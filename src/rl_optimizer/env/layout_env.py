# src/rl_optimizer/env/layout_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from sklearn.cluster import DBSCAN

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.utils.setup import setup_logger
from src.algorithms.constraint_manager import ConstraintManager

logger = setup_logger(__name__)  # 使用默认INFO级别

class LayoutEnv(gym.Env):
    """
    医院布局优化的强化学习环境 (范式A: 选科室，填槽位)。

    遵循Gymnasium接口，通过自回归方式构建布局。环境按照一个在每个回合
    开始时随机打乱的槽位顺序进行填充。在每一步，智能体从所有尚未被
    放置的科室中选择一个，放入当前待填充的槽位。

    - **状态 (Observation)**: 字典，包含当前部分布局、已放置科室的掩码、
                              以及当前待填充槽位的面积信息。
    - **动作 (Action)**: 离散值，代表在 `placeable_depts` 列表中的科室索引。
    - **奖励 (Reward)**: 仅在回合结束时给予的稀疏奖励，基于加权总通行时间
                      和软约束。
    - **约束 (Constraints)**: 硬约束（面积、强制相邻）通过动作掩码实现，
                          确保智能体只能选择合法的科室。
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: RLConfig, cache_manager: CacheManager, cost_calculator: CostCalculator, constraint_manager: ConstraintManager):
        """
        初始化布局优化环境。

        Args:
            config (RLConfig): RL优化器的配置对象。
            cache_manager (CacheManager): 已初始化的数据缓存管理器。
            cost_calculator (CostCalculator): 已初始化的成本计算器。
            constraint_manager (ConstraintManager): 约束管理器。
        """
        super().__init__()
        self.config = config
        self.cm = cache_manager
        self.cc = cost_calculator
        self.constraint_manager = constraint_manager

        self._initialize_nodes_and_slots()
        self._define_spaces()
        self._initialize_state_variables()
        
        # 初始化相邻性奖励相关组件
        if self.config.ENABLE_ADJACENCY_REWARD:
            self._initialize_adjacency_components()

        logger.info(f"环境初始化完成：{self.num_slots}个可用槽位，{self.num_depts}个待放置科室。")

    def _initialize_nodes_and_slots(self):
        """从CacheManager获取并设置节点和槽位信息。"""
        self.placeable_slots = self.cm.placeable_slots
        self.slot_areas = self.cm.placeable_nodes_df['area'].values
        self.num_slots = len(self.placeable_slots)

        self.placeable_depts = self.cm.placeable_departments
        self.dept_areas_map = dict(zip(self.cm.placeable_nodes_df['node_id'], self.cm.placeable_nodes_df['area']))
        self.num_depts = len(self.placeable_depts)
        
        if self.num_slots != self.num_depts:
            raise ValueError(f"槽位数 ({self.num_slots}) 与待布局科室数 ({self.num_depts}) 不匹配!")
        
        self.dept_to_idx = {dept: i for i, dept in enumerate(self.placeable_depts)}
        
    def _initialize_adjacency_components(self):
        """
        初始化相邻性奖励计算所需的组件和预计算矩阵。
        """
        logger.info("初始化相邻性奖励组件...")
        
        # 获取行程时间矩阵（已过滤掉面积行）
        self.travel_times_matrix = self.cm.travel_times_matrix.copy()
        
        # 预计算空间相邻性矩阵
        if self.config.ADJACENCY_PRECOMPUTE:
            self._precompute_spatial_adjacency()
            
        # 初始化功能相邻性映射
        self._initialize_functional_adjacency()
        
        # 初始化连通性相邻性（如果需要）
        if self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0:
            self._precompute_connectivity_adjacency()
            
        logger.info("相邻性奖励组件初始化完成")

    def _precompute_spatial_adjacency(self):
        """
        预计算基于分位数的空间相邻性矩阵。
        使用分位数阈值避免硬编码距离值。
        """
        logger.debug("预计算空间相邻性矩阵...")
        
        # 获取所有有效的节点名称（排除None值）
        valid_nodes = [node for node in self.placeable_depts if node in self.travel_times_matrix.columns]
        n_nodes = len(valid_nodes)
        
        if n_nodes < 2:
            logger.warning("可放置节点数量过少，跳过空间相邻性预计算")
            self.spatial_adjacency_matrix = np.zeros((n_nodes, n_nodes))
            return
        
        # 构建距离矩阵
        distance_matrix = np.zeros((n_nodes, n_nodes))
        for i, node1 in enumerate(valid_nodes):
            for j, node2 in enumerate(valid_nodes):
                if i != j and node1 in self.travel_times_matrix.index and node2 in self.travel_times_matrix.columns:
                    distance_matrix[i, j] = self.travel_times_matrix.loc[node1, node2]
        
        # 计算距离的分位数阈值
        upper_triangle_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        valid_distances = upper_triangle_distances[upper_triangle_distances > 0]
        
        if len(valid_distances) == 0:
            logger.warning("无有效距离数据，使用默认空间相邻性矩阵")
            self.spatial_adjacency_matrix = np.eye(n_nodes)
            return
        
        threshold = np.percentile(valid_distances, self.config.ADJACENCY_PERCENTILE_THRESHOLD * 100)
        logger.debug(f"空间相邻性距离阈值（{self.config.ADJACENCY_PERCENTILE_THRESHOLD*100}分位数）: {threshold:.2f}")
        
        # 生成空间相邻性矩阵
        self.spatial_adjacency_matrix = (distance_matrix <= threshold).astype(float)
        np.fill_diagonal(self.spatial_adjacency_matrix, 0)  # 自身不相邻
        
        # 确保矩阵对称
        self.spatial_adjacency_matrix = (self.spatial_adjacency_matrix + self.spatial_adjacency_matrix.T) / 2
        
        adjacency_ratio = np.sum(self.spatial_adjacency_matrix) / (n_nodes * (n_nodes - 1))
        logger.debug(f"空间相邻性矩阵生成完成，相邻比例: {adjacency_ratio:.3f}")

    def _initialize_functional_adjacency(self):
        """
        初始化基于医疗功能的相邻性偏好映射。
        """
        logger.debug("初始化功能相邻性映射...")
        
        # 创建通用名称到节点的映射
        self.generic_to_nodes = {}
        for dept in self.placeable_depts:
            generic_name = dept.split('_')[0]
            if generic_name not in self.generic_to_nodes:
                self.generic_to_nodes[generic_name] = []
            self.generic_to_nodes[generic_name].append(dept)
        
        # 创建功能相邻性偏好矩阵
        n_nodes = len(self.placeable_depts)
        self.functional_adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for i, dept1 in enumerate(self.placeable_depts):
            generic1 = dept1.split('_')[0]
            for j, dept2 in enumerate(self.placeable_depts):
                generic2 = dept2.split('_')[0]
                
                if i != j:
                    # 查找医疗功能相邻性偏好
                    preference_score = self._get_functional_preference(generic1, generic2)
                    self.functional_adjacency_matrix[i, j] = preference_score
        
        logger.debug(f"功能相邻性矩阵生成完成，形状: {self.functional_adjacency_matrix.shape}")

    def _get_functional_preference(self, generic1: str, generic2: str) -> float:
        """
        获取两个通用科室之间的功能相邻性偏好分数。
        
        Args:
            generic1: 第一个科室的通用名称
            generic2: 第二个科室的通用名称
            
        Returns:
            float: 偏好分数，正数表示偏好相邻，负数表示偏好分离
        """
        preferences = self.config.MEDICAL_ADJACENCY_PREFERENCES
        
        # 正向偏好
        if generic1 in preferences and generic2 in preferences[generic1]:
            return preferences[generic1][generic2]
        
        # 反向偏好
        if generic2 in preferences and generic1 in preferences[generic2]:
            return preferences[generic2][generic1]
        
        # 默认无偏好
        return 0.0

    def _precompute_connectivity_adjacency(self):
        """
        预计算基于图连通性的相邻性矩阵。
        考虑多跳路径的连通性。
        """
        logger.debug("预计算连通性相邻性矩阵...")
        
        n_nodes = len(self.placeable_depts)
        self.connectivity_adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        # 获取距离矩阵
        if not hasattr(self, 'spatial_adjacency_matrix'):
            logger.warning("空间相邻性矩阵未初始化，跳过连通性相邻性计算")
            return
        
        # 计算多跳路径的连通性
        distance_matrix = np.zeros((n_nodes, n_nodes))
        for i, node1 in enumerate(self.placeable_depts):
            for j, node2 in enumerate(self.placeable_depts):
                if i != j and node1 in self.travel_times_matrix.index and node2 in self.travel_times_matrix.columns:
                    distance_matrix[i, j] = self.travel_times_matrix.loc[node1, node2]
        
        # 基于距离的多跳连通性
        valid_distances = distance_matrix[distance_matrix > 0]
        if len(valid_distances) > 0:
            connectivity_threshold = np.percentile(valid_distances, self.config.CONNECTIVITY_DISTANCE_PERCENTILE * 100)
            
            # 计算连通性权重（距离越近权重越高）
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and distance_matrix[i, j] > 0:
                        if distance_matrix[i, j] <= connectivity_threshold:
                            # 使用指数衰减函数计算连通性权重
                            weight = np.exp(-distance_matrix[i, j] / connectivity_threshold)
                            self.connectivity_adjacency_matrix[i, j] = weight
        
        logger.debug(f"连通性相邻性矩阵生成完成，非零元素数: {np.count_nonzero(self.connectivity_adjacency_matrix)}")

    def _define_spaces(self):
        """定义观测空间和动作空间。"""
        # 动作空间：选择一个科室进行放置，或者跳过当前槽位
        # 动作 0 到 num_depts-1：选择科室索引
        # 动作 num_depts：跳过当前槽位
        self.action_space = spaces.Discrete(self.num_depts + 1)
        self.SKIP_ACTION = self.num_depts  # 跳过动作的索引

        # 观测空间：使用Box空间来明确定义形状，避免SB3的意外转换
        self.observation_space = spaces.Dict({
            # layout[i] = k+1 表示槽位i放置了索引为k的科室。0表示空。
            "layout": spaces.Box(
                low=0, 
                high=self.num_depts, # 最大值为科室数 (num_depts-1)+1
                shape=(self.num_slots,), 
                dtype=np.int32
            ),
            # placed_mask[k] = 1 表示索引为k的科室已被放置。
            "placed_mask": spaces.MultiBinary(self.num_depts),
            # 当前待填充槽位的索引
            "current_slot_idx": spaces.Box(low=0, high=self.num_slots - 1, shape=(1,), dtype=np.int32),
            # 跳过的槽位数量
            "num_skipped_slots": spaces.Box(low=0, high=self.num_slots, shape=(1,), dtype=np.int32)
        })

    def _initialize_state_variables(self):
        """初始化每个回合都会改变的状态变量。"""
        self.current_step = 0
        # layout的索引是物理槽位索引，值是科室索引+1
        self.layout = np.zeros(self.num_slots, dtype=np.int32)
        self.placed_mask = np.zeros(self.num_depts, dtype=bool)
        # 每个回合开始时需要被打乱的槽位处理顺序
        self.shuffled_slot_indices = np.arange(self.num_slots)
        # 跟踪跳过的槽位
        self.skipped_slots = set()
        # 势函数相关状态
        self.previous_potential = 0.0  # 上一步的势函数值

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境，按面积从小到大排序槽位，并返回初始观测。"""
        super().reset(seed=seed)
        self._initialize_state_variables()
        # 修改：按槽位面积从小到大排序，而不是随机打乱
        self.shuffled_slot_indices = np.argsort(self.slot_areas)
        
        # 初始化势函数值（空布局的势函数为0）
        self.previous_potential = 0.0
        
        # 添加调试日志，显示槽位填充顺序
        logger.debug(f"槽位填充顺序（按面积从小到大）:")
        for i, slot_idx in enumerate(self.shuffled_slot_indices[:5]):  # 只显示前5个
            logger.debug(f"  {i+1}. {self.placeable_slots[slot_idx]} - 面积: {self.slot_areas[slot_idx]:.2f}")
        if len(self.shuffled_slot_indices) > 5:
            logger.debug(f"  ... 共 {len(self.shuffled_slot_indices)} 个槽位")
        
        return self._get_obs(), self._get_info(terminated=False)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一个动作：在当前槽位放置选定的科室，或者跳过当前槽位。"""
        # 更严格的输入验证
        if not (0 <= action <= self.num_depts):  # 包括跳过动作
            logger.error(f"无效的动作索引: {action}，有效范围是 [0, {self.num_depts}]")
            return self._get_obs(), self.config.INVALID_ACTION_PENALTY, True, False, self._get_info(terminated=False)
        
        # 如果启用势函数奖励，计算当前状态的势函数
        if self.config.ENABLE_POTENTIAL_REWARD:
            current_potential = self._calculate_potential()
        
        # 检查是否为跳过动作
        if action == self.SKIP_ACTION:
            # 跳过当前槽位
            current_slot_idx = self.shuffled_slot_indices[self.current_step]
            self.skipped_slots.add(current_slot_idx)
            logger.debug(f"跳过槽位 {self.placeable_slots[current_slot_idx]} (索引: {current_slot_idx})")
            immediate_reward = 0.0  # 跳过动作不给奖励
        else:
            # 检查科室是否已被放置
            if self.placed_mask[action]:
                # 这是一个安全检查。理论上动作掩码会阻止这种情况。
                # 如果发生，说明上游逻辑有误，应给予重罚并终止。
                logger.error(f"严重错误：智能体选择了已被放置的科室！动作={action}, 科室={self.placeable_depts[action]}")
                logger.error(f"当前步骤: {self.current_step}, placed_mask: {self.placed_mask}")
                logger.error(f"当前动作掩码: {self.get_action_mask()}")
                return self._get_obs(), self.config.INVALID_ACTION_PENALTY, True, False, self._get_info(terminated=False)

            # 确定当前要填充的物理槽位索引
            slot_to_fill = self.shuffled_slot_indices[self.current_step]
            
            # 更新状态：在布局中记录放置，并标记科室为"已用"
            self.layout[slot_to_fill] = action + 1  # 动作是科室索引，存储时+1
            self.placed_mask[action] = True
            logger.debug(f"在槽位 {self.placeable_slots[slot_to_fill]} 放置科室 {self.placeable_depts[action]}")
            
            # 成功放置一个科室，给予即时奖励
            immediate_reward = self.config.REWARD_PLACEMENT_BONUS
        
        # 更新步骤计数
        self.current_step += 1

        # 改进的终止条件：考虑所有科室都已放置或所有槽位都已处理
        all_departments_placed = all(self.placed_mask)
        all_slots_processed = (self.current_step >= self.num_slots)
        terminated = all_departments_placed or all_slots_processed
        
        # 计算奖励
        if self.config.ENABLE_POTENTIAL_REWARD:
            # 计算新状态的势函数
            new_potential = self._calculate_potential()
            # 势函数奖励 = γ * Φ(s') - Φ(s)
            potential_reward = self.config.GAMMA * new_potential - current_potential
            potential_reward *= self.config.POTENTIAL_REWARD_WEIGHT
            
            # 更新势函数值
            self.previous_potential = new_potential
            
            if terminated:
                # 终止时的奖励 = 即时奖励 + 势函数奖励 + 最终奖励
                self.cached_final_reward = self._calculate_final_reward()
                reward = immediate_reward + potential_reward + self.cached_final_reward
                # 只在回合结束时显示势函数奖励汇总（包括面积匹配信息）
                logger.info(f"势函数奖励汇总: 最终势={new_potential:.4f}, 势函数总奖励={potential_reward:.4f}")
                # 计算并显示面积匹配统计
                self._log_area_match_statistics()
            else:
                # 非终止时的奖励 = 即时奖励 + 势函数奖励
                reward = immediate_reward + potential_reward
                # 调试级别的日志，正常训练时不显示
                logger.debug(f"势函数奖励计算: 当前势={current_potential:.4f}, 新势={new_potential:.4f}, 势函数奖励={potential_reward:.4f}")
        else:
            # 原有奖励机制：即时奖励（成功放置）+ 最终奖励（终止时的总体评分）
            if terminated:
                self.cached_final_reward = self._calculate_final_reward()  # 缓存最终奖励，避免重复计算
                reward = immediate_reward + self.cached_final_reward
            else:
                reward = immediate_reward  # 只有即时奖励
        
        info = self._get_info(terminated)
        
        # 调试日志：检查episode结束时的info内容
        if terminated and logger.isEnabledFor(20):  # INFO级别
            logger.debug(f"Episode结束，info内容: {info}")
            if 'episode' in info:
                logger.debug(f"Episode数据: {info['episode']}")

        return self._get_obs(), reward, terminated, False, info
    
    def _get_obs(self) -> Dict[str, Any]:
        """构建并返回当前观测字典。"""
        if self.current_step >= self.num_slots:
            current_slot_idx = 0
        else:
            current_slot_idx = self.shuffled_slot_indices[self.current_step]

        return {
            "layout": self.layout,
            "placed_mask": self.placed_mask,
            "current_slot_idx": np.array([current_slot_idx], dtype=np.int32),
            "num_skipped_slots": np.array([len(self.skipped_slots)], dtype=np.int32)
        }
    
    def _get_info(self, terminated: bool = False) -> Dict[str, Any]:
        """返回包含动作掩码和episode信息的附加信息。"""
        info = {"action_mask": self.get_action_mask()}
        
        # 如果episode结束，添加训练指标信息
        if terminated:
            # 构造最终布局（包括空槽位）
            final_layout_depts = []
            placed_depts = []
            
            for slot_idx in range(self.num_slots):
                dept_id = self.layout[slot_idx]
                if dept_id > 0:
                    dept_name = self.placeable_depts[dept_id - 1]
                    final_layout_depts.append(dept_name)
                    placed_depts.append(dept_name)
                else:
                    final_layout_depts.append(None)
            
            num_empty_slots = len(self.skipped_slots)
            num_placed_depts = len(placed_depts)
            
            # 计算时间成本（只基于已放置的科室）
            if num_placed_depts > 0:
                raw_time_cost = self.cc.calculate_total_cost(placed_depts)
                time_reward = -raw_time_cost / self.config.REWARD_SCALE_FACTOR
            else:
                raw_time_cost = 0.0
                time_reward = -1000.0
            
            # 计算空槽位惩罚
            empty_penalty = -num_empty_slots * self.config.REWARD_EMPTY_SLOT_PENALTY
            
            # 调试日志：确认传递的是原始值
            if logger.isEnabledFor(20):  # INFO级别
                logger.debug(f"Episode结束 - 原始时间成本: {raw_time_cost:.2f}, "
                           f"放置科室数: {num_placed_depts}, 空槽位数: {num_empty_slots}")
            
            # Episode结束时添加详细信息
            info['episode'] = {
                'time_cost': raw_time_cost,  # 原始时间成本
                'time_reward': time_reward,  # 时间奖励部分
                'empty_penalty': empty_penalty,  # 空槽位惩罚
                'placement_bonus': num_placed_depts * self.config.REWARD_PLACEMENT_BONUS,  # 成功放置的累积奖励
                'final_reward': self.cached_final_reward if hasattr(self, 'cached_final_reward') else 0.0,  # 使用缓存的最终奖励
                'num_placed_depts': num_placed_depts,  # 放置的科室数
                'num_empty_slots': num_empty_slots,  # 空槽位数
                'layout': final_layout_depts.copy(),  # 完整布局（包括None）
                'placed_layout': placed_depts.copy(),  # 只包含已放置的科室
                'r': time_reward + empty_penalty + num_placed_depts * self.config.REWARD_PLACEMENT_BONUS,  # 总奖励
                'l': self.current_step  # episode长度
            }
            
            # 如果启用了详细日志，还可以添加更多信息
            if logger.isEnabledFor(20):  # INFO级别
                if num_placed_depts > 0:
                    info['episode']['per_process_costs'] = self.cc.calculate_per_process_cost(placed_depts)
            
            # 添加面积匹配统计信息
            area_match_stats = self._calculate_area_match_statistics()
            info['episode']['area_match_avg'] = area_match_stats['avg_match_score']
            info['episode']['area_match_min'] = area_match_stats['min_match_score']
            info['episode']['area_match_max'] = area_match_stats['max_match_score']
            
            # 添加相邻性奖励统计信息
            if self.config.ENABLE_ADJACENCY_REWARD and num_placed_depts >= 2:
                adjacency_stats = self._calculate_adjacency_statistics(placed_depts)
                info['episode']['adjacency_spatial'] = adjacency_stats['spatial_reward']
                info['episode']['adjacency_functional'] = adjacency_stats['functional_reward'] 
                info['episode']['adjacency_connectivity'] = adjacency_stats['connectivity_reward']
                info['episode']['adjacency_total'] = adjacency_stats['total_reward']
        
        return info
    
    def get_action_mask(self) -> np.ndarray:
        """
        计算当前步骤下所有合法的动作掩码。
        
        该方法根据当前待填充槽位的面积约束，返回一个布尔数组，指示哪些未放置的科室可被合法选择。
        如果没有合适的科室，则允许跳过当前槽位。
        
        返回值:
            np.ndarray: 长度等于(num_depts + 1)的布尔数组，前 num_depts 个元素对应科室，最后一个元素对应跳过动作。
        """
        if self.current_step >= self.num_slots:
            return np.zeros(self.num_depts + 1, dtype=bool)

        # 1. 初始化动作掩码（包括跳过动作）
        action_mask = np.zeros(self.num_depts + 1, dtype=bool)
        
        # 2. 检查哪些科室可以放置在当前槽位
        current_slot_idx = self.shuffled_slot_indices[self.current_step]
        
        # 检查每个未放置的科室的兼容性
        for dept_idx in range(self.num_depts):
            if not self.placed_mask[dept_idx]:  # 只检查未放置的科室
                # 使用 ConstraintManager 的兼容性矩阵进行面积兼容性检查
                if self.constraint_manager.area_compatibility_matrix[current_slot_idx, dept_idx]:
                    action_mask[dept_idx] = True
        
        # 3. 始终允许跳过动作（如果配置允许）
        if self.config.ALLOW_PARTIAL_LAYOUT:
            action_mask[self.SKIP_ACTION] = True
        
        # 4. 如果没有任何合法动作，强制允许跳过（安全检查）
        if not np.any(action_mask):
            current_slot_name = self.placeable_slots[current_slot_idx]
            logger.warning(
                f"在步骤 {self.current_step}，槽位 '{current_slot_name}' 无可放置的科室，强制允许跳过动作"
            )
            action_mask[self.SKIP_ACTION] = True
        
        return action_mask
    
    def _calculate_final_reward(self) -> float:
        """
        在回合结束时，计算最终奖励。
        
        最终奖励 = -(路程加权时间) / 缩放因子 - 空槽位数 * 惩罚
        注意：成功放置的奖励已经在每步动作时给出
        """
        # 构建最终布局（包括空槽位）
        final_layout_depts = []
        placed_depts = []
        
        for slot_idx in range(self.num_slots):
            dept_id = self.layout[slot_idx]
            if dept_id > 0:
                dept_name = self.placeable_depts[dept_id - 1]
                final_layout_depts.append(dept_name)
                placed_depts.append(dept_name)
            else:
                final_layout_depts.append(None)
        
        # 计算空槽位数量
        num_empty_slots = len(self.skipped_slots)
        num_placed_depts = len(placed_depts)
        
        # 如果终止但还有未放置的科室，显示警告
        if num_placed_depts < self.num_depts:
            num_unplaced = self.num_depts - num_placed_depts
            logger.warning(f"Episode终止但仍有 {num_unplaced} 个科室未放置")
        
        logger.debug(f"布局统计: 放置 {num_placed_depts} 个科室，跳过 {num_empty_slots} 个槽位")
        
        # 1. 计算时间成本（只基于已放置的科室）
        if num_placed_depts > 0:
            raw_time_cost = self.cc.calculate_total_cost(placed_depts)
            # 将时间成本转换为负奖励（时间越少越好）
            time_reward = -raw_time_cost / self.config.REWARD_SCALE_FACTOR
        else:
            # 如果没有放置任何科室，给予极大惩罚
            raw_time_cost = float('inf')
            time_reward = -1000.0
            logger.warning("没有放置任何科室，给予极大惩罚")
        
        # 2. 计算空槽位惩罚
        empty_slot_penalty = -num_empty_slots * self.config.REWARD_EMPTY_SLOT_PENALTY
        
        # 3. 计算成功放置奖励（累积值）
        placement_bonus = num_placed_depts * self.config.REWARD_PLACEMENT_BONUS
        
        # 4. 计算总奖励
        total_reward = time_reward * self.config.REWARD_TIME_WEIGHT + empty_slot_penalty + placement_bonus
        
        logger.info(f"最终奖励计算: 时间成本={raw_time_cost:.2f}, "
                   f"时间奖励={time_reward:.2f}, 成功放置奖励={placement_bonus:.2f}, "
                   f"空槽位惩罚={empty_slot_penalty:.2f}, 总奖励={total_reward:.2f}")
        
        return total_reward
    
    def _calculate_reward(self) -> float:
        """
        在回合结束时，根据最终布局计算并返回总奖励。
        
        奖励由以下部分组成：
        1. 时间成本奖励：根据已放置科室的通行时间计算
        2. 邻接约束奖励：软约束奖励
        3. 空槽位惩罚：根据跳过的槽位数量给予惩罚
        """
        # 构建最终布局（包括空槽位）
        final_layout_depts = []
        placed_depts = []
        
        for slot_idx in range(self.num_slots):
            dept_id = self.layout[slot_idx]
            if dept_id > 0:
                dept_name = self.placeable_depts[dept_id - 1]
                final_layout_depts.append(dept_name)
                placed_depts.append(dept_name)
            else:
                final_layout_depts.append(None)
        
        # 计算空槽位数量
        num_empty_slots = len(self.skipped_slots)
        num_placed_depts = len(placed_depts)
        
        logger.debug(f"布局统计: 放置 {num_placed_depts} 个科室，跳过 {num_empty_slots} 个槽位")
        
        # 1. 计算时间成本奖励（只基于已放置的科室）
        if num_placed_depts > 0:
            # 只传递已放置的科室给成本计算器
            time_cost = self.cc.calculate_total_cost(placed_depts)
            time_reward = -time_cost / 1e4  # 缩放以稳定训练
        else:
            # 如果没有放置任何科室，时间成本为0
            time_cost = 0.0
            time_reward = 0.0
        
        # 2. 邻接约束奖励（只基于已放置的科室）
        if self.config.ENABLE_ADJACENCY_REWARD:
            layout_tuple = tuple(final_layout_depts)  # 转换为元组以便缓存
            adjacency_reward = self._calculate_adjacency_reward(layout_tuple)
        else:
            adjacency_reward = 0.0
        
        # 3. 空槽位惩罚
        empty_penalty = num_empty_slots * self.config.EMPTY_SLOT_PENALTY_FACTOR / 1e4  # 缩放保持一致
        
        # 总奖励计算
        total_reward = (
            self.config.REWARD_TIME_WEIGHT * time_reward + 
            self.config.REWARD_ADJACENCY_WEIGHT * adjacency_reward -
            empty_penalty  # 直接减去惩罚
        )
        
        logger.debug(f"奖励组成: 时间={time_reward:.6f}, 邻接={adjacency_reward:.6f}, 空槽位惩罚={empty_penalty:.6f}, 总计={total_reward:.6f}")
        
        return total_reward
    
    @lru_cache(maxsize=500)
    def _calculate_adjacency_reward(self, layout_tuple: Tuple[str, ...]) -> float:
        """
        计算当前布局的多维度相邻性奖励。
        
        相邻性奖励包含三个维度：
        1. 空间相邻性：基于距离分位数的空间邻近关系
        2. 功能相邻性：基于医疗流程的功能协作关系  
        3. 连通性相邻性：基于图连通性的可达性关系
        
        Args:
            layout_tuple: 当前布局的元组形式（用于LRU缓存）
            
        Returns:
            float: 综合相邻性奖励分数
        """
        if not self.config.ENABLE_ADJACENCY_REWARD:
            return 0.0
        
        # 转换元组为列表，过滤掉None值
        placed_depts = [dept for dept in layout_tuple if dept is not None]
        
        if len(placed_depts) < 2:
            return 0.0  # 少于两个科室无法计算相邻性
        
        # 计算各维度相邻性奖励
        spatial_reward = self._calculate_spatial_adjacency_reward(placed_depts)
        functional_reward = self._calculate_functional_adjacency_reward(placed_depts)
        connectivity_reward = 0.0
        
        if self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0:
            connectivity_reward = self._calculate_connectivity_adjacency_reward(placed_depts)
        
        # 加权组合各维度奖励
        total_reward = (
            self.config.SPATIAL_ADJACENCY_WEIGHT * spatial_reward +
            self.config.FUNCTIONAL_ADJACENCY_WEIGHT * functional_reward +
            self.config.CONNECTIVITY_ADJACENCY_WEIGHT * connectivity_reward
        ) * self.config.ADJACENCY_REWARD_BASE
        
        # 调试日志
        if logger.isEnabledFor(10):  # DEBUG级别
            logger.debug(f"相邻性奖励详情: 空间={spatial_reward:.3f}, "
                        f"功能={functional_reward:.3f}, 连通性={connectivity_reward:.3f}, "
                        f"总计={total_reward:.3f}")
        
        return total_reward

    def _calculate_spatial_adjacency_reward(self, placed_depts: List[str]) -> float:
        """
        计算空间相邻性奖励。
        基于预计算的空间相邻性矩阵。
        
        Args:
            placed_depts: 已放置的科室列表
            
        Returns:
            float: 空间相邻性奖励分数
        """
        # 安全性检查：矩阵是否存在
        if not hasattr(self, 'spatial_adjacency_matrix') or self.spatial_adjacency_matrix is None:
            logger.debug("空间相邻性矩阵不存在，返回0奖励")
            return 0.0
        
        # 安全性检查：输入验证
        if not placed_depts or len(placed_depts) < 2:
            logger.debug(f"科室数量不足以计算空间相邻性：{len(placed_depts) if placed_depts else 0}")
            return 0.0
        
        # 安全性检查：矩阵维度验证
        matrix_shape = self.spatial_adjacency_matrix.shape
        if len(matrix_shape) != 2 or matrix_shape[0] == 0 or matrix_shape[1] == 0:
            logger.error(f"空间相邻性矩阵维度异常：{matrix_shape}")
            return 0.0
        
        reward = 0.0
        count = 0
        failed_lookups = 0
        
        try:
            for i, dept1 in enumerate(placed_depts):
                # 安全的字典查找
                if dept1 is None or dept1 not in self.dept_to_idx:
                    failed_lookups += 1
                    logger.debug(f"科室 '{dept1}' 不在索引映射中")
                    continue
                    
                idx1 = self.dept_to_idx[dept1]
                
                # 完整的边界检查
                if not (0 <= idx1 < matrix_shape[0]):
                    failed_lookups += 1
                    logger.warning(f"科室 '{dept1}' 索引 {idx1} 超出矩阵行范围 [0, {matrix_shape[0]})")
                    continue
                
                for j, dept2 in enumerate(placed_depts[i+1:], i+1):
                    # 安全的字典查找
                    if dept2 is None or dept2 not in self.dept_to_idx:
                        failed_lookups += 1
                        logger.debug(f"科室 '{dept2}' 不在索引映射中")
                        continue
                        
                    idx2 = self.dept_to_idx[dept2]
                    
                    # 完整的边界检查
                    if not (0 <= idx2 < matrix_shape[1]):
                        failed_lookups += 1
                        logger.warning(f"科室 '{dept2}' 索引 {idx2} 超出矩阵列范围 [0, {matrix_shape[1]})")
                        continue
                    
                    # 安全的矩阵访问
                    try:
                        adjacency_score = self.spatial_adjacency_matrix[idx1, idx2]
                        
                        # 验证矩阵值的有效性
                        if np.isnan(adjacency_score) or np.isinf(adjacency_score):
                            logger.warning(f"空间相邻性矩阵包含无效值：[{idx1}, {idx2}] = {adjacency_score}")
                            continue
                        
                        reward += adjacency_score
                        count += 1
                        
                    except (IndexError, TypeError) as e:
                        failed_lookups += 1
                        logger.error(f"空间相邻性矩阵访问错误 [{idx1}, {idx2}]：{e}")
                        continue
        
        except Exception as e:
            logger.error(f"空间相邻性奖励计算过程中发生未预期错误：{e}")
            return 0.0
        
        # 记录统计信息
        if failed_lookups > 0:
            logger.debug(f"空间相邻性计算中有 {failed_lookups} 次查找失败")
        
        # 返回平均相邻性分数
        if count > 0:
            avg_reward = reward / count
            logger.debug(f"空间相邻性奖励：{reward:.4f} / {count} = {avg_reward:.4f}")
            return avg_reward
        else:
            logger.debug("空间相邻性计算：无有效科室对")
            return 0.0

    def _calculate_functional_adjacency_reward(self, placed_depts: List[str]) -> float:
        """
        计算功能相邻性奖励。
        基于医疗流程驱动的功能协作关系。
        
        Args:
            placed_depts: 已放置的科室列表
            
        Returns:
            float: 功能相邻性奖励分数
        """
        # 安全性检查：矩阵是否存在
        if not hasattr(self, 'functional_adjacency_matrix') or self.functional_adjacency_matrix is None:
            logger.debug("功能相邻性矩阵不存在，返回0奖励")
            return 0.0
        
        # 安全性检查：输入验证
        if not placed_depts or len(placed_depts) < 2:
            logger.debug(f"科室数量不足以计算功能相邻性：{len(placed_depts) if placed_depts else 0}")
            return 0.0
        
        # 安全性检查：矩阵维度验证
        matrix_shape = self.functional_adjacency_matrix.shape
        if len(matrix_shape) != 2 or matrix_shape[0] == 0 or matrix_shape[1] == 0:
            logger.error(f"功能相邻性矩阵维度异常：{matrix_shape}")
            return 0.0
        
        reward = 0.0
        count = 0
        failed_lookups = 0
        
        try:
            for i, dept1 in enumerate(placed_depts):
                # 安全的字典查找
                if dept1 is None or dept1 not in self.dept_to_idx:
                    failed_lookups += 1
                    logger.debug(f"科室 '{dept1}' 不在索引映射中")
                    continue
                    
                idx1 = self.dept_to_idx[dept1]
                
                # 完整的边界检查
                if not (0 <= idx1 < matrix_shape[0]):
                    failed_lookups += 1
                    logger.warning(f"科室 '{dept1}' 索引 {idx1} 超出矩阵行范围 [0, {matrix_shape[0]})")
                    continue
                
                for j, dept2 in enumerate(placed_depts[i+1:], i+1):
                    # 安全的字典查找
                    if dept2 is None or dept2 not in self.dept_to_idx:
                        failed_lookups += 1
                        logger.debug(f"科室 '{dept2}' 不在索引映射中")
                        continue
                        
                    idx2 = self.dept_to_idx[dept2]
                    
                    # 完整的边界检查
                    if not (0 <= idx2 < matrix_shape[1]):
                        failed_lookups += 1
                        logger.warning(f"科室 '{dept2}' 索引 {idx2} 超出矩阵列范围 [0, {matrix_shape[1]})")
                        continue
                    
                    # 安全的矩阵访问
                    try:
                        preference_score = self.functional_adjacency_matrix[idx1, idx2]
                        
                        # 验证矩阵值的有效性
                        if np.isnan(preference_score) or np.isinf(preference_score):
                            logger.warning(f"功能相邻性矩阵包含无效值：[{idx1}, {idx2}] = {preference_score}")
                            continue
                        
                        # 正向偏好给予奖励，负向偏好给予惩罚
                        if preference_score > 0:
                            reward += preference_score
                        elif preference_score < 0:
                            # 安全地获取惩罚倍数
                            penalty_multiplier = getattr(self.config, 'ADJACENCY_PENALTY_MULTIPLIER', 1.0)
                            reward += preference_score * penalty_multiplier
                        # preference_score == 0 时不计入奖励
                        
                        count += 1
                        
                    except (IndexError, TypeError) as e:
                        failed_lookups += 1
                        logger.error(f"功能相邻性矩阵访问错误 [{idx1}, {idx2}]：{e}")
                        continue
        
        except Exception as e:
            logger.error(f"功能相邻性奖励计算过程中发生未预期错误：{e}")
            return 0.0
        
        # 记录统计信息
        if failed_lookups > 0:
            logger.debug(f"功能相邻性计算中有 {failed_lookups} 次查找失败")
        
        # 返回平均功能相邻性分数
        if count > 0:
            avg_reward = reward / count
            logger.debug(f"功能相邻性奖励：{reward:.4f} / {count} = {avg_reward:.4f}")
            return avg_reward
        else:
            logger.debug("功能相邻性计算：无有效科室对")
            return 0.0

    def _calculate_connectivity_adjacency_reward(self, placed_depts: List[str]) -> float:
        """
        计算连通性相邻性奖励。
        基于图连通性的可达性关系，实现真正的多跳路径分析。
        
        Args:
            placed_depts: 已放置的科室列表
            
        Returns:
            float: 连通性相邻性奖励分数
        """
        # 安全性检查：矩阵是否存在
        if not hasattr(self, 'connectivity_adjacency_matrix') or self.connectivity_adjacency_matrix is None:
            logger.debug("连通性相邻性矩阵不存在，返回0奖励")
            return 0.0
        
        # 安全性检查：输入验证
        if not placed_depts or len(placed_depts) < 2:
            logger.debug(f"科室数量不足以计算连通性相邻性：{len(placed_depts) if placed_depts else 0}")
            return 0.0
        
        # 安全性检查：矩阵维度验证
        matrix_shape = self.connectivity_adjacency_matrix.shape
        if len(matrix_shape) != 2 or matrix_shape[0] == 0 or matrix_shape[1] == 0:
            logger.error(f"连通性相邻性矩阵维度异常：{matrix_shape}")
            return 0.0
        
        reward = 0.0
        count = 0
        failed_lookups = 0
        
        try:
            for i, dept1 in enumerate(placed_depts):
                # 安全的字典查找
                if dept1 is None or dept1 not in self.dept_to_idx:
                    failed_lookups += 1
                    logger.debug(f"科室 '{dept1}' 不在索引映射中")
                    continue
                    
                idx1 = self.dept_to_idx[dept1]
                
                # 完整的边界检查
                if not (0 <= idx1 < matrix_shape[0]):
                    failed_lookups += 1
                    logger.warning(f"科室 '{dept1}' 索引 {idx1} 超出矩阵行范围 [0, {matrix_shape[0]})")
                    continue
                
                for j, dept2 in enumerate(placed_depts[i+1:], i+1):
                    # 安全的字典查找
                    if dept2 is None or dept2 not in self.dept_to_idx:
                        failed_lookups += 1
                        logger.debug(f"科室 '{dept2}' 不在索引映射中")
                        continue
                        
                    idx2 = self.dept_to_idx[dept2]
                    
                    # 完整的边界检查
                    if not (0 <= idx2 < matrix_shape[1]):
                        failed_lookups += 1
                        logger.warning(f"科室 '{dept2}' 索引 {idx2} 超出矩阵列范围 [0, {matrix_shape[1]})")
                        continue
                    
                    # 安全的矩阵访问和多跳路径分析
                    try:
                        # 直接连通性权重
                        direct_connectivity = self.connectivity_adjacency_matrix[idx1, idx2]
                        
                        # 验证矩阵值的有效性
                        if np.isnan(direct_connectivity) or np.isinf(direct_connectivity):
                            logger.warning(f"连通性相邻性矩阵包含无效值：[{idx1}, {idx2}] = {direct_connectivity}")
                            continue
                        
                        # 实现多跳路径分析
                        multi_hop_connectivity = self._calculate_multi_hop_connectivity(idx1, idx2, dept1, dept2)
                        
                        # 组合直接连通性和多跳连通性
                        total_connectivity = direct_connectivity + 0.5 * multi_hop_connectivity
                        
                        reward += total_connectivity
                        count += 1
                        
                    except (IndexError, TypeError) as e:
                        failed_lookups += 1
                        logger.error(f"连通性矩阵访问错误 [{idx1}, {idx2}]：{e}")
                        continue
        
        except Exception as e:
            logger.error(f"连通性相邻性奖励计算过程中发生未预期错误：{e}")
            return 0.0
        
        # 记录统计信息
        if failed_lookups > 0:
            logger.debug(f"连通性相邻性计算中有 {failed_lookups} 次查找失败")
        
        # 返回平均连通性相邻性分数
        if count > 0:
            avg_reward = reward / count
            logger.debug(f"连通性相邻性奖励：{reward:.4f} / {count} = {avg_reward:.4f}")
            return avg_reward
        else:
            logger.debug("连通性相邻性计算：无有效科室对")
            return 0.0

    def _calculate_multi_hop_connectivity(self, idx1: int, idx2: int, dept1: str, dept2: str) -> float:
        """
        计算两个科室之间的多跳连通性权重。
        基于真正的图连通性分析，考虑2-3跳的间接路径。
        
        Args:
            idx1: 第一个科室的索引
            idx2: 第二个科室的索引
            dept1: 第一个科室的名称（用于日志）
            dept2: 第二个科室的名称（用于日志）
            
        Returns:
            float: 多跳连通性权重（0-1之间）
        """
        try:
            # 获取配置参数并进行验证
            max_path_length = getattr(self.config, 'CONNECTIVITY_MAX_PATH_LENGTH', 3)
            weight_decay = getattr(self.config, 'CONNECTIVITY_WEIGHT_DECAY', 0.8)
            
            # 参数验证
            if max_path_length < 2 or max_path_length > 5:
                logger.warning(f"多跳路径长度配置异常：{max_path_length}，使用默认值3")
                max_path_length = 3
                
            if weight_decay <= 0 or weight_decay >= 1:
                logger.warning(f"权重衰减因子配置异常：{weight_decay}，使用默认值0.8")
                weight_decay = 0.8
            
            # 获取行程时间矩阵进行多跳分析
            if not hasattr(self, 'travel_times_matrix') or self.travel_times_matrix is None:
                logger.debug("行程时间矩阵不存在，无法计算多跳连通性")
                return 0.0
            
            # 安全获取节点名称并验证存在性
            try:
                if dept1 not in self.travel_times_matrix.index or dept2 not in self.travel_times_matrix.columns:
                    logger.debug(f"科室 '{dept1}' 或 '{dept2}' 不在行程时间矩阵中")
                    return 0.0
            except Exception as e:
                logger.debug(f"检查科室是否在行程时间矩阵中时发生错误：{e}")
                return 0.0

            multi_hop_weight = 0.0
            
            # 计算2跳和3跳路径的连通性
            for path_length in range(2, min(max_path_length + 1, 4)):  # 限制最大路径长度为3
                try:
                    path_weight = self._calculate_path_connectivity(dept1, dept2, path_length, weight_decay)
                    if path_weight > 0:
                        # 路径长度越长，权重衰减越多
                        discounted_weight = path_weight * (weight_decay ** (path_length - 1))
                        multi_hop_weight += discounted_weight
                        
                        logger.debug(f"{path_length}跳路径连通性：{dept1} -> {dept2}，权重={discounted_weight:.4f}")
                        
                except Exception as e:
                    logger.debug(f"计算{path_length}跳路径时发生错误：{e}")
                    continue
            
            # 限制多跳权重在合理范围内
            multi_hop_weight = min(multi_hop_weight, 1.0)
            
            if multi_hop_weight > 0:
                logger.debug(f"多跳连通性总权重：{dept1} <-> {dept2} = {multi_hop_weight:.4f}")
                
            return multi_hop_weight
            
        except Exception as e:
            logger.error(f"多跳连通性计算过程中发生未预期错误：{e}")
            return 0.0
    
    def _calculate_path_connectivity(self, start_dept: str, end_dept: str, path_length: int, weight_decay: float) -> float:
        """
        计算指定路径长度的连通性权重。
        使用动态规划方法计算最短路径权重。
        
        Args:
            start_dept: 起始科室名称
            end_dept: 目标科室名称
            path_length: 路径长度（跳数）
            weight_decay: 权重衰减因子
            
        Returns:
            float: 路径连通性权重
        """
        try:
            # 安全性检查：行程时间矩阵验证
            if not hasattr(self, 'travel_times_matrix') or self.travel_times_matrix is None:
                logger.debug("行程时间矩阵不存在，无法计算多跳路径连通性")
                return 0.0
            
            # 安全性检查：起始和结束科室存在性
            if (start_dept not in self.travel_times_matrix.index or 
                end_dept not in self.travel_times_matrix.columns):
                logger.debug(f"科室 '{start_dept}' 或 '{end_dept}' 不在行程时间矩阵中")
                return 0.0
                
            # 获取所有可能的中介节点（增加安全检查）
            try:
                available_nodes = []
                for node in self.placeable_depts:
                    if (node and node != start_dept and node != end_dept and 
                        node in self.travel_times_matrix.index and 
                        node in self.travel_times_matrix.columns):
                        available_nodes.append(node)
            except Exception as e:
                logger.error(f"获取可用中介节点时发生错误：{e}")
                return 0.0
            
            if len(available_nodes) == 0:
                logger.debug(f"没有可用的中介节点用于计算{path_length}跳路径：{start_dept} -> {end_dept}")
                return 0.0
                
            best_path_weight = 0.0
            
            if path_length == 2:
                # 2跳路径：start -> intermediate -> end
                for intermediate in available_nodes:
                    try:
                        # 安全的矩阵访问
                        time1 = self.travel_times_matrix.loc[start_dept, intermediate]
                        time2 = self.travel_times_matrix.loc[intermediate, end_dept]
                        
                        # 数值有效性检查
                        if (time1 > 0 and time2 > 0 and 
                            not (np.isnan(time1) or np.isnan(time2) or np.isinf(time1) or np.isinf(time2))):
                            # 使用调和平均数计算路径权重（安全除法）
                            try:
                                path_weight = 2.0 / (1.0/time1 + 1.0/time2)
                                # 转换为连通性权重（时间越短，连通性越强）
                                connectivity_weight = 1.0 / (1.0 + path_weight / 100.0)  # 标准化到0-1
                                best_path_weight = max(best_path_weight, connectivity_weight)
                            except ZeroDivisionError:
                                logger.debug(f"2跳路径计算中出现除零错误：{start_dept}->{intermediate}->{end_dept}")
                                continue
                            
                    except (KeyError, TypeError, ValueError) as e:
                        logger.debug(f"计算2跳路径权重时出错：{start_dept}->{intermediate}->{end_dept}，错误：{e}")
                        continue
                        
            elif path_length == 3:
                # 3跳路径：start -> int1 -> int2 -> end
                for int1 in available_nodes:
                    for int2 in available_nodes:
                        if int1 != int2:  # 避免循环
                            try:
                                # 安全的矩阵访问
                                time1 = self.travel_times_matrix.loc[start_dept, int1]
                                time2 = self.travel_times_matrix.loc[int1, int2]
                                time3 = self.travel_times_matrix.loc[int2, end_dept]
                                
                                # 数值有效性检查
                                if (time1 > 0 and time2 > 0 and time3 > 0 and 
                                    not any(np.isnan([time1, time2, time3]) or np.isinf([time1, time2, time3]))):
                                    # 使用调和平均数计算路径权重（安全除法）
                                    try:
                                        path_weight = 3.0 / (1.0/time1 + 1.0/time2 + 1.0/time3)
                                        connectivity_weight = 1.0 / (1.0 + path_weight / 100.0)
                                        best_path_weight = max(best_path_weight, connectivity_weight)
                                    except ZeroDivisionError:
                                        logger.debug(f"3跳路径计算中出现除零错误：{start_dept}->{int1}->{int2}->{end_dept}")
                                        continue
                                    
                            except (KeyError, TypeError, ValueError) as e:
                                logger.debug(f"计算3跳路径权重时出错：{start_dept}->{int1}->{int2}->{end_dept}，错误：{e}")
                                continue
            
            return best_path_weight
            
        except Exception as e:
            logger.error(f"路径连通性计算过程中发生未预期错误：{e}")
            return 0.0
    
    def _calculate_area_match_score(self, dept_idx: int, slot_idx: int) -> float:
        """
        计算科室与槽位的面积匹配度分数。
        
        Args:
            dept_idx: 科室在placeable_depts中的索引
            slot_idx: 槽位在placeable_slots中的索引
            
        Returns:
            float: 0到1之间的匹配度分数，1表示完美匹配，0表示差异最大
        """
        # 获取科室和槽位的面积
        dept_name = self.placeable_depts[dept_idx]
        dept_area = self.dept_areas_map[dept_name]
        slot_area = self.slot_areas[slot_idx]
        
        # 避免除零错误
        if dept_area == 0 or slot_area == 0:
            logger.warning(f"面积为0：科室 {dept_name} 面积={dept_area}, 槽位索引 {slot_idx} 面积={slot_area}")
            return 0.0
        
        # 计算相对差异（使用较大值作为基准）
        relative_diff = abs(dept_area - slot_area) / max(dept_area, slot_area)
        
        # 转换为0-1的匹配分数（差异越小，分数越高）
        # 使用AREA_SCALING_FACTOR作为容差阈值
        match_score = max(0.0, 1.0 - relative_diff / self.config.AREA_SCALING_FACTOR)
        
        return match_score
    
    def _calculate_potential(self) -> float:
        """
        计算当前布局状态的势函数值。
        势函数 Φ(layout) = -1 * (时间成本部分) + 面积匹配奖励部分 + 相邻性奖励部分
        
        Returns:
            float: 当前状态的势函数值
        """
        # 构建完整的布局（包括空槽位）
        layout_with_nulls = []
        placed_depts = []
        
        for slot_idx in range(self.num_slots):
            dept_id = self.layout[slot_idx]
            if dept_id > 0:
                dept_name = self.placeable_depts[dept_id - 1]
                layout_with_nulls.append(dept_name)
                placed_depts.append(dept_name)
            else:
                layout_with_nulls.append(None)
        
        # === 时间成本部分 ===
        # 如果没有放置科室或只有一个科室，时间成本势函数为0
        time_cost_potential = 0.0
        if len(placed_depts) > 1:
            # 计算已放置科室的总时间成本
            # 传递完整布局（包含None），让CostCalculator正确映射科室到槽位
            time_cost = self.cc.calculate_total_cost(layout_with_nulls)
            # 势函数为负的时间成本（缩放后）
            time_cost_potential = -time_cost / self.config.REWARD_SCALE_FACTOR
        
        # === 面积匹配奖励部分 ===
        area_match_reward = 0.0
        num_matched_depts = 0
        total_match_score = 0.0
        
        for slot_idx in range(self.num_slots):
            dept_id = self.layout[slot_idx]
            if dept_id > 0:
                dept_idx = dept_id - 1
                match_score = self._calculate_area_match_score(dept_idx, slot_idx)
                area_match_reward += match_score * self.config.AREA_MATCH_BONUS_BASE
                total_match_score += match_score
                num_matched_depts += 1
        
        # 计算平均匹配度（用于日志）
        avg_match_score = total_match_score / num_matched_depts if num_matched_depts > 0 else 0.0
        
        # === 相邻性奖励部分 ===
        adjacency_reward = 0.0
        if self.config.ENABLE_ADJACENCY_REWARD and len(placed_depts) >= 2:
            # 使用元组以便LRU缓存
            layout_tuple = tuple(layout_with_nulls)
            adjacency_reward = self._calculate_adjacency_reward(layout_tuple)
        
        # === 组合所有部分势函数 ===
        total_potential = (
            time_cost_potential + 
            area_match_reward * self.config.AREA_MATCH_REWARD_WEIGHT +
            adjacency_reward * self.config.ADJACENCY_REWARD_WEIGHT
        )
        
        # 详细的调试日志
        if logger.isEnabledFor(10):  # DEBUG级别
            logger.debug(f"势函数计算详情: "
                        f"已放置{len(placed_depts)}个科室, "
                        f"时间成本势={time_cost_potential:.4f}, "
                        f"面积匹配奖励={area_match_reward:.4f}, "
                        f"相邻性奖励={adjacency_reward:.4f}, "
                        f"平均匹配度={avg_match_score:.3f}, "
                        f"总势函数={total_potential:.4f}")
        
        return total_potential
    
    def _calculate_area_match_statistics(self) -> Dict[str, float]:
        """
        计算当前布局的面积匹配统计信息。
        
        Returns:
            Dict[str, float]: 包含平均、最小、最大匹配度的字典
        """
        match_scores = []
        
        for slot_idx in range(self.num_slots):
            dept_id = self.layout[slot_idx]
            if dept_id > 0:
                dept_idx = dept_id - 1
                match_score = self._calculate_area_match_score(dept_idx, slot_idx)
                match_scores.append(match_score)
        
        if not match_scores:
            return {
                'avg_match_score': 0.0,
                'min_match_score': 0.0,
                'max_match_score': 0.0,
                'num_matched': 0
            }
        
        return {
            'avg_match_score': sum(match_scores) / len(match_scores),
            'min_match_score': min(match_scores),
            'max_match_score': max(match_scores),
            'num_matched': len(match_scores)
        }
    
    def _calculate_adjacency_statistics(self, placed_depts: List[str]) -> Dict[str, float]:
        """
        计算当前布局的相邻性统计信息。
        
        Args:
            placed_depts: 已放置的科室列表
            
        Returns:
            Dict[str, float]: 包含各维度相邻性奖励的字典
        """
        if not self.config.ENABLE_ADJACENCY_REWARD or len(placed_depts) < 2:
            return {
                'spatial_reward': 0.0,
                'functional_reward': 0.0,
                'connectivity_reward': 0.0,
                'total_reward': 0.0
            }
        
        # 计算各维度相邻性奖励
        spatial_reward = self._calculate_spatial_adjacency_reward(placed_depts)
        functional_reward = self._calculate_functional_adjacency_reward(placed_depts)
        connectivity_reward = 0.0
        
        if self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0:
            connectivity_reward = self._calculate_connectivity_adjacency_reward(placed_depts)
        
        # 计算总奖励
        total_reward = (
            self.config.SPATIAL_ADJACENCY_WEIGHT * spatial_reward +
            self.config.FUNCTIONAL_ADJACENCY_WEIGHT * functional_reward +
            self.config.CONNECTIVITY_ADJACENCY_WEIGHT * connectivity_reward
        ) * self.config.ADJACENCY_REWARD_BASE
        
        return {
            'spatial_reward': spatial_reward,
            'functional_reward': functional_reward,
            'connectivity_reward': connectivity_reward,
            'total_reward': total_reward
        }

    def _log_area_match_statistics(self):
        """记录面积匹配统计信息到日志。"""
        stats = self._calculate_area_match_statistics()
        if stats['num_matched'] > 0:
            adjacency_info = ""
            if self.config.ENABLE_ADJACENCY_REWARD:
                # 获取当前已放置的科室
                placed_depts = []
                for slot_idx in range(self.num_slots):
                    dept_id = self.layout[slot_idx]
                    if dept_id > 0:
                        placed_depts.append(self.placeable_depts[dept_id - 1])
                
                if len(placed_depts) >= 2:
                    adj_stats = self._calculate_adjacency_statistics(placed_depts)
                    adjacency_info = (f", 相邻性奖励: 空间={adj_stats['spatial_reward']:.3f}, "
                                    f"功能={adj_stats['functional_reward']:.3f}, "
                                    f"连通性={adj_stats['connectivity_reward']:.3f}, "
                                    f"总计={adj_stats['total_reward']:.3f}")
            
            logger.info(f"面积匹配统计: 平均匹配度={stats['avg_match_score']:.3f}, "
                       f"最小={stats['min_match_score']:.3f}, 最大={stats['max_match_score']:.3f}, "
                       f"已匹配科室数={stats['num_matched']}{adjacency_info}")
    
    
    def render(self, mode="human"):
        """(可选) 渲染环境状态，用于调试。"""
        if mode == "human":
            print(f"--- Step: {self.current_step} ---")
            current_slot_idx = self.shuffled_slot_indices[self.current_step]
            current_slot = self.placeable_slots[current_slot_idx]
            print(f"Current Slot to Fill: {current_slot} (Area: {self.slot_areas[current_slot_idx]:.2f})")
            
            layout_str = []
            for i in range(self.num_slots):
                dept_id = self.layout[i]
                if dept_id > 0:
                    layout_str.append(f"{self.placeable_slots[i]}: {self.placeable_depts[dept_id-1]}")
                else:
                    layout_str.append(f"{self.placeable_slots[i]}: EMPTY")
            print("Current Layout:\n" + "\n".join(layout_str))

    @staticmethod
    def _action_mask_fn(env: 'LayoutEnv') -> np.ndarray:
        """
        ActionMasker 包装器所需的静态方法，用于提取动作掩码。
        
        Args:
            env (LayoutEnv): 环境实例。
            
        Returns:
            np.ndarray: 当前状态下的动作掩码。
        """
        return env.get_action_mask()

    def _get_final_layout_str(self) -> List[str]:
        """
        获取最终布局的科室名称列表。
        
        Returns:
            List[str]: 按槽位顺序排列的科室名称列表。
        """
        final_layout = [None] * self.num_slots
        for slot_idx, dept_id in enumerate(self.layout):
            if dept_id > 0:
                final_layout[slot_idx] = self.placeable_depts[dept_id - 1]
        return final_layout