# src/rl_optimizer/env/layout_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__, level=10)  # 临时启用DEBUG级别

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

    def __init__(self, config: RLConfig, cache_manager: CacheManager, cost_calculator: CostCalculator):
        """
        初始化布局优化环境。

        Args:
            config (RLConfig): RL优化器的配置对象。
            cache_manager (CacheManager): 已初始化的数据缓存管理器。
            cost_calculator (CostCalculator): 已初始化的成本计算器。
        """
        super().__init__()
        self.config = config
        self.cm = cache_manager
        self.cc = cost_calculator

        self._initialize_nodes_and_slots()
        self._define_spaces()
        self._initialize_state_variables()

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

    def _define_spaces(self):
        """定义观测空间和动作空间。"""
        # 动作空间：选择一个科室进行放置。动作是科室的索引。
        self.action_space = spaces.Discrete(self.num_depts)

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
            "current_slot_idx": spaces.Box(low=0, high=self.num_slots - 1, shape=(1,), dtype=np.int32)
        })

    def _initialize_state_variables(self):
        """初始化每个回合都会改变的状态变量。"""
        self.current_step = 0
        # layout的索引是物理槽位索引，值是科室索引+1
        self.layout = np.zeros(self.num_slots, dtype=np.int32)
        self.placed_mask = np.zeros(self.num_depts, dtype=bool)
        # 每个回合开始时需要被打乱的槽位处理顺序
        self.shuffled_slot_indices = np.arange(self.num_slots)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境，随机化槽位顺序，并返回初始观测。"""
        super().reset(seed=seed)
        self._initialize_state_variables()
        self.np_random.shuffle(self.shuffled_slot_indices)
        return self._get_obs(), self._get_info(terminated=False)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一个动作：在当前槽位放置选定的科室。"""
        # 更严格的输入验证
        if not (0 <= action < self.num_depts):
            logger.error(f"无效的动作索引: {action}，有效范围是 [0, {self.num_depts-1}]")
            return self._get_obs(), -100.0, True, False, self._get_info(terminated=False)
        
        if self.placed_mask[action]:
            # 这是一个安全检查。理论上动作掩码会阻止这种情况。
            # 如果发生，说明上游逻辑有误，应给予重罚并终止。
            logger.error(f"严重错误：智能体选择了已被放置的科室！动作={action}, 科室={self.placeable_depts[action]}")
            logger.error(f"当前步骤: {self.current_step}, placed_mask: {self.placed_mask}")
            logger.error(f"当前动作掩码: {self.get_action_mask()}")
            return self._get_obs(), -100.0, True, False, self._get_info(terminated=False)

        # 确定当前要填充的物理槽位索引
        slot_to_fill = self.shuffled_slot_indices[self.current_step]
        
        # 更新状态：在布局中记录放置，并标记科室为"已用"
        self.layout[slot_to_fill] = action + 1  # 动作是科室索引，存储时+1
        self.placed_mask[action] = True
        self.current_step += 1

        terminated = (self.current_step == self.num_slots)
        reward = self._calculate_reward() if terminated else 0.0
        info = self._get_info(terminated)
        
        # 调试日志：检查episode结束时的info内容
        if terminated and logger.isEnabledFor(10):  # DEBUG级别
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
            "current_slot_idx": np.array([current_slot_idx], dtype=np.int32)
        }
    
    def _get_info(self, terminated: bool = False) -> Dict[str, Any]:
        """返回包含动作掩码和episode信息的附加信息。"""
        info = {"action_mask": self.get_action_mask()}
        
        # 如果episode结束，添加训练指标信息
        if terminated:
            # 构造最终布局
            final_layout_depts = [None] * self.num_slots
            for slot_idx, dept_id in enumerate(self.layout):
                if dept_id > 0:
                    final_layout_depts[slot_idx] = self.placeable_depts[dept_id - 1]
            
            # 如果布局完整，计算时间成本
            if None not in final_layout_depts:
                # 计算原始的总时间成本（未缩放）
                raw_time_cost = self.cc.calculate_total_cost(final_layout_depts)
                
                # 计算用于训练的缩放reward（与_calculate_reward中的逻辑一致）
                scaled_time_reward = -raw_time_cost / 1e4
                
                # 调试日志：确认传递的是原始值
                if logger.isEnabledFor(10):  # DEBUG级别
                    logger.debug(f"Episode结束 - 原始时间成本: {raw_time_cost:.2f}, "
                               f"缩放后训练reward: {scaled_time_reward:.6f}")
                
                # 添加episode信息供TrainingMetricsCallback使用
                info['episode'] = {
                    'time_cost': raw_time_cost,  # 明确传递原始值
                    'scaled_reward': scaled_time_reward,  # 同时提供缩放值用于对比
                    'layout': final_layout_depts.copy(),
                    'r': self._calculate_reward(),  # 完整的奖励值（包含时间+邻接）
                    'l': self.current_step  # episode长度
                }
                
                # 如果启用了详细日志，还可以添加更多信息
                if logger.isEnabledFor(20):  # INFO级别
                    info['episode']['per_process_costs'] = self.cc.calculate_per_process_cost(final_layout_depts)
        
        return info
    
    def get_action_mask(self) -> np.ndarray:
        """
        计算当前步骤下所有合法的科室动作掩码。
        
        该方法根据当前待填充槽位的面积约束，返回一个布尔数组，指示哪些未放置的科室可被合法选择。如果所有未放置科室均不满足面积约束，将逐步放宽容差，直至至少有一个合法动作；如仍无合法动作，则强制允许所有未放置科室。
        
        返回值:
            np.ndarray: 长度等于科室数的布尔数组，True 表示对应科室当前可被选择。
        """
        if self.current_step >= self.num_slots:
            return np.zeros(self.num_depts, dtype=bool)

        # 1. 初始掩码：所有未放置的科室都是潜在的合法动作
        action_mask = ~self.placed_mask

        # 2. 面积约束：科室面积必须符合当前槽位的容差
        current_slot_idx = self.shuffled_slot_indices[self.current_step]
        current_slot_area = self.slot_areas[current_slot_idx]
        min_area = current_slot_area * (1 - self.config.AREA_SCALING_FACTOR)
        max_area = current_slot_area * (1 + self.config.AREA_SCALING_FACTOR)

        for dept_idx in range(self.num_depts):
            if action_mask[dept_idx]:  # 只检查当前合法的动作
                dept_name = self.placeable_depts[dept_idx]
                dept_area = self.dept_areas_map[dept_name]
                if not (min_area <= dept_area <= max_area):
                    action_mask[dept_idx] = False
        
        # 容错机制：如果没有合法动作，逐步放松约束
        if np.sum(action_mask) == 0:
            logger.warning(f"在步骤 {self.current_step}，没有找到任何合法的动作！尝试放松约束...")
            
            # 重新计算，使用更宽松的面积约束
            relaxation_factors = [0.2, 0.3, 0.5, 0.7, 1.0]  # 逐步放松
            for factor in relaxation_factors:
                action_mask = ~self.placed_mask  # 重新开始，只考虑未放置的科室
                min_area = current_slot_area * (1 - factor)
                max_area = current_slot_area * (1 + factor)
                
                for dept_idx in range(self.num_depts):
                    if action_mask[dept_idx]:
                        dept_name = self.placeable_depts[dept_idx]
                        dept_area = self.dept_areas_map[dept_name]
                        if not (min_area <= dept_area <= max_area):
                            action_mask[dept_idx] = False
                
                if np.sum(action_mask) > 0:
                    logger.info(f"使用松弛因子 {factor} 找到了 {np.sum(action_mask)} 个合法动作")
                    break
            
            # 最后的安全检查：如果仍然没有合法动作，至少允许所有未放置的科室
            if np.sum(action_mask) == 0:
                current_slot_name = self.placeable_slots[current_slot_idx]
                current_slot_area = self.slot_areas[current_slot_idx]
                
                unplaced_indices = np.where(~self.placed_mask)[0]
                unplaced_depts_info = [
                    f"{self.placeable_depts[i]}({self.dept_areas_map[self.placeable_depts[i]]:.2f})"
                    for i in unplaced_indices
                ]
                
                logger.error(
                    f"即使放松所有约束也没有为槽位 '{current_slot_name}' (面积: {current_slot_area:.2f}) 找到合法科室！"
                    f"剩余待放置科室: [{', '.join(unplaced_depts_info)}]. "
                    f"将强制允许所有未放置的科室。"
                )
                action_mask = ~self.placed_mask
        
        return action_mask
    
    def _calculate_reward(self) -> float:
        """
        在回合结束时，根据最终布局计算并返回总奖励。
        
        如果布局未完成，则返回重罚。奖励由时间成本和邻接约束两部分加权组成，其中时间成本通过 `CostCalculator` 计算并缩放，邻接奖励由 `_calculate_adjacency_reward` 计算。
        """
        final_layout_depts = [None] * self.num_slots
        for slot_idx, dept_id in enumerate(self.layout):
            if dept_id > 0:
                final_layout_depts[slot_idx] = self.placeable_depts[dept_id - 1]
        
        if None in final_layout_depts:
            logger.error("布局计算奖励时发现未完成的布局，返回重罚。")
            return -500.0

        time_cost = self.cc.calculate_total_cost(final_layout_depts)
        time_reward = -time_cost / 1e4  # 缩放以稳定训练

        adjacency_reward = self._calculate_adjacency_reward(final_layout_depts)
        
        total_reward = (self.config.REWARD_TIME_WEIGHT * time_reward + 
                        self.config.REWARD_ADJACENCY_WEIGHT * adjacency_reward)
        
        return total_reward
    
    def _calculate_adjacency_reward(self, _final_layout: List[str]) -> float:
        """
        计算偏好相邻软约束的奖励。
        (TODO: 需要一个可靠的邻接关系数据源)
        """
        # 这是一个示例实现，实际部署时需要替换为真实邻接数据
        # 假设 travel_times 矩阵中时间小于某个阈值（例如 10）即为相邻
        reward = 0.0
        # ... 实现软约束奖励计算 ...
        return reward
    
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