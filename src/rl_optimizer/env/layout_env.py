# src/rl_optimizer/env/layout_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境，按面积从小到大排序槽位，并返回初始观测。"""
        super().reset(seed=seed)
        self._initialize_state_variables()
        # 修改：按槽位面积从小到大排序，而不是随机打乱
        self.shuffled_slot_indices = np.argsort(self.slot_areas)
        
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
        
        # 计算奖励：即时奖励（成功放置）+ 最终奖励（终止时的总体评分）
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
        adjacency_reward = self._calculate_adjacency_reward(placed_depts)
        
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