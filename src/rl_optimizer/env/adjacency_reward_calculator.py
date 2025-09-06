# src/rl_optimizer/env/adjacency_reward_calculator.py

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import time

from src.config import RLConfig
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.utils.setup import setup_logger

# @dataclass
# class AdjacencyMetrics:
#     """相邻性奖励计算的性能指标数据类。"""
#     total_time: float = 0.0
#     spatial_time: float = 0.0
#     functional_time: float = 0.0
#     connectivity_time: float = 0.0
#     cache_hits: int = 0
#     cache_misses: int = 0
#     computation_count: int = 0

class AdjacencyRewardCalculator:
    """
    相邻性奖励计算器。
    """
    
    def __init__(self, config: RLConfig, placeable_depts: List[str], 
                 travel_times_matrix, constraint_manager: ConstraintManager):
        self.config = config
        self.placeable_depts = placeable_depts
        self.dept_to_idx = {dept: idx for idx, dept in enumerate(placeable_depts)}
        self.dept_to_idx[None] = -1
        self.num_depts = len(placeable_depts)

        self.placeable_slots = constraint_manager.placeable_slots
        self.num_slots = len(self.placeable_slots)

        self.travel_times_matrix = travel_times_matrix

        self.logger = setup_logger(__name__)

        self.functional_preference_matrix = self._precompute_functional_preference_matrix()

        self.slot_adjacency_matrix = self._precompute_slot_adjacency_matrix()

        self.logger.info(f"动态相邻性奖励计算器初始化完成。")

    def _precompute_functional_preference_matrix(self):
        """
        预计算功能偏好矩阵。
        """
        matrix  = np.zeros((self.num_depts, self.num_depts), dtype=np.float32)
        for i, dept1 in enumerate(self.placeable_depts):
            generic1 = dept1.split('_')[0]
            for j, dept2 in enumerate(self.placeable_depts):
                if i != j:
                    generic2 = dept2.split('_')[0]
                    pref = self.config.MEDICAL_ADJACENCY_PREFERENCES.get(generic1, {}).get(generic2, 0)
                    if pref == 0:
                        pref = self.config.MEDICAL_ADJACENCY_PREFERENCES.get(generic2, {}).get(generic1, 0)
                    matrix[i, j] = pref
        return matrix
    
    def _precompute_slot_adjacency_matrix(self):
        """
        预计算槽位之间的空间相邻强度矩阵（使用连续强度值）。
        返回值: (N_slots, N_slots) 的矩阵，存储槽位间的相邻强度(0到1)。
        由于当前通行时间矩阵包含科室就诊时间，空间相邻性奖励不建议使用
        """

        slot_names = self.placeable_slots
        distance_matrix = self.travel_times_matrix.loc[slot_names, slot_names].values

        uppper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        valid_distances = uppper_triangle[uppper_triangle > 0]

        if len(valid_distances) == 0:
            self.logger.warning("所有槽位间的距离均为零，无法计算空间相邻强度。")
            return np.zeros((self.num_slots, self.num_slots), dtype=np.float32)
        
        threshold = np.percentile(valid_distances, self.config.ADJACENCY_PERCENTILE_THRESHOLD * 100)
        self.logger.info(f"空间相邻性距离阈值设为 {threshold:.2f}")

        with np.errstate(divide='ignore', invalid='ignore'):
            strength_matrix = 1.0 - (distance_matrix / threshold)

        strength_matrix[distance_matrix > threshold] = 0
        strength_matrix[distance_matrix <= 0] = 0
        np.fill_diagonal(strength_matrix, 0)
        strength_matrix = np.maximum(strength_matrix, 0)

        self.logger.debug(f"槽位相邻强度矩阵计算完成，平均强度: {np.mean(strength_matrix):.3f}")

        return strength_matrix
    
    @lru_cache(maxsize=1000)
    def calculate_reward(self, layout_tuple: Tuple[Optional[str], ...]) -> float:
        """
        计算给定完整布局的相邻性总奖励。
        
        Args:
            layout_tuple: 一个元组，长度为槽位数。每个元素是科室名称或None。
                          例如: ('儿科_10006', '全科_10004', None, ...)
        
        Returns:
            综合相邻性奖励分数。
        """
        if len(layout_tuple) != self.num_slots:
            self.logger.error(f"布局长度 ({len(layout_tuple)}) 与槽位数 ({self.num_slots}) 不匹配。")
            return 0.0
        
        dept_indices_in_layout = np.array([self.dept_to_idx[dept] for dept in layout_tuple], dtype=np.int32)

        slot_i_indices, slot_j_indices = np.meshgrid(np.arange(self.num_slots), np.arange(self.num_slots), indexing='ij')

        slot_adjacency_values = self.slot_adjacency_matrix[slot_i_indices, slot_j_indices]

        depts_at_i = dept_indices_in_layout[slot_i_indices]
        depts_at_j = dept_indices_in_layout[slot_j_indices]

        valid_pair_mask = (depts_at_i >= 0) & (depts_at_j >= 0) & (depts_at_i != depts_at_j)

        functional_pref_values = np.zeros_like(depts_at_i, dtype=np.float32)
        valid_depts_at_i = depts_at_i[valid_pair_mask]
        valid_depts_at_j = depts_at_j[valid_pair_mask]
        functional_pref_values[valid_pair_mask] = self.functional_preference_matrix[valid_depts_at_i, valid_depts_at_j]

        pair_scores = functional_pref_values * slot_adjacency_values

        total_score = np.sum(np.triu(pair_scores, k=1))

        num_valid_pairs = np.sum(np.triu(valid_pair_mask, k=1))
        if num_valid_pairs == 0:
            return 0.0
        
        normalized_score = total_score / num_valid_pairs
        
        final_reward = normalized_score * self.config.ADJACENCY_REWARD_BASE
        
        self.logger.info(f"动态相邻性奖励计算: 总分={total_score:.3f}, 有效对数={num_valid_pairs}, 归一化分={normalized_score:.3f}, 最终奖励={final_reward:.3f}")

        return {'total_reward': final_reward}
    
def create_adjacency_calculator(config: RLConfig, placeable_depts: List[str], 
                               travel_times_matrix, constraint_manager: ConstraintManager) -> AdjacencyRewardCalculator:
    return AdjacencyRewardCalculator(
            config=config,
            placeable_depts=placeable_depts,
            travel_times_matrix=travel_times_matrix,
            constraint_manager=constraint_manager
        )