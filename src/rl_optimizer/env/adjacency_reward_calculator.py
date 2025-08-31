# src/rl_optimizer/env/adjacency_reward_calculator.py

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Optional, Set
from functools import lru_cache
import logging
from dataclasses import dataclass
import time

from src.config import RLConfig

logger = logging.getLogger(__name__)

@dataclass
class AdjacencyMetrics:
    """相邻性奖励计算的性能指标数据类。"""
    total_time: float = 0.0
    spatial_time: float = 0.0
    functional_time: float = 0.0
    connectivity_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    computation_count: int = 0

class OptimizedAdjacencyRewardCalculator:
    """
    高性能相邻性奖励计算器。
    
    实现的优化策略：
    1. 向量化计算：使用NumPy广播替代循环
    2. 稀疏矩阵：减少内存使用和计算开销
    3. 多级缓存：LRU缓存 + 预计算缓存
    4. 批量处理：合并多种相邻性的计算
    5. 索引优化：预计算索引映射
    """
    
    def __init__(self, config: RLConfig, placeable_depts: List[str], 
                 travel_times_matrix, constraint_manager):
        """
        初始化优化的相邻性奖励计算器。
        
        Args:
            config: 配置对象
            placeable_depts: 可放置科室列表
            travel_times_matrix: 行程时间矩阵
            constraint_manager: 约束管理器
        """
        self.config = config
        self.placeable_depts = placeable_depts
        self.travel_times_matrix = travel_times_matrix
        self.constraint_manager = constraint_manager
        
        # 性能监控
        self.metrics = AdjacencyMetrics()
        self.enable_optimization = getattr(config, 'ENABLE_ADJACENCY_OPTIMIZATION', True)
        
        # 科室索引映射（预计算）
        self.dept_to_idx = {dept: i for i, dept in enumerate(placeable_depts)}
        self.n_depts = len(placeable_depts)
        
        # 初始化优化组件
        self._initialize_optimized_matrices()
        self._initialize_vectorized_indices()
        self._initialize_batch_computation_cache()
        
        logger.info(f"优化相邻性计算器初始化完成：{self.n_depts}个科室，"
                   f"稀疏矩阵优化={'启用' if self.enable_optimization else '禁用'}")
    
    def _initialize_optimized_matrices(self):
        """初始化优化的稀疏矩阵结构。"""
        logger.debug("初始化优化矩阵结构...")
        
        # 初始化稀疏矩阵存储
        self.spatial_adjacency_sparse = None
        self.functional_adjacency_sparse = None
        self.connectivity_adjacency_sparse = None
        
        # 预计算所有相邻性矩阵
        if self.config.ADJACENCY_PRECOMPUTE:
            self._precompute_all_adjacency_matrices()
    
    def _precompute_all_adjacency_matrices(self):
        """预计算所有相邻性矩阵并转换为稀疏格式。"""
        logger.debug("预计算并稀疏化相邻性矩阵...")
        
        # 1. 空间相邻性矩阵
        spatial_matrix = self._compute_spatial_adjacency_matrix()
        if spatial_matrix is not None:
            # 转换为稀疏矩阵（COO格式便于后续处理）
            self.spatial_adjacency_sparse = sp.coo_matrix(spatial_matrix)
            logger.debug(f"空间相邻性稀疏矩阵：{self.spatial_adjacency_sparse.nnz}/{self.n_depts**2} 非零元素")
        
        # 2. 功能相邻性矩阵
        functional_matrix = self._compute_functional_adjacency_matrix()
        if functional_matrix is not None:
            self.functional_adjacency_sparse = sp.coo_matrix(functional_matrix)
            logger.debug(f"功能相邻性稀疏矩阵：{self.functional_adjacency_sparse.nnz}/{self.n_depts**2} 非零元素")
        
        # 3. 连通性相邻性矩阵
        if self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0:
            connectivity_matrix = self._compute_connectivity_adjacency_matrix()
            if connectivity_matrix is not None:
                self.connectivity_adjacency_sparse = sp.coo_matrix(connectivity_matrix)
                logger.debug(f"连通性相邻性稀疏矩阵：{self.connectivity_adjacency_sparse.nnz}/{self.n_depts**2} 非零元素")
    
    def _compute_spatial_adjacency_matrix(self) -> Optional[np.ndarray]:
        """计算空间相邻性矩阵。"""
        try:
            # 获取有效节点
            valid_nodes = [node for node in self.placeable_depts 
                          if node in self.travel_times_matrix.columns]
            
            if len(valid_nodes) < 2:
                logger.warning("可放置节点数量过少，跳过空间相邻性计算")
                return None
            
            # 构建距离矩阵（向量化）
            node_indices = [self.travel_times_matrix.columns.get_loc(node) for node in valid_nodes]
            distance_submatrix = self.travel_times_matrix.iloc[node_indices, node_indices].values
            
            # 计算分位数阈值
            upper_triangle = np.triu(distance_submatrix, k=1)
            valid_distances = upper_triangle[upper_triangle > 0]
            
            if len(valid_distances) == 0:
                logger.warning("无有效距离数据，使用默认空间相邻性矩阵")
                return np.eye(len(valid_nodes))
            
            threshold = np.percentile(valid_distances, self.config.ADJACENCY_PERCENTILE_THRESHOLD * 100)
            
            # 向量化生成相邻性矩阵
            adjacency_matrix = (distance_submatrix <= threshold).astype(np.float32)
            np.fill_diagonal(adjacency_matrix, 0)  # 自身不相邻
            
            # 确保对称性
            adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
            
            return adjacency_matrix
            
        except Exception as e:
            logger.error(f"空间相邻性矩阵计算失败：{e}")
            return None
    
    def _compute_functional_adjacency_matrix(self) -> Optional[np.ndarray]:
        """计算功能相邻性矩阵。"""
        try:
            adjacency_matrix = np.zeros((self.n_depts, self.n_depts), dtype=np.float32)
            
            # 向量化计算功能偏好
            for i, dept1 in enumerate(self.placeable_depts):
                generic1 = dept1.split('_')[0]
                for j, dept2 in enumerate(self.placeable_depts):
                    if i != j:
                        generic2 = dept2.split('_')[0]
                        preference_score = self._get_functional_preference(generic1, generic2)
                        adjacency_matrix[i, j] = preference_score
            
            return adjacency_matrix
            
        except Exception as e:
            logger.error(f"功能相邻性矩阵计算失败：{e}")
            return None
    
    def _compute_connectivity_adjacency_matrix(self) -> Optional[np.ndarray]:
        """计算连通性相邻性矩阵。"""
        try:
            adjacency_matrix = np.zeros((self.n_depts, self.n_depts), dtype=np.float32)
            
            # 获取距离矩阵
            valid_nodes = [node for node in self.placeable_depts 
                          if node in self.travel_times_matrix.columns]
            
            if len(valid_nodes) < 2:
                return adjacency_matrix
            
            # 构建距离矩阵
            node_indices = [self.travel_times_matrix.columns.get_loc(node) for node in valid_nodes]
            distance_matrix = self.travel_times_matrix.iloc[node_indices, node_indices].values
            
            # 计算连通性阈值
            valid_distances = distance_matrix[distance_matrix > 0]
            if len(valid_distances) == 0:
                return adjacency_matrix
            
            connectivity_threshold = np.percentile(
                valid_distances, 
                self.config.CONNECTIVITY_DISTANCE_PERCENTILE * 100
            )
            
            # 向量化计算连通性权重
            mask = (distance_matrix > 0) & (distance_matrix <= connectivity_threshold)
            adjacency_matrix[mask] = np.exp(-distance_matrix[mask] / connectivity_threshold)
            
            return adjacency_matrix
            
        except Exception as e:
            logger.error(f"连通性相邻性矩阵计算失败：{e}")
            return None
    
    def _initialize_vectorized_indices(self):
        """初始化向量化计算所需的索引结构。"""
        # 预计算所有可能的科室对索引
        dept_pairs = []
        for i in range(self.n_depts):
            for j in range(i + 1, self.n_depts):
                dept_pairs.append((i, j))
        
        self.dept_pairs = np.array(dept_pairs)
        self.n_pairs = len(dept_pairs)
        
        logger.debug(f"预计算科室对索引：{self.n_pairs}个科室对")
    
    def _initialize_batch_computation_cache(self):
        """初始化批量计算缓存。"""
        # 布局模式缓存（用于识别常见的科室组合模式）
        self.layout_pattern_cache = {}
        
        # 预计算缓存
        self.precomputed_rewards = {}
        
        logger.debug("批量计算缓存初始化完成")
    
    def _get_functional_preference(self, generic1: str, generic2: str) -> float:
        """获取功能相邻性偏好分数。"""
        preferences = self.config.MEDICAL_ADJACENCY_PREFERENCES
        
        # 正向偏好
        if generic1 in preferences and generic2 in preferences[generic1]:
            return preferences[generic1][generic2]
        
        # 反向偏好
        if generic2 in preferences and generic1 in preferences[generic2]:
            return preferences[generic2][generic1]
        
        # 默认无偏好
        return 0.0
    
    @lru_cache(maxsize=1000)
    def calculate_adjacency_reward_optimized(self, placed_depts_tuple: Tuple[str, ...]) -> Dict[str, float]:
        """
        优化的相邻性奖励计算方法。
        
        Args:
            placed_depts_tuple: 已放置科室的元组（用于缓存）
            
        Returns:
            Dict[str, float]: 包含各维度奖励的字典
        """
        start_time = time.time()
        self.metrics.computation_count += 1
        
        # 过滤有效科室
        placed_depts = [dept for dept in placed_depts_tuple if dept is not None]
        
        if len(placed_depts) < 2:
            return {
                'spatial_reward': 0.0,
                'functional_reward': 0.0,
                'connectivity_reward': 0.0,
                'total_reward': 0.0
            }
        
        # 检查缓存
        cache_key = tuple(sorted(placed_depts))
        if cache_key in self.precomputed_rewards:
            self.metrics.cache_hits += 1
            return self.precomputed_rewards[cache_key]
        
        self.metrics.cache_misses += 1
        
        # 获取科室索引
        try:
            dept_indices = [self.dept_to_idx[dept] for dept in placed_depts]
        except KeyError as e:
            logger.warning(f"科室不在索引映射中：{e}")
            return {'spatial_reward': 0.0, 'functional_reward': 0.0, 'connectivity_reward': 0.0, 'total_reward': 0.0}
        
        # 向量化计算各维度奖励
        if self.enable_optimization and hasattr(self, 'spatial_adjacency_sparse'):
            rewards = self._calculate_vectorized_rewards(dept_indices)
        else:
            # 降级到原有算法
            rewards = self._calculate_legacy_rewards(placed_depts)
        
        # 计算总奖励
        total_reward = (
            self.config.SPATIAL_ADJACENCY_WEIGHT * rewards['spatial_reward'] +
            self.config.FUNCTIONAL_ADJACENCY_WEIGHT * rewards['functional_reward'] +
            self.config.CONNECTIVITY_ADJACENCY_WEIGHT * rewards['connectivity_reward']
        ) * self.config.ADJACENCY_REWARD_BASE
        
        rewards['total_reward'] = total_reward
        
        # 缓存结果
        self.precomputed_rewards[cache_key] = rewards
        
        # 更新性能指标
        self.metrics.total_time += time.time() - start_time
        
        return rewards
    
    def _calculate_vectorized_rewards(self, dept_indices: List[int]) -> Dict[str, float]:
        """使用向量化方法计算相邻性奖励。"""
        n_depts = len(dept_indices)
        
        # 创建索引网格
        i_indices, j_indices = np.meshgrid(dept_indices, dept_indices, indexing='ij')
        mask = i_indices < j_indices  # 只计算上三角
        
        valid_i = i_indices[mask]
        valid_j = j_indices[mask]
        
        rewards = {
            'spatial_reward': 0.0,
            'functional_reward': 0.0,
            'connectivity_reward': 0.0
        }
        
        if len(valid_i) == 0:
            return rewards
        
        # 1. 空间相邻性奖励（向量化）
        if self.spatial_adjacency_sparse is not None:
            spatial_start = time.time()
            
            # 使用稀疏矩阵的高效索引
            spatial_values = self.spatial_adjacency_sparse.tocsr()[valid_i, valid_j].A1
            rewards['spatial_reward'] = np.mean(spatial_values)
            
            self.metrics.spatial_time += time.time() - spatial_start
        
        # 2. 功能相邻性奖励（向量化）
        if self.functional_adjacency_sparse is not None:
            functional_start = time.time()
            
            functional_values = self.functional_adjacency_sparse.tocsr()[valid_i, valid_j].A1
            # 分别处理正向和负向偏好
            positive_values = functional_values[functional_values > 0]
            negative_values = functional_values[functional_values < 0]
            
            reward = 0.0
            if len(positive_values) > 0:
                reward += np.mean(positive_values)
            if len(negative_values) > 0:
                penalty_multiplier = getattr(self.config, 'ADJACENCY_PENALTY_MULTIPLIER', 1.0)
                reward += np.mean(negative_values) * penalty_multiplier
            
            rewards['functional_reward'] = reward
            
            self.metrics.functional_time += time.time() - functional_start
        
        # 3. 连通性相邻性奖励（向量化）
        if (self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0 and 
            self.connectivity_adjacency_sparse is not None):
            connectivity_start = time.time()
            
            connectivity_values = self.connectivity_adjacency_sparse.tocsr()[valid_i, valid_j].A1
            rewards['connectivity_reward'] = np.mean(connectivity_values)
            
            self.metrics.connectivity_time += time.time() - connectivity_start
        
        return rewards
    
    def _calculate_legacy_rewards(self, placed_depts: List[str]) -> Dict[str, float]:
        """降级算法：使用原有的循环计算方法。"""
        logger.debug("使用降级算法计算相邻性奖励")
        
        rewards = {
            'spatial_reward': 0.0,
            'functional_reward': 0.0,
            'connectivity_reward': 0.0
        }
        
        count = 0
        
        for i, dept1 in enumerate(placed_depts):
            for j, dept2 in enumerate(placed_depts[i+1:], i+1):
                if dept1 in self.dept_to_idx and dept2 in self.dept_to_idx:
                    idx1, idx2 = self.dept_to_idx[dept1], self.dept_to_idx[dept2]
                    
                    # 空间相邻性
                    if hasattr(self, 'spatial_adjacency_sparse') and self.spatial_adjacency_sparse is not None:
                        spatial_score = self.spatial_adjacency_sparse.tocsr()[idx1, idx2]
                        rewards['spatial_reward'] += spatial_score
                    
                    # 功能相邻性
                    if hasattr(self, 'functional_adjacency_sparse') and self.functional_adjacency_sparse is not None:
                        functional_score = self.functional_adjacency_sparse.tocsr()[idx1, idx2]
                        if functional_score > 0:
                            rewards['functional_reward'] += functional_score
                        elif functional_score < 0:
                            penalty_multiplier = getattr(self.config, 'ADJACENCY_PENALTY_MULTIPLIER', 1.0)
                            rewards['functional_reward'] += functional_score * penalty_multiplier
                    
                    # 连通性相邻性
                    if (self.config.CONNECTIVITY_ADJACENCY_WEIGHT > 0 and 
                        hasattr(self, 'connectivity_adjacency_sparse') and 
                        self.connectivity_adjacency_sparse is not None):
                        connectivity_score = self.connectivity_adjacency_sparse.tocsr()[idx1, idx2]
                        rewards['connectivity_reward'] += connectivity_score
                    
                    count += 1
        
        # 计算平均值
        if count > 0:
            for key in rewards:
                rewards[key] /= count
        
        return rewards
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标。"""
        total_calls = self.metrics.cache_hits + self.metrics.cache_misses
        cache_hit_rate = self.metrics.cache_hits / total_calls if total_calls > 0 else 0.0
        
        return {
            'total_computation_time': self.metrics.total_time,
            'avg_computation_time': self.metrics.total_time / max(1, self.metrics.computation_count),
            'spatial_computation_time': self.metrics.spatial_time,
            'functional_computation_time': self.metrics.functional_time,
            'connectivity_computation_time': self.metrics.connectivity_time,
            'cache_hit_rate': cache_hit_rate,
            'total_computations': self.metrics.computation_count,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses
        }
    
    def reset_metrics(self):
        """重置性能指标。"""
        self.metrics = AdjacencyMetrics()
    
    def clear_cache(self):
        """清除所有缓存。"""
        self.calculate_adjacency_reward_optimized.cache_clear()
        self.precomputed_rewards.clear()
        self.layout_pattern_cache.clear()
        logger.info("相邻性奖励计算器缓存已清除")

# 兼容性包装函数
def create_adjacency_calculator(config: RLConfig, placeable_depts: List[str], 
                               travel_times_matrix, constraint_manager) -> OptimizedAdjacencyRewardCalculator:
    """
    创建相邻性奖励计算器的工厂函数。
    
    Args:
        config: 配置对象
        placeable_depts: 可放置科室列表
        travel_times_matrix: 行程时间矩阵
        constraint_manager: 约束管理器
        
    Returns:
        OptimizedAdjacencyRewardCalculator: 优化的相邻性奖励计算器实例
    """
    return OptimizedAdjacencyRewardCalculator(
        config=config,
        placeable_depts=placeable_depts,
        travel_times_matrix=travel_times_matrix,
        constraint_manager=constraint_manager
    )