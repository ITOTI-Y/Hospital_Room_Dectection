"""
空间相邻性计算器

基于通行时间的空间距离计算相邻性关系
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger
from .utils import AdjacencyCalculator, MatrixBasedCalculator, safe_adjacency_calculation, calculate_distance_percentile

logger = setup_logger(__name__)


class SpatialAdjacencyCalculator(MatrixBasedCalculator):
    """
    空间相邻性计算器
    
    基于通行时间的空间距离计算相邻性关系
    """
    
    def _initialize(self, travel_times: np.ndarray):
        """
        初始化空间相邻性计算器
        
        Args:
            travel_times: 通行时间矩阵 (n_slots x n_slots)
        """
        self.travel_times = travel_times
        self.n_slots = len(travel_times)
        
        # 动态计算相邻性参数
        self.percentile_threshold = self.config.ADJACENCY_PERCENTILE_THRESHOLD
        self.k_nearest = self.config.ADJACENCY_K_NEAREST or max(2, int(np.sqrt(self.n_slots)))
        
        self._initialized = True
        
        logger.debug(f"空间相邻性计算器初始化：{self.n_slots}个槽位，"
                    f"分位数阈值={self.percentile_threshold}，K近邻={self.k_nearest}")
    
    @safe_adjacency_calculation
    def calculate_adjacency_matrix(self) -> np.ndarray:
        """
        计算空间相邻性关系矩阵
        
        Returns:
            adjacency_matrix: n_slots x n_slots 的相邻性强度矩阵
        """
        adjacency_matrix = np.zeros((self.n_slots, self.n_slots))
        
        for slot_idx in range(self.n_slots):
            # 获取该槽位到其他所有槽位的通行时间
            times_from_slot = self.travel_times[slot_idx]
            
            # 计算分位数阈值
            threshold_time = calculate_distance_percentile(times_from_slot, self.percentile_threshold)
            
            # 计算相邻性强度（基于相对距离）
            for other_slot_idx in range(self.n_slots):
                if slot_idx != other_slot_idx:
                    travel_time = times_from_slot[other_slot_idx]
                    if travel_time <= threshold_time and travel_time > 0:
                        # 相邻性强度与距离成反比
                        strength = 1.0 - (travel_time / threshold_time)
                        adjacency_matrix[slot_idx, other_slot_idx] = max(0.0, strength)
        
        logger.debug(f"空间相邻性矩阵计算完成，平均相邻槽位数="
                    f"{np.mean(np.sum(adjacency_matrix > 0, axis=1)):.2f}")
        
        return adjacency_matrix
    
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """
        获取指定槽位的相邻槽位列表
        
        Args:
            slot_index: 槽位索引
            
        Returns:
            相邻槽位索引列表
        """
        if slot_index >= self.n_slots:
            return []
            
        # 获取该槽位的通行时间
        times_from_slot = self.travel_times[slot_index]
        
        # 计算分位数阈值
        threshold_time = calculate_distance_percentile(times_from_slot, self.percentile_threshold)
        
        # 筛选相邻槽位
        adjacent_slots = [i for i in range(self.n_slots) 
                         if i != slot_index and 
                         times_from_slot[i] <= threshold_time and 
                         times_from_slot[i] > 0]
        
        return adjacent_slots
    
    @safe_adjacency_calculation
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """
        计算布局的空间相邻性得分
        
        Args:
            layout: 当前布局（科室名称列表）
            
        Returns:
            空间相邻性得分
        """
        if not self.validate_layout(layout):
            return 0.0
            
        adjacency_matrix = self.get_or_compute_matrix()
        return self.calculate_score_from_matrix(layout, adjacency_matrix)
    
    def calculate_score_from_matrix(self, layout: List[str], 
                                  adjacency_matrix: np.ndarray) -> float:
        """
        基于预计算的相邻性矩阵计算得分
        
        Args:
            layout: 当前布局
            adjacency_matrix: 相邻性关系矩阵
            
        Returns:
            空间相邻性得分
        """
        total_score = 0.0
        adjacency_count = 0
        
        for slot_idx, dept_name in enumerate(layout):
            if dept_name is None:
                continue
                
            for other_slot_idx, other_dept_name in enumerate(layout):
                if other_dept_name is None or slot_idx == other_slot_idx:
                    continue
                
                # 获取相邻性强度
                adjacency_strength = adjacency_matrix[slot_idx, other_slot_idx]
                if adjacency_strength > 0:
                    # 检查科室偏好
                    preference = self._get_department_preference(dept_name, other_dept_name)
                    
                    if preference != 0:
                        score_contribution = preference * adjacency_strength
                        total_score += score_contribution
                        adjacency_count += 1
                        
                        if logger.isEnabledFor(10):  # DEBUG级别
                            logger.debug(f"空间相邻性：{dept_name}-{other_dept_name}，"
                                       f"强度={adjacency_strength:.3f}，偏好={preference:.3f}，"
                                       f"贡献={score_contribution:.3f}")
        
        # 归一化得分
        if adjacency_count > 0:
            normalized_score = total_score / adjacency_count
        else:
            normalized_score = 0.0
            
        return normalized_score * self.config.ADJACENCY_REWARD_BASE
    
    def _get_department_preference(self, dept1: str, dept2: str) -> float:
        """
        获取两个科室之间的相邻偏好值
        
        Args:
            dept1, dept2: 科室名称
            
        Returns:
            相邻偏好值（正数表示偏好相邻，负数表示避免相邻，0表示无偏好）
        """
        # 从配置中获取医疗功能相邻性偏好
        preferences = self.config.MEDICAL_ADJACENCY_PREFERENCES
        
        # 检查正向偏好
        if dept1 in preferences and dept2 in preferences[dept1]:
            return preferences[dept1][dept2]
        
        # 检查反向偏好
        if dept2 in preferences and dept1 in preferences[dept2]:
            return preferences[dept2][dept1]
        
        # 无特殊偏好
        return 0.0
    
    def calculate_cluster_based_adjacency(self) -> Dict[int, List[int]]:
        """
        基于密度聚类的区域相邻性分析
        
        Returns:
            聚类相邻性映射字典
        """
        try:
            # 构建通行时间特征矩阵
            time_features = self.travel_times.copy()
            
            # 动态计算DBSCAN参数
            eps_percentile = self.config.ADJACENCY_CLUSTER_EPS_PERCENTILE
            min_samples = self.config.ADJACENCY_MIN_CLUSTER_SIZE
            
            # 计算eps参数（基于数据分布）
            distances = pdist(time_features)
            if len(distances) == 0:
                return {}
            
            eps = np.percentile(distances, eps_percentile * 100)
            
            # 执行DBSCAN聚类
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(time_features)
            
            # 构建相邻性映射
            adjacency_map = {}
            for slot_idx in range(self.n_slots):
                cluster_id = cluster_labels[slot_idx]
                if cluster_id != -1:  # 非噪声点
                    cluster_members = [i for i, label in enumerate(cluster_labels) 
                                     if label == cluster_id and i != slot_idx]
                    adjacency_map[slot_idx] = cluster_members
                else:
                    adjacency_map[slot_idx] = []
            
            logger.debug(f"聚类相邻性分析完成：eps={eps:.2f}，"
                        f"聚类数={len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}，"
                        f"噪声点数={list(cluster_labels).count(-1)}")
            
            return adjacency_map
        
        except Exception as e:
            logger.warning(f"聚类相邻性分析失败：{e}")
            return {}
    
    def get_adjacency_statistics(self) -> Dict[str, float]:
        """
        获取空间相邻性统计信息
        
        Returns:
            统计信息字典
        """
        try:
            adjacency_matrix = self.get_or_compute_matrix()
            
            # 计算基础统计
            non_zero_count = np.count_nonzero(adjacency_matrix)
            total_pairs = self.n_slots * (self.n_slots - 1)  # 排除对角线
            
            stats = {
                'total_slots': self.n_slots,
                'adjacency_pairs': non_zero_count,
                'adjacency_density': non_zero_count / total_pairs if total_pairs > 0 else 0.0,
                'avg_adjacency_strength': np.mean(adjacency_matrix[adjacency_matrix > 0]) if non_zero_count > 0 else 0.0,
                'max_adjacency_strength': np.max(adjacency_matrix),
                'avg_neighbors_per_slot': np.mean(np.sum(adjacency_matrix > 0, axis=1))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取空间相邻性统计信息失败：{e}")
            return {}