# 完善_calculate_adjacency_reward方法 - 实现方案

## 实现概述

本文档详细说明了动态相邻性奖励机制的具体实现方法，包括分步骤实现指南、关键代码示例、模块接口定义、第三方库配置和错误处理方案。

## 实现架构总览

### 核心模块划分

```
src/algorithms/adjacency/
├── __init__.py                          # 包初始化
├── adjacency_analyzer.py               # 相邻性分析器主模块
├── spatial_calculator.py               # 空间相邻性计算器
├── functional_calculator.py            # 功能相邻性计算器
├── travel_time_calculator.py           # 通行时间相邻性计算器
├── adjacency_cache.py                  # 相邻性缓存管理
├── reward_integrator.py                # 奖励集成器
└── utils.py                            # 工具函数
```

### 接口定义

```python
# 主要接口协议
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class AdjacencyCalculator(ABC):
    """相邻性计算器基类接口"""
    
    @abstractmethod
    def calculate_adjacency_matrix(self) -> np.ndarray:
        """计算相邻性关系矩阵"""
        pass
    
    @abstractmethod
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """获取指定槽位的相邻槽位列表"""
        pass
    
    @abstractmethod
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """计算布局的相邻性得分"""
        pass
```

## 分步骤实现指南

### 步骤1：创建相邻性分析器模块

#### 1.1 创建adjacency包

```bash
# 创建目录结构
mkdir -p src/algorithms/adjacency
touch src/algorithms/adjacency/__init__.py
```

#### 1.2 实现AdjacencyAnalyzer主类

**文件路径：** `src/algorithms/adjacency/adjacency_analyzer.py`

```python
# src/algorithms/adjacency/adjacency_analyzer.py

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger
from .spatial_calculator import SpatialAdjacencyCalculator
from .functional_calculator import FunctionalAdjacencyCalculator
from .travel_time_calculator import TravelTimeAdjacencyCalculator
from .adjacency_cache import AdjacencyCache
from .reward_integrator import RewardIntegrator

logger = setup_logger(__name__)

class AdjacencyAnalyzer:
    """
    相邻性分析器主类
    
    协调各种相邻性计算器，提供统一的相邻性分析接口
    """
    
    def __init__(self, 
                 config: RLConfig,
                 travel_times: np.ndarray,
                 placeable_slots: List[str],
                 placeable_departments: List[str],
                 resolved_pathways: List[Dict[str, Any]]):
        """
        初始化相邻性分析器
        
        Args:
            config: RL配置对象
            travel_times: 通行时间矩阵 (n_slots x n_slots)
            placeable_slots: 可放置槽位列表
            placeable_departments: 可放置科室列表
            resolved_pathways: 已解析的就医流程数据
        """
        self.config = config
        self.travel_times = travel_times
        self.placeable_slots = placeable_slots
        self.placeable_departments = placeable_departments
        self.resolved_pathways = resolved_pathways
        
        # 初始化各计算器
        self.spatial_calc = SpatialAdjacencyCalculator(config, travel_times)
        self.functional_calc = FunctionalAdjacencyCalculator(
            config, resolved_pathways, placeable_departments)
        self.travel_time_calc = TravelTimeAdjacencyCalculator(config, travel_times)
        
        # 初始化缓存和奖励集成器
        self.cache = AdjacencyCache(config.ADJACENCY_CACHE_SIZE)
        self.reward_integrator = RewardIntegrator(config)
        
        # 预计算相邻性矩阵
        if config.ADJACENCY_PRECOMPUTE:
            self._precompute_adjacency_matrices()
            
        logger.info(f"相邻性分析器初始化完成，槽位数={len(placeable_slots)}")
    
    def _precompute_adjacency_matrices(self):
        """预计算各种相邻性关系矩阵"""
        logger.info("开始预计算相邻性关系矩阵...")
        
        # 预计算空间相邻性矩阵
        self.spatial_adjacency_matrix = self.spatial_calc.calculate_adjacency_matrix()
        
        # 预计算功能相邻性矩阵
        self.functional_adjacency_matrix = self.functional_calc.calculate_adjacency_matrix()
        
        # 预计算通行时间相邻性矩阵
        self.travel_time_adjacency_matrix = self.travel_time_calc.calculate_adjacency_matrix()
        
        logger.info("相邻性关系矩阵预计算完成")
    
    def calculate_adjacency_reward(self, layout: List[str]) -> float:
        """
        计算布局的综合相邻性奖励
        
        Args:
            layout: 当前布局（科室名称列表，None表示空槽位）
            
        Returns:
            综合相邻性奖励值
        """
        if not self.config.ENABLE_ADJACENCY_REWARD:
            return 0.0
            
        # 检查缓存
        layout_key = tuple(layout)
        cached_reward = self.cache.get_reward(layout_key)
        if cached_reward is not None:
            return cached_reward
        
        # 计算各维度相邻性得分
        spatial_score = self._calculate_spatial_score(layout)
        functional_score = self._calculate_functional_score(layout)
        connectivity_score = self._calculate_connectivity_score(layout)
        
        # 综合评分
        total_reward = self.reward_integrator.integrate_scores(
            spatial_score, functional_score, connectivity_score)
        
        # 缓存结果
        self.cache.put_reward(layout_key, total_reward)
        
        # 记录统计信息
        self._log_adjacency_statistics(layout, spatial_score, 
                                     functional_score, connectivity_score, total_reward)
        
        return total_reward
    
    def _calculate_spatial_score(self, layout: List[str]) -> float:
        """计算空间相邻性得分"""
        if hasattr(self, 'spatial_adjacency_matrix'):
            return self.spatial_calc.calculate_score_from_matrix(
                layout, self.spatial_adjacency_matrix)
        else:
            return self.spatial_calc.calculate_adjacency_score(layout)
    
    def _calculate_functional_score(self, layout: List[str]) -> float:
        """计算功能相邻性得分"""
        if hasattr(self, 'functional_adjacency_matrix'):
            return self.functional_calc.calculate_score_from_matrix(
                layout, self.functional_adjacency_matrix)
        else:
            return self.functional_calc.calculate_adjacency_score(layout)
    
    def _calculate_connectivity_score(self, layout: List[str]) -> float:
        """计算连通性相邻性得分"""
        if hasattr(self, 'travel_time_adjacency_matrix'):
            return self.travel_time_calc.calculate_score_from_matrix(
                layout, self.travel_time_adjacency_matrix)
        else:
            return self.travel_time_calc.calculate_adjacency_score(layout)
    
    def _log_adjacency_statistics(self, layout: List[str], spatial: float, 
                                functional: float, connectivity: float, total: float):
        """记录相邻性统计信息"""
        if logger.isEnabledFor(logging.DEBUG):
            placed_count = sum(1 for dept in layout if dept is not None)
            logger.debug(f"相邻性奖励计算：已放置{placed_count}个科室，"
                        f"空间得分={spatial:.4f}，功能得分={functional:.4f}，"
                        f"连通性得分={connectivity:.4f}，总奖励={total:.4f}")
    
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """获取指定槽位的相邻槽位（综合多种算法）"""
        # 综合多种算法的相邻性判定结果
        spatial_adjacent = self.spatial_calc.get_adjacent_slots(slot_index)
        travel_time_adjacent = self.travel_time_calc.get_adjacent_slots(slot_index)
        
        # 合并并去重
        all_adjacent = list(set(spatial_adjacent + travel_time_adjacent))
        
        return all_adjacent
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("相邻性分析器缓存已清空")
```

### 步骤2：实现空间相邻性计算器

**文件路径：** `src/algorithms/adjacency/spatial_calculator.py`

```python
# src/algorithms/adjacency/spatial_calculator.py

import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import networkx as nx

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger
from .utils import AdjacencyCalculator

logger = setup_logger(__name__)

class SpatialAdjacencyCalculator(AdjacencyCalculator):
    """
    空间相邻性计算器
    
    基于通行时间的空间距离计算相邻性关系
    """
    
    def __init__(self, config: RLConfig, travel_times: np.ndarray):
        self.config = config
        self.travel_times = travel_times
        self.n_slots = len(travel_times)
        
        # 动态计算相邻性参数
        self.percentile_threshold = config.ADJACENCY_PERCENTILE_THRESHOLD
        self.k_nearest = config.ADJACENCY_K_NEAREST or max(2, int(np.sqrt(self.n_slots)))
        
        logger.debug(f"空间相邻性计算器初始化：{self.n_slots}个槽位，"
                    f"分位数阈值={self.percentile_threshold}，K近邻={self.k_nearest}")
    
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
            # 排除自身（距离为0）
            other_times = times_from_slot[times_from_slot > 0]
            threshold_time = np.percentile(other_times, self.percentile_threshold * 100)
            
            # 计算相邻性强度（基于相对距离）
            for other_slot_idx in range(self.n_slots):
                if slot_idx != other_slot_idx:
                    travel_time = times_from_slot[other_slot_idx]
                    if travel_time <= threshold_time and travel_time > 0:
                        # 相邻性强度与距离成反比
                        strength = 1.0 - (travel_time / threshold_time)
                        adjacency_matrix[slot_idx, other_slot_idx] = strength
        
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
        
        # 排除自身，计算分位数阈值
        other_times = [times_from_slot[i] for i in range(self.n_slots) if i != slot_index]
        threshold_time = np.percentile(other_times, self.percentile_threshold * 100)
        
        # 筛选相邻槽位
        adjacent_slots = [i for i in range(self.n_slots) 
                         if i != slot_index and 
                         times_from_slot[i] <= threshold_time and 
                         times_from_slot[i] > 0]
        
        return adjacent_slots
    
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """
        计算布局的空间相邻性得分
        
        Args:
            layout: 当前布局（科室名称列表）
            
        Returns:
            空间相邻性得分
        """
        if not hasattr(self, '_adjacency_matrix'):
            self._adjacency_matrix = self.calculate_adjacency_matrix()
            
        return self.calculate_score_from_matrix(layout, self._adjacency_matrix)
    
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
        # 构建通行时间特征矩阵
        time_features = self.travel_times.copy()
        
        # 动态计算DBSCAN参数
        eps_percentile = self.config.ADJACENCY_CLUSTER_EPS_PERCENTILE
        min_samples = self.config.ADJACENCY_MIN_CLUSTER_SIZE
        
        # 计算eps参数（基于数据分布）
        distances = pdist(time_features)
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
```

### 步骤3：实现功能相邻性计算器

**文件路径：** `src/algorithms/adjacency/functional_calculator.py`

```python
# src/algorithms/adjacency/functional_calculator.py

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger
from .utils import AdjacencyCalculator

logger = setup_logger(__name__)

class FunctionalAdjacencyCalculator(AdjacencyCalculator):
    """
    功能相邻性计算器
    
    基于医疗流程数据计算科室间的功能相邻性
    """
    
    def __init__(self, config: RLConfig, resolved_pathways: List[Dict[str, Any]], 
                 placeable_departments: List[str]):
        self.config = config
        self.resolved_pathways = resolved_pathways
        self.placeable_departments = placeable_departments
        self.dept_to_idx = {dept: i for i, dept in enumerate(placeable_departments)}
        self.n_depts = len(placeable_departments)
        
        # 计算功能相邻性权重矩阵
        self.functional_weights = self._calculate_functional_weights()
        
        logger.debug(f"功能相邻性计算器初始化：{self.n_depts}个科室，"
                    f"{len(self.functional_weights)}个功能相邻关系")
    
    def _calculate_functional_weights(self) -> Dict[Tuple[str, str], float]:
        """
        基于医疗流程计算功能相邻性权重
        
        Returns:
            科室对的功能相邻性权重字典
        """
        functional_weights = defaultdict(float)
        total_pathways_weight = 0.0
        
        # 从就医流程中计算转移权重
        for pathway in self.resolved_pathways:
            path = pathway['path']
            weight = pathway['weight']
            total_pathways_weight += weight
            
            # 计算路径中相邻科室对的权重
            for i in range(len(path) - 1):
                dept_from = path[i]
                dept_to = path[i + 1]
                
                # 只考虑可放置的科室
                if dept_from in self.dept_to_idx and dept_to in self.dept_to_idx:
                    dept_pair = (dept_from, dept_to)
                    functional_weights[dept_pair] += weight
        
        # 归一化权重
        if total_pathways_weight > 0:
            for pair in functional_weights:
                functional_weights[pair] /= total_pathways_weight
        
        logger.debug(f"功能相邻性权重计算完成：{len(functional_weights)}个科室对")
        
        return dict(functional_weights)
    
    def calculate_adjacency_matrix(self) -> np.ndarray:
        """
        计算功能相邻性关系矩阵
        
        Returns:
            functional_matrix: n_depts x n_depts 的功能相邻性矩阵
        """
        functional_matrix = np.zeros((self.n_depts, self.n_depts))
        
        for (dept_from, dept_to), weight in self.functional_weights.items():
            if dept_from in self.dept_to_idx and dept_to in self.dept_to_idx:
                from_idx = self.dept_to_idx[dept_from]
                to_idx = self.dept_to_idx[dept_to]
                functional_matrix[from_idx, to_idx] = weight
                # 功能相邻性通常是双向的
                functional_matrix[to_idx, from_idx] = weight
        
        logger.debug(f"功能相邻性矩阵计算完成，非零元素数={np.count_nonzero(functional_matrix)}")
        
        return functional_matrix
    
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """
        功能相邻性主要应用于科室级别，此方法返回空列表
        
        实际的功能相邻性在calculate_adjacency_score中体现
        """
        return []
    
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """
        计算布局的功能相邻性得分
        
        Args:
            layout: 当前布局（科室名称列表）
            
        Returns:
            功能相邻性得分
        """
        if not hasattr(self, '_functional_matrix'):
            self._functional_matrix = self.calculate_adjacency_matrix()
            
        return self.calculate_score_from_matrix(layout, self._functional_matrix)
    
    def calculate_score_from_matrix(self, layout: List[str], 
                                  functional_matrix: np.ndarray) -> float:
        """
        基于预计算的功能相邻性矩阵计算得分
        
        Args:
            layout: 当前布局
            functional_matrix: 功能相邻性矩阵
            
        Returns:
            功能相邻性得分
        """
        total_functional_score = 0.0
        functional_pairs_count = 0
        
        # 获取已放置的科室及其位置
        placed_depts = [(i, dept) for i, dept in enumerate(layout) if dept is not None]
        
        for i, (slot1, dept1) in enumerate(placed_depts):
            for j, (slot2, dept2) in enumerate(placed_depts):
                if i >= j:  # 避免重复计算
                    continue
                    
                # 检查功能相邻性权重
                if dept1 in self.dept_to_idx and dept2 in self.dept_to_idx:
                    dept1_idx = self.dept_to_idx[dept1]
                    dept2_idx = self.dept_to_idx[dept2]
                    
                    functional_weight = functional_matrix[dept1_idx, dept2_idx]
                    if functional_weight > 0:
                        # 计算空间距离因子（功能相邻的科室应该在空间上也相对较近）
                        spatial_factor = self._calculate_spatial_factor(slot1, slot2, layout)
                        
                        # 功能相邻性得分 = 功能权重 * 空间因子
                        score_contribution = functional_weight * spatial_factor
                        total_functional_score += score_contribution
                        functional_pairs_count += 1
                        
                        logger.debug(f"功能相邻性：{dept1}-{dept2}，"
                                   f"功能权重={functional_weight:.3f}，空间因子={spatial_factor:.3f}，"
                                   f"贡献={score_contribution:.3f}")
        
        # 归一化得分
        if functional_pairs_count > 0:
            normalized_score = total_functional_score / functional_pairs_count
        else:
            normalized_score = 0.0
            
        return normalized_score * self.config.ADJACENCY_REWARD_BASE
    
    def _calculate_spatial_factor(self, slot1: int, slot2: int, layout: List[str]) -> float:
        """
        计算两个槽位之间的空间因子
        
        功能相邻的科室如果在空间上也相近，应该获得更高的奖励
        
        Args:
            slot1, slot2: 槽位索引
            layout: 当前布局
            
        Returns:
            空间因子（0-1之间，越近越高）
        """
        # 这里需要访问通行时间数据，实际实现时需要传入travel_times
        # 暂时使用简化的距离计算
        
        # 计算槽位索引距离（简化方法）
        slot_distance = abs(slot1 - slot2)
        max_distance = len(layout)
        
        # 空间因子与距离成反比
        if max_distance > 0:
            spatial_factor = 1.0 - (slot_distance / max_distance)
        else:
            spatial_factor = 1.0
            
        return max(0.1, spatial_factor)  # 最小保持0.1的基础因子
    
    def get_functional_relationship(self, dept1: str, dept2: str) -> float:
        """
        获取两个科室之间的功能关系强度
        
        Args:
            dept1, dept2: 科室名称
            
        Returns:
            功能关系强度（0-1之间）
        """
        # 检查正向关系
        forward_pair = (dept1, dept2)
        if forward_pair in self.functional_weights:
            return self.functional_weights[forward_pair]
        
        # 检查反向关系
        backward_pair = (dept2, dept1)
        if backward_pair in self.functional_weights:
            return self.functional_weights[backward_pair]
        
        return 0.0
    
    def get_top_functional_pairs(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        获取功能关系最强的科室对
        
        Args:
            top_k: 返回前k个最强关系
            
        Returns:
            按功能关系强度排序的科室对列表
        """
        sorted_pairs = sorted(self.functional_weights.items(), 
                            key=lambda x: x[1], reverse=True)
        
        return [(pair[0], pair[1], weight) for pair, weight in sorted_pairs[:top_k]]
```

### 步骤4：实现通行时间相邻性计算器

**文件路径：** `src/algorithms/adjacency/travel_time_calculator.py`

```python
# src/algorithms/adjacency/travel_time_calculator.py

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from scipy.sparse.csgraph import shortest_path

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger
from .utils import AdjacencyCalculator

logger = setup_logger(__name__)

class TravelTimeAdjacencyCalculator(AdjacencyCalculator):
    """
    通行时间相邻性计算器
    
    基于通行时间构建图结构，分析连通性相邻关系
    """
    
    def __init__(self, config: RLConfig, travel_times: np.ndarray):
        self.config = config
        self.travel_times = travel_times
        self.n_slots = len(travel_times)
        
        # 动态计算图连接参数
        self.k_nearest = config.ADJACENCY_K_NEAREST or max(2, int(np.sqrt(self.n_slots)))
        self.connection_percentile = 0.3  # 连接阈值分位数
        
        # 构建通行时间图
        self.adjacency_graph = self._build_adjacency_graph()
        
        logger.debug(f"通行时间相邻性计算器初始化：{self.n_slots}个槽位，"
                    f"K近邻={self.k_nearest}，图连接数={self.adjacency_graph.number_of_edges()}")
    
    def _build_adjacency_graph(self) -> nx.Graph:
        """
        基于通行时间构建邻接图
        
        Returns:
            邻接图对象
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_slots))
        
        # 为每个槽位添加到最近邻居的边
        for slot_idx in range(self.n_slots):
            travel_times_from_slot = self.travel_times[slot_idx]
            
            # 获取k个最近邻居（排除自身）
            sorted_indices = np.argsort(travel_times_from_slot)
            nearest_neighbors = [idx for idx in sorted_indices[1:self.k_nearest+1] 
                               if travel_times_from_slot[idx] > 0]
            
            # 添加边连接
            for neighbor_idx in nearest_neighbors:
                if not graph.has_edge(slot_idx, neighbor_idx):
                    edge_weight = travel_times_from_slot[neighbor_idx]
                    graph.add_edge(slot_idx, neighbor_idx, weight=edge_weight)
        
        logger.debug(f"邻接图构建完成：{graph.number_of_nodes()}个节点，"
                    f"{graph.number_of_edges()}条边")
        
        return graph
    
    def calculate_adjacency_matrix(self) -> np.ndarray:
        """
        基于图结构计算连通性相邻关系矩阵
        
        Returns:
            connectivity_matrix: n_slots x n_slots 的连通性相邻矩阵
        """
        connectivity_matrix = np.zeros((self.n_slots, self.n_slots))
        
        # 计算图中所有最短路径
        try:
            # 使用NetworkX计算最短路径长度
            path_lengths = dict(nx.all_pairs_shortest_path_length(
                self.adjacency_graph, cutoff=3))  # 限制路径长度以提高效率
            
            for source in range(self.n_slots):
                if source in path_lengths:
                    for target, length in path_lengths[source].items():
                        if source != target and length <= 2:  # 直接相邻或间接相邻
                            # 连通性强度与路径长度成反比
                            connectivity_strength = 1.0 / length
                            connectivity_matrix[source, target] = connectivity_strength
                            
        except Exception as e:
            logger.warning(f"图路径计算失败，使用基础连接关系：{e}")
            # 降级方案：使用图的直接边连接
            for edge in self.adjacency_graph.edges(data=True):
                source, target = edge[0], edge[1]
                # 基于边权重计算连通性强度
                edge_weight = edge[2].get('weight', 1.0)
                max_weight = np.max(self.travel_times[self.travel_times > 0])
                connectivity_strength = 1.0 - (edge_weight / max_weight)
                connectivity_matrix[source, target] = connectivity_strength
                connectivity_matrix[target, source] = connectivity_strength
        
        logger.debug(f"连通性相邻矩阵计算完成，平均连接度="
                    f"{np.mean(np.sum(connectivity_matrix > 0, axis=1)):.2f}")
        
        return connectivity_matrix
    
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """
        基于图结构获取相邻槽位
        
        Args:
            slot_index: 槽位索引
            
        Returns:
            相邻槽位索引列表
        """
        if slot_index >= self.n_slots or slot_index not in self.adjacency_graph:
            return []
        
        # 获取图中的直接邻居
        direct_neighbors = list(self.adjacency_graph.neighbors(slot_index))
        
        # 可选：添加二度邻居（间接相邻）
        if self.config.get('INCLUDE_INDIRECT_ADJACENCY', False):
            indirect_neighbors = []
            for neighbor in direct_neighbors:
                second_degree = list(self.adjacency_graph.neighbors(neighbor))
                indirect_neighbors.extend([n for n in second_degree 
                                         if n != slot_index and n not in direct_neighbors])
            
            # 去重并返回
            all_neighbors = list(set(direct_neighbors + indirect_neighbors))
            return all_neighbors
        
        return direct_neighbors
    
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """
        计算布局的连通性相邻得分
        
        Args:
            layout: 当前布局（科室名称列表）
            
        Returns:
            连通性相邻得分
        """
        if not hasattr(self, '_connectivity_matrix'):
            self._connectivity_matrix = self.calculate_adjacency_matrix()
            
        return self.calculate_score_from_matrix(layout, self._connectivity_matrix)
    
    def calculate_score_from_matrix(self, layout: List[str], 
                                  connectivity_matrix: np.ndarray) -> float:
        """
        基于预计算的连通性矩阵计算得分
        
        Args:
            layout: 当前布局
            connectivity_matrix: 连通性相邻矩阵
            
        Returns:
            连通性相邻得分
        """
        total_connectivity_score = 0.0
        connection_count = 0
        
        for slot1 in range(len(layout)):
            dept1 = layout[slot1]
            if dept1 is None:
                continue
                
            for slot2 in range(slot1 + 1, len(layout)):
                dept2 = layout[slot2]
                if dept2 is None:
                    continue
                
                # 获取连通性强度
                connectivity_strength = connectivity_matrix[slot1, slot2]
                if connectivity_strength > 0:
                    # 检查科室间是否有功能相邻偏好
                    preference = self._get_connectivity_preference(dept1, dept2)
                    
                    if preference != 0:
                        score_contribution = preference * connectivity_strength
                        total_connectivity_score += score_contribution
                        connection_count += 1
                        
                        logger.debug(f"连通性相邻：{dept1}-{dept2}，"
                                   f"连通强度={connectivity_strength:.3f}，偏好={preference:.3f}，"
                                   f"贡献={score_contribution:.3f}")
        
        # 归一化得分
        if connection_count > 0:
            normalized_score = total_connectivity_score / connection_count
        else:
            normalized_score = 0.0
            
        return normalized_score * self.config.ADJACENCY_REWARD_BASE
    
    def _get_connectivity_preference(self, dept1: str, dept2: str) -> float:
        """
        获取两个科室之间的连通性偏好
        
        连通性偏好主要考虑需要频繁交流的科室应该保持良好连通性
        
        Args:
            dept1, dept2: 科室名称
            
        Returns:
            连通性偏好值
        """
        # 从配置中获取偏好，如果没有则使用默认值
        preferences = self.config.MEDICAL_ADJACENCY_PREFERENCES
        
        # 检查是否有明确的相邻偏好
        if dept1 in preferences and dept2 in preferences[dept1]:
            return preferences[dept1][dept2] * 0.8  # 连通性权重稍低于直接相邻
        
        if dept2 in preferences and dept1 in preferences[dept2]:
            return preferences[dept2][dept1] * 0.8
        
        # 对于一些通用的连通性需求，给予小的正向偏好
        high_traffic_depts = {'急诊科', '挂号收费', '检验中心', '放射科'}
        if dept1 in high_traffic_depts or dept2 in high_traffic_depts:
            return 0.3  # 高流量科室与其他科室保持良好连通性
        
        return 0.0
    
    def analyze_graph_properties(self) -> Dict[str, Any]:
        """
        分析图的拓扑属性
        
        Returns:
            图属性统计字典
        """
        properties = {
            'nodes': self.adjacency_graph.number_of_nodes(),
            'edges': self.adjacency_graph.number_of_edges(),
            'density': nx.density(self.adjacency_graph),
            'is_connected': nx.is_connected(self.adjacency_graph),
            'average_clustering': nx.average_clustering(self.adjacency_graph),
            'average_shortest_path_length': None
        }
        
        # 计算平均最短路径长度（仅对连通图）
        if properties['is_connected']:
            try:
                properties['average_shortest_path_length'] = \
                    nx.average_shortest_path_length(self.adjacency_graph, weight='weight')
            except Exception as e:
                logger.warning(f"无法计算平均最短路径长度：{e}")
        
        # 计算度分布
        degrees = dict(self.adjacency_graph.degree())
        properties['average_degree'] = np.mean(list(degrees.values()))
        properties['max_degree'] = max(degrees.values()) if degrees else 0
        properties['min_degree'] = min(degrees.values()) if degrees else 0
        
        return properties
```

### 步骤5：实现缓存管理和奖励集成器

**文件路径：** `src/algorithms/adjacency/adjacency_cache.py`

```python
# src/algorithms/adjacency/adjacency_cache.py

from typing import Any, Dict, Tuple, Optional
from collections import OrderedDict
import numpy as np

class AdjacencyCache:
    """
    相邻性计算结果的LRU缓存管理器
    """
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.reward_cache = OrderedDict()
        self.adjacency_cache = OrderedDict()
        
    def get_reward(self, layout_key: Tuple[str]) -> Optional[float]:
        """获取缓存的奖励值"""
        if layout_key in self.reward_cache:
            # 移到最近使用位置
            self.reward_cache.move_to_end(layout_key)
            return self.reward_cache[layout_key]
        return None
    
    def put_reward(self, layout_key: Tuple[str], reward: float):
        """缓存奖励值"""
        if layout_key in self.reward_cache:
            self.reward_cache.move_to_end(layout_key)
        else:
            if len(self.reward_cache) >= self.max_size:
                self.reward_cache.popitem(last=False)
            self.reward_cache[layout_key] = reward
    
    def get_adjacency(self, slot_idx: int) -> Optional[Any]:
        """获取缓存的相邻性数据"""
        if slot_idx in self.adjacency_cache:
            self.adjacency_cache.move_to_end(slot_idx)
            return self.adjacency_cache[slot_idx]
        return None
    
    def put_adjacency(self, slot_idx: int, adjacency_data: Any):
        """缓存相邻性数据"""
        if slot_idx in self.adjacency_cache:
            self.adjacency_cache.move_to_end(slot_idx)
        else:
            if len(self.adjacency_cache) >= self.max_size:
                self.adjacency_cache.popitem(last=False)
            self.adjacency_cache[slot_idx] = adjacency_data
    
    def clear(self):
        """清空缓存"""
        self.reward_cache.clear()
        self.adjacency_cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'reward_cache_size': len(self.reward_cache),
            'adjacency_cache_size': len(self.adjacency_cache),
            'max_size': self.max_size
        }
```

**文件路径：** `src/algorithms/adjacency/reward_integrator.py`

```python
# src/algorithms/adjacency/reward_integrator.py

import numpy as np
from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)

class RewardIntegrator:
    """
    相邻性奖励集成器
    
    整合多维度相邻性得分，生成最终奖励
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # 权重归一化
        total_weight = (config.SPATIAL_ADJACENCY_WEIGHT + 
                       config.FUNCTIONAL_ADJACENCY_WEIGHT + 
                       config.CONNECTIVITY_ADJACENCY_WEIGHT)
        
        if total_weight > 0:
            self.spatial_weight = config.SPATIAL_ADJACENCY_WEIGHT / total_weight
            self.functional_weight = config.FUNCTIONAL_ADJACENCY_WEIGHT / total_weight
            self.connectivity_weight = config.CONNECTIVITY_ADJACENCY_WEIGHT / total_weight
        else:
            self.spatial_weight = self.functional_weight = self.connectivity_weight = 0.0
            
        logger.debug(f"奖励集成器初始化：空间权重={self.spatial_weight:.3f}，"
                    f"功能权重={self.functional_weight:.3f}，连通性权重={self.connectivity_weight:.3f}")
    
    def integrate_scores(self, spatial_score: float, functional_score: float, 
                        connectivity_score: float) -> float:
        """
        整合多维度相邻性得分
        
        Args:
            spatial_score: 空间相邻性得分
            functional_score: 功能相邻性得分
            connectivity_score: 连通性相邻性得分
            
        Returns:
            综合相邻性奖励
        """
        # 加权求和
        integrated_score = (
            self.spatial_weight * spatial_score +
            self.functional_weight * functional_score +
            self.connectivity_weight * connectivity_score
        )
        
        # 应用奖励权重
        final_reward = integrated_score * self.config.ADJACENCY_REWARD_WEIGHT
        
        logger.debug(f"奖励集成：空间={spatial_score:.3f}，功能={functional_score:.3f}，"
                    f"连通={connectivity_score:.3f}，综合={integrated_score:.3f}，"
                    f"最终={final_reward:.3f}")
        
        return final_reward
    
    def apply_bonus_penalty_modifiers(self, base_reward: float, 
                                    layout_properties: dict) -> float:
        """
        应用额外的奖励修正因子
        
        Args:
            base_reward: 基础奖励
            layout_properties: 布局属性字典
            
        Returns:
            修正后的奖励
        """
        modified_reward = base_reward
        
        # 布局完整性奖励
        if layout_properties.get('completion_rate', 0) > 0.9:
            modified_reward *= 1.1  # 10%奖励
            
        # 相邻性平衡奖励
        spatial_score = layout_properties.get('spatial_score', 0)
        functional_score = layout_properties.get('functional_score', 0)
        if abs(spatial_score - functional_score) < 0.2:  # 得分平衡
            modified_reward *= 1.05  # 5%奖励
        
        return modified_reward
```

## 第三方库配置

### 依赖库版本要求

```toml
# pyproject.toml 中添加新依赖

[tool.uv.sources]
scikit-learn = "^1.3.0"
networkx = "^3.1"
scipy = "^1.11.0"
```

### 导入模块清单

```python
# 必需的第三方导入
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import DBSCAN
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional
import logging
```

## 错误处理和边界情况

### 异常处理策略

```python
def safe_adjacency_calculation(func):
    """相邻性计算的异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, IndexError) as e:
            logger.error(f"相邻性计算错误：{e}")
            return 0.0
        except Exception as e:
            logger.error(f"未预期的相邻性计算错误：{e}")
            return 0.0
    return wrapper

# 应用示例
@safe_adjacency_calculation
def calculate_adjacency_reward(self, layout: List[str]) -> float:
    # 实现代码
    pass
```

### 边界情况处理

1. **空布局处理**：当layout中所有元素为None时，返回0.0奖励
2. **单科室布局**：只有一个科室时，返回基础奖励
3. **数据缺失处理**：通行时间数据缺失时使用默认值
4. **配置错误处理**：权重配置无效时使用默认均匀权重
5. **内存不足处理**：缓存大小动态调整，避免内存溢出

## 集成到LayoutEnv

### 修改LayoutEnv._calculate_adjacency_reward方法

```python
# 在src/rl_optimizer/env/layout_env.py中修改
def _calculate_adjacency_reward(self, layout: List[str]) -> float:
    """
    计算相邻性奖励
    """
    if not hasattr(self, 'adjacency_analyzer'):
        # 延迟初始化相邻性分析器
        self._initialize_adjacency_analyzer()
    
    return self.adjacency_analyzer.calculate_adjacency_reward(layout)

def _initialize_adjacency_analyzer(self):
    """初始化相邻性分析器"""
    from src.algorithms.adjacency.adjacency_analyzer import AdjacencyAnalyzer
    
    # 获取通行时间矩阵
    travel_times_matrix = self._get_travel_times_matrix()
    
    self.adjacency_analyzer = AdjacencyAnalyzer(
        config=self.config,
        travel_times=travel_times_matrix,
        placeable_slots=self.placeable_slots,
        placeable_departments=self.placeable_depts,
        resolved_pathways=self.cm.resolved_pathways if hasattr(self.cm, 'resolved_pathways') else []
    )
```

---

**文档版本：** v1.0  
**实现指导：** 产品经理架构师  
**预计实现时间：** 4-6个工作日  
**技术栈：** Python 3.9+, NetworkX 3.1+, Scikit-learn 1.3+, SciPy 1.11+