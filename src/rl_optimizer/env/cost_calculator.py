# src/rl_optimizer/env/cost_calculator.py

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix
from typing import List, Dict, Tuple, Any
from collections import defaultdict, OrderedDict
import itertools

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger, load_pickle, save_pickle

logger = setup_logger(__name__)

class LRUCache:
    """
    简单的LRU缓存实现，用于限制缓存大小
    """
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: tuple) -> Any:
        """获取缓存值，并将其移到最近使用位置"""
        if key not in self.cache:
            return None
        # 移到最近使用位置
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: tuple, value: Any):
        """设置缓存值"""
        if key in self.cache:
            # 如果已存在，移到最近使用位置
            self.cache.move_to_end(key)
        else:
            # 如果缓存已满，删除最久未使用的项
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self.cache)


class CostCalculator:
    """
    一个高效的成本计算器，用于评估给定布局下的加权总通行时间。
    它通过预计算一个稀疏矩阵来实现高性能计算。
    """

    def __init__(self, 
                 config: RLConfig, 
                 resolved_pathways: List[Dict[str, Any]], 
                 travel_times: pd.DataFrame, 
                 placeable_slots: List[str],
                 placeable_departments: List[str]):
        """
        初始化成本计算器。

        Args:
            config (RLConfig): RL优化器的配置对象。
            resolved_pathways (List[Dict]): 已解析的流线字典列表。
            travel_times (pd.DataFrame): 纯通行时间矩阵，其行列为原始节点名。
            placeable_slots (List[str]): 可用物理槽位的原始节点名称列表。
            placeable_departments (List[str]): 所有需要被布局的科室的完整列表。
        """
        self.config = config
        self.travel_times = travel_times
        self.placeable_slots = placeable_slots
        self.num_slots = len(placeable_slots)
        
        self.dept_list = sorted(placeable_departments)
        self.pair_to_col, self.num_dept_pairs = self._create_dept_pair_mapping()
        
        self.M, self.W, self.pathway_to_process_id = self._build_cost_matrices(resolved_pathways)
        
        # 初始化LRU缓存，限制缓存大小以防止内存泄漏
        cache_size = getattr(config, 'COST_CACHE_SIZE', 1000)
        self._cost_cache = LRUCache(max_size=cache_size)
        self._time_vector_cache = LRUCache(max_size=cache_size // 2)
        
        logger.info(f"成本计算器初始化完成，LRU缓存大小: {cache_size}")

    def _create_dept_pair_mapping(self) -> Tuple[Dict[Tuple[str, str], int], int]:
        """创建科室对到列索引的映射。"""
        dept_pairs = list(itertools.permutations(self.dept_list, 2))
        return {pair: i for i, pair in enumerate(dept_pairs)}, len(dept_pairs)
    
    def _build_cost_matrices(self, pathways: List[Dict[str, Any]]) -> Tuple[csr_matrix, np.ndarray, List[str]]:
        """构建成本矩阵。"""
        num_pathways = len(pathways)
        pathway_to_process_id = [p['process_id'] for p in pathways]
        
        M_dok = dok_matrix((num_pathways, self.num_dept_pairs), dtype=np.float32)
        W_arr = np.zeros(num_pathways, dtype=np.float32)

        for i, p_data in enumerate(pathways):
            path = p_data['path']
            W_arr[i] = p_data['weight']
            for j in range(len(path) - 1):
                pair = (path[j], path[j+1])
                if pair in self.pair_to_col:
                    M_dok[i, self.pair_to_col[pair]] += 1
        
        return M_dok.tocsr(), W_arr, pathway_to_process_id
    
    def calculate_total_cost(self, layout: List[str]) -> float:
        """
        给定一个布局，高效计算总加权通行时间（使用LRU缓存优化）。

        Args:
            layout (List[str]): 一个表示当前布局的科室名称列表。
                               其索引对应于 self.placeable_nodes 的索引。

        Returns:
            float: 计算出的总加权成本。
        """
        # 创建布局的元组作为缓存键
        layout_key = tuple(layout)
        
        # 检查缓存
        cached_cost = self._cost_cache.get(layout_key)
        if cached_cost is not None:
            return cached_cost
        
        # 计算成本
        time_vector = self._get_time_vector(layout)
        time_per_pathway = self.M.dot(time_vector)
        total_cost = time_per_pathway.dot(self.W)
        total_cost = float(total_cost)
        
        # 存入缓存
        self._cost_cache.put(layout_key, total_cost)
        
        return total_cost
    
    def calculate_per_process_cost(self, layout: List[str]) -> Dict[str, float]:
        """
        评估每个就医流程模板的单独通行时间（未加权）。
        """
        time_vector = self._get_time_vector(layout)
        time_per_pathway = self.M.dot(time_vector)
        
        process_costs = defaultdict(float)
        # 遍历所有流线，将它们的时间累加到对应的流程模板上
        for i, process_id in enumerate(self.pathway_to_process_id):
            process_costs[process_id] += time_per_pathway[i]
            
        return dict(process_costs)
    
    def _get_time_vector(self, layout: List[str]) -> np.ndarray:
        """
        内部辅助函数，根据布局计算所有科室对的通行时间向量（使用LRU缓存优化）。
        
        Args:
            layout: 科室名称列表。可以是：
                   1. 完整布局（长度=槽位数，包含None）
                   2. 仅已放置科室的列表（不包含None）
        """
        # 创建布局的元组作为缓存键
        layout_key = tuple(layout)
        
        # 检查缓存
        cached_vector = self._time_vector_cache.get(layout_key)
        if cached_vector is not None:
            return cached_vector
        
        time_vector = np.zeros(self.num_dept_pairs, dtype=np.float32)

        # 判断layout是完整布局还是仅已放置科室
        if len(layout) == self.num_slots:
            # 完整布局：索引对应槽位
            dept_to_slot_node = {dept: self.placeable_slots[i] for i, dept in enumerate(layout) if dept is not None}
        else:
            # 仅已放置科室：需要为每个科室分配一个虚拟槽位
            # 使用科室名本身作为节点名（简化处理）
            dept_to_slot_node = {dept: dept for dept in layout if dept is not None}

        # 遍历所有需要计算时间的科室对
        for pair, col_idx in self.pair_to_col.items():
            dept_from, dept_to = pair
            
            # 查找科室所在的原始节点名
            node_from = dept_to_slot_node.get(dept_from)
            node_to = dept_to_slot_node.get(dept_to)
            
            # 如果两个科室都已放置，则从通行时间矩阵中查找时间
            if node_from is not None and node_to is not None:
                try:
                    time_vector[col_idx] = self.travel_times.loc[node_from, node_to]
                except KeyError as e:
                    # 错误处理：如果找不到节点对，使用默认值
                    default_penalty = getattr(self.config, 'DEFAULT_PENALTY', 1000.0)
                    logger.warning(f"找不到节点对 {node_from}->{node_to} 的行程时间，使用默认值{default_penalty}")
                    time_vector[col_idx] = default_penalty
        
        # 存入缓存
        self._time_vector_cache.put(layout_key, time_vector)
        
        return time_vector
    
    def clear_cache(self):
        """清空所有缓存"""
        self._cost_cache.clear()
        self._time_vector_cache.clear()
        logger.info("成本计算器缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'cost_cache_size': self._cost_cache.size(),
            'time_vector_cache_size': self._time_vector_cache.size()
        }