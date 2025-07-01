# src/rl_optimizer/env/cost_calculator.py

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import itertools

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger, load_pickle, save_pickle

logger = setup_logger(__name__)

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
        给定一个布局，高效计算总加权通行时间。

        Args:
            layout (List[str]): 一个表示当前布局的科室名称列表。
                               其索引对应于 self.placeable_nodes 的索引。

        Returns:
            float: 计算出的总加权成本。
        """
        time_vector = self._get_time_vector(layout)
        time_per_pathway = self.M.dot(time_vector)
        total_cost = time_per_pathway.dot(self.W)
        return float(total_cost)
    
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
        内部辅助函数，根据布局计算所有科室对的通行时间向量。
        """
        time_vector = np.zeros(self.num_dept_pairs, dtype=np.float32)

        # 创建一个从科室到其当前所在槽位（原始节点名）的映射
        # layout 的索引是槽位索引，值是科室名
        # self.placeable_slots 的索引是槽位索引，值是原始节点名
        dept_to_slot_node = {dept: self.placeable_slots[i] for i, dept in enumerate(layout) if dept is not None}

        # 遍历所有需要计算时间的科室对
        for pair, col_idx in self.pair_to_col.items():
            dept_from, dept_to = pair
            
            # 查找科室所在的原始节点名
            node_from = dept_to_slot_node.get(dept_from)
            node_to = dept_to_slot_node.get(dept_to)
            
            # 如果两个科室都已放置，则从通行时间矩阵中查找时间
            if node_from is not None and node_to is not None:
                time_vector[col_idx] = self.travel_times.loc[node_from, node_to]
                
        return time_vector