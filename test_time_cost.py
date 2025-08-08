#!/usr/bin/env python3
"""
验证势函数时间成本计算
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RLConfig
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.env.cost_calculator import CostCalculator  
from src.rl_optimizer.data.cache_manager import CacheManager
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)

def test_time_cost():
    """验证时间成本计算是否正确"""
    
    # 创建配置
    config = RLConfig()
    
    # 创建缓存管理器并加载数据
    cache_manager = CacheManager(config)
    
    # 从缓存管理器获取必要的数据
    travel_times = cache_manager.travel_times_matrix
    # 从travel_times矩阵获取实际的节点名称
    all_nodes = list(travel_times.index)
    # 过滤掉固定节点类型，获取可放置的节点
    fixed_types = config.FIXED_NODE_TYPES
    placeable_slots = [node for node in all_nodes if not any(node.startswith(ft) for ft in fixed_types)]
    placeable_departments = placeable_slots.copy()  # 简化测试，使用相同的列表
    resolved_pathways = cache_manager.get_resolved_pathways()
    
    # 创建成本计算器
    cost_calculator = CostCalculator(
        config, 
        resolved_pathways,
        travel_times,
        placeable_slots,
        placeable_departments
    )
    
    print("=== 测试时间成本计算 ===")
    print(f"槽位数: {len(placeable_slots)}")
    print(f"科室数: {len(placeable_departments)}")
    
    # 测试1：空布局
    empty_layout = [None] * len(placeable_slots)
    cost1 = cost_calculator.calculate_total_cost(empty_layout)
    print(f"\n空布局时间成本: {cost1:.2f}")
    
    # 测试2：放置几个科室
    test_layout = [None] * len(placeable_slots)
    test_layout[0] = placeable_departments[0]  # 第一个科室放在第一个槽位
    test_layout[1] = placeable_departments[1]  # 第二个科室放在第二个槽位
    test_layout[2] = placeable_departments[2]  # 第三个科室放在第三个槽位
    
    cost2 = cost_calculator.calculate_total_cost(test_layout)
    print(f"\n放置3个科室后的时间成本: {cost2:.2f}")
    print(f"已放置科室: {[d for d in test_layout if d is not None][:3]}")
    
    # 测试3：放置更多科室
    for i in range(3, min(10, len(placeable_departments))):
        test_layout[i] = placeable_departments[i]
    
    cost3 = cost_calculator.calculate_total_cost(test_layout)
    print(f"\n放置{sum(1 for d in test_layout if d is not None)}个科室后的时间成本: {cost3:.2f}")
    
    # 检查流线中实际使用的节点
    all_pathway_nodes = set()
    for pathway in resolved_pathways:
        path = pathway.get('path', [])
        all_pathway_nodes.update(path)
    
    print(f"\n流线中使用的节点总数: {len(all_pathway_nodes)}")
    print(f"前10个节点: {list(all_pathway_nodes)[:10]}")
    
    # 找出流线中使用的可放置节点
    pathway_placeable = [node for node in all_pathway_nodes if not any(node.startswith(ft) for ft in fixed_types)]
    print(f"\n流线中的可放置节点数: {len(pathway_placeable)}")
    print(f"前10个可放置节点: {pathway_placeable[:10]}")
    
    # 使用流线中实际的可放置节点进行测试
    if pathway_placeable:
        real_layout = [None] * len(placeable_slots)
        for i in range(min(5, len(pathway_placeable))):
            real_layout[i] = pathway_placeable[i]
        
        real_cost = cost_calculator.calculate_total_cost(real_layout)
        print(f"\n使用流线中的节点放置5个科室后的时间成本: {real_cost:.2f}")
        print(f"已放置: {[d for d in real_layout if d is not None]}")
    
    # 显示行程时间矩阵的一些样本值
    print("\n=== 行程时间矩阵样本 ===")
    sample_nodes = list(travel_times.index)[:5]
    print(f"节点样本: {sample_nodes}")
    for i in range(min(3, len(sample_nodes))):
        for j in range(min(3, len(sample_nodes))):
            if i != j:
                time = travel_times.loc[sample_nodes[i], sample_nodes[j]]
                print(f"{sample_nodes[i]} -> {sample_nodes[j]}: {time:.2f}秒")

if __name__ == "__main__":
    test_time_cost()