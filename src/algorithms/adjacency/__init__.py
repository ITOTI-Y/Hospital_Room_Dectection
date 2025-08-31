"""
医院布局相邻性奖励计算模块

主要功能：
1. 动态相邻性判定算法
2. 多维度相邻性评分
3. 医疗功能导向的奖励计算
4. 高性能缓存机制

使用示例：
    from src.algorithms.adjacency import AdjacencyAnalyzer
    
    analyzer = AdjacencyAnalyzer(config, travel_times, slots, depts, pathways)
    reward = analyzer.calculate_adjacency_reward(layout)
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Hospital Layout Optimization Team"

# 模块级配置
DEFAULT_CONFIG = {
    'ENABLE_ADJACENCY_REWARD': True,
    'ADJACENCY_REWARD_WEIGHT': 0.15,
    'ADJACENCY_CACHE_SIZE': 500,
    'ADJACENCY_PRECOMPUTE': True
}