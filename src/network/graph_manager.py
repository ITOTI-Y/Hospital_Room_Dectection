"""简化的图管理器，用于存储和管理网络图中的节点和边"""

import itertools
import networkx as nx
import logging
from typing import Dict, Optional, Iterator, List, Tuple, Any

from .node import Node

logger = logging.getLogger(__name__)


class GraphManager:
    """
    管理网络图结构、节点存储和ID生成
    
    提供统一的接口来创建、存储、查询节点，并维护NetworkX图结构
    """
    
    def __init__(self, id_start: int = 1):
        """
        初始化图管理器
        
        Args:
            id_start: 节点ID的起始值，默认为1
        """
        self._nodes: Dict[int, Node] = {}
        self._graph = nx.Graph()
        self._id_counter = itertools.count(start=id_start)
        
        logger.debug(f"GraphManager 初始化完成，起始ID: {id_start}")
    
    def get_next_id(self) -> int:
        """获取下一个可用的节点ID"""
        return next(self._id_counter)
    
    def generate_node_id(self) -> int:
        """
        生成新的节点ID（get_next_id的别名）
        
        Returns:
            新的唯一节点ID
        """
        return self.get_next_id()
    
    def get_current_id_value(self) -> int:
        """
        获取当前ID计数器的值（不消耗ID）
        
        Returns:
            当前的ID值（下一个将被分配的ID）
        """
        # 创建计数器的副本以获取当前值
        temp_counter = itertools.tee(self._id_counter, 1)[0]
        current_value = next(temp_counter)
        return current_value
    
    def add_node(self, node: Node) -> None:
        """
        添加节点到图中
        
        Args:
            node: 要添加的节点对象
        """
        self._nodes[node.id] = node
        # 添加节点到NetworkX图中，包含节点属性和完整的Node对象
        self._graph.add_node(
            node.id,
            x=node.x,
            y=node.y,
            z=node.z,
            type=node.node_type,
            name=node.name,
            time=node.time,
            node_obj=node  # 添加完整的Node对象引用
        )
        
        logger.debug(f"添加节点: {node}")
    
    def add_edge(self, node1_id: int, node2_id: int, weight: float = 1.0) -> None:
        """
        在两个节点之间添加边
        
        Args:
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID
            weight: 边的权重（通常是距离或时间）
        """
        if node1_id in self._nodes and node2_id in self._nodes:
            self._graph.add_edge(node1_id, node2_id, weight=weight)
            logger.debug(f"添加边: {node1_id} -> {node2_id} (权重: {weight})")
        else:
            logger.warning(f"尝试连接不存在的节点: {node1_id}, {node2_id}")
    
    def connect_nodes_by_ids(self, node1_id: int, node2_id: int, weight: float = 1.0) -> None:
        """
        通过ID连接两个节点（add_edge的别名）
        
        Args:
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID
            weight: 边的权重
        """
        self.add_edge(node1_id, node2_id, weight)
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """
        根据ID获取节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点对象，如果不存在则返回None
        """
        return self._nodes.get(node_id)
    
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """
        根据ID获取节点（get_node的别名）
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点对象，如果不存在则返回None
        """
        return self.get_node(node_id)
    
    def get_all_nodes(self) -> Dict[int, Node]:
        """获取所有节点"""
        return self._nodes.copy()
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """
        根据类型获取节点列表
        
        Args:
            node_type: 节点类型
            
        Returns:
            指定类型的节点列表
        """
        return [node for node in self._nodes.values() if node.node_type == node_type]
    
    def get_graph(self) -> nx.Graph:
        """获取NetworkX图对象"""
        return self._graph
    
    def get_graph_copy(self) -> nx.Graph:
        """获取NetworkX图对象的副本"""
        return self._graph.copy()
    
    def get_next_available_node_id_estimate(self) -> int:
        """
        获取下一个可用节点ID的估计值
        
        Returns:
            下一个将被分配的节点ID
        """
        return self.get_current_id_value()
    
    def node_count(self) -> int:
        """获取节点总数"""
        return len(self._nodes)
    
    def edge_count(self) -> int:
        """获取边总数"""
        return self._graph.number_of_edges()
    
    def get_node_positions(self) -> Dict[int, Tuple[float, float, float]]:
        """获取所有节点的位置坐标"""
        return {node_id: (node.x, node.y, node.z) for node_id, node in self._nodes.items()}
    
    def clear(self) -> None:
        """清空所有节点和边"""
        self._nodes.clear()
        self._graph.clear()
        logger.debug("GraphManager 已清空")
    
    def __len__(self) -> int:
        """返回节点数量"""
        return len(self._nodes)
    
    def __str__(self) -> str:
        """图管理器的字符串表示"""
        return f"GraphManager(nodes={self.node_count()}, edges={self.edge_count()})"