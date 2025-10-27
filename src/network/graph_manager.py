<<<<<<< HEAD
"""简化的图管理器，用于存储和管理网络图中的节点和边"""

import itertools
import networkx as nx
from src.rl_optimizer.utils.setup import setup_logger
from typing import Dict, Optional, Iterator, List, Tuple, Any

from .node import Node

logger = setup_logger(__name__)


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
        # 添加节点到NetworkX图中，使用节点ID作为图节点
        # 将完整的Node对象作为属性存储
        self._graph.add_node(
            node.id,
            node_obj=node,  # 存储完整的Node对象
            # 冗余存储一些常用属性便于快速访问
            pos=(node.x, node.y, node.z),
            type=node.node_type,
            name=node.name,
            time=node.time
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
=======
"""Manages the network graph structure, node storage, and ID generation."""

import itertools
import networkx as nx
import numpy as np
from loguru import logger
from typing import Dict, Optional, Iterator, List, Tuple, Any

from src.config import graph_config

logger = logger.bind(module=__name__)


class GraphManager:
    """Manages the NetworkX graph and provides node/edge manipulation methods."""

    def __init__(self, id_start: int = 1):
        """Initializes the GraphManager.

        Args:
            id_start: The starting value for the node ID counter.
        """
        self._graph = nx.Graph()
        self._id_counter = itertools.count(start=id_start)
        logger.debug(f"GraphManager initialized with starting ID: {id_start}")

    def generate_node_id(self) -> int:
        """Generates a new unique node ID."""
        return next(self._id_counter)

    def get_current_id_value(self) -> int:
        """Gets the next ID value without consuming it."""
        current_value = next(self._id_counter)
        self._id_counter = itertools.chain([current_value], self._id_counter)
        return current_value

    def add_node(self, node_id: int, **attrs: Any) -> None:
        """Adds a node with its attributes to the graph.

        Args:
            node_id: The unique ID for the node.
            **attrs: A dictionary of attributes for the node.
        """
        self._graph.add_node(node_id, **attrs)
        logger.debug(f"Added node: {node_id} with attributes {attrs}")

    def add_edge(self, node1_id: int, node2_id: int, **attrs: Any) -> None:
        """Adds an edge between two nodes, calculating weight if not provided.

        If 'weight' is not in attrs, it calculates the travel time based on
        Euclidean distance and settings from the configuration.

        Args:
            node1_id: The ID of the first node.
            node2_id: The ID of the second node.
            **attrs: A dictionary of attributes for the edge.
        """
        if self._graph.has_node(node1_id) and self._graph.has_node(node2_id):
            if "weight" not in attrs:
                node1_data = self.get_node_attributes(node1_id)
                node2_data = self.get_node_attributes(node2_id)
                if node1_data and node2_data:
                    dist = np.linalg.norm(
                        np.array(
                            [
                                node1_data["pos_x"],
                                node1_data["pos_y"],
                                node1_data["pos_z"],
                            ]
                        )
                        - np.array(
                            [
                                node2_data["pos_x"],
                                node2_data["pos_y"],
                                node2_data["pos_z"],
                            ]
                        )
                    )
                    geo_config = graph_config.get_geometry_config()
                    scale = geo_config.get("pixel_to_meter_scale", 0.1)
                    speed = geo_config.get("pedestrian_speed", 1.2)
                    attrs["weight"] = (dist * scale) / speed

            self._graph.add_edge(node1_id, node2_id, **attrs)
            logger.debug(f"Added edge: {node1_id} -> {node2_id} with {attrs}")
        else:
            logger.warning(
                f"Attempted to connect non-existent node: {node1_id} or {node2_id}"
            )

    def connect_nodes_by_ids(self, node1_id: int, node2_id: int, **attrs: Any) -> None:
        """Alias for add_edge."""
        self.add_edge(node1_id, node2_id, **attrs)

    def get_node_attributes(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Gets the attribute dictionary of a node."""
        if self._graph.has_node(node_id):
            return self._graph.nodes[node_id]
        return None

    def get_all_nodes_data(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Returns an iterator over all nodes and their data."""
        return self._graph.nodes(data=True)

    def remove_node(self, node_id: int) -> None:
        """Removes a node from the graph."""
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.debug(f"Removed node: {node_id}")
        else:
            logger.warning(f"Attempted to remove non-existent node: {node_id}")

    def get_nodes_by_attribute(
        self, attr_name: str, attr_value: Any
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Gets a list of nodes matching a specific attribute value."""
        return [
            (n, d)
            for n, d in self._graph.nodes(data=True)
            if d.get(attr_name) == attr_value
        ]

    def get_graph(self) -> nx.Graph:
        """Returns the internal NetworkX graph object."""
        return self._graph

    def get_graph_copy(self) -> nx.Graph:
        """Returns a copy of the internal NetworkX graph object."""
        return self._graph.copy()

    def get_next_available_node_id_estimate(self) -> int:
        """Estimates the next available node ID."""
        return self.get_current_id_value()

    def node_count(self) -> int:
        """Returns the total number of nodes."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Returns the total number of edges."""
        return self._graph.number_of_edges()

    def clear(self) -> None:
        """Clears all nodes and edges from the graph."""
        self._graph.clear()
        logger.debug("GraphManager cleared.")

    def __len__(self) -> int:
        """Returns the number of nodes."""
        return self.node_count()

    def __str__(self) -> str:
        """Returns a string representation of the GraphManager."""
        return f"GraphManager(nodes={self.node_count()}, edges={self.edge_count()})"
>>>>>>> dev-refactor
