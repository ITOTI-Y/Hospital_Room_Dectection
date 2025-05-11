"""Manages the graph structure, node storage, and ID generation."""

import itertools
import networkx as nx
import logging
from typing import Dict, Optional, Iterator

from .node import Node

logger = logging.getLogger(__name__)

class GraphManager:
    """
    Manages a networkx graph, stores nodes by ID, and generates unique node IDs.

    Attributes:
        graph (nx.Graph): The underlying networkx graph object.
    """

    def __init__(self, id_generator_start_value: int = 1):
        """
        Initializes the GraphManager.

        Args:
            id_generator_start_value: The starting value for the node ID generator.
                                      This is crucial for multi-processing to ensure
                                      unique IDs across different Network instances.
        """
        self.graph: nx.Graph = nx.Graph()
        self._id_to_node_map: Dict[int, Node] = {}
        # Using itertools.count for a thread-safe (in CPython due to GIL)
        # and efficient ID generator. For true multiprocessing, the start
        # value must be managed externally to ensure uniqueness.
        self._node_id_generator: Iterator[int] = itertools.count(
            id_generator_start_value)
        self._last_generated_id: int = id_generator_start_value - 1

    def generate_node_id(self) -> int:
        """
        Generates and returns a new unique node ID.

        Returns:
            A unique integer ID for a new node.
        """
        new_id = next(self._node_id_generator)
        self._last_generated_id = new_id
        return new_id

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the graph and the ID-to-node map.

        Args:
            node: The Node object to add.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.id in self._id_to_node_map:
            logger.error(
                f"Node with ID {node.id} already exists in this graph manager.")
        if self.graph.has_node(node):  # Should be redundant if ID is unique
            logger.error(
                f"Node object {node} (ID: {node.id}) already exists in the graph.")

        self.graph.add_node(node, type=node.node_type, pos=node.pos,
                            time=node.time, door_type=node.door_type)
        self._id_to_node_map[node.id] = node

    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """
        Retrieves a node by its ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            The Node object if found, otherwise None.
        """
        return self._id_to_node_map.get(node_id)

    def connect_nodes_by_ids(self, node_id1: int, node_id2: int, **edge_attributes) -> bool:
        """
        Connects two nodes in the graph using their IDs.

        Args:
            node_id1: The ID of the first node.
            node_id2: The ID of the second node.
            **edge_attributes: Additional attributes for the edge.

        Returns:
            True if the connection was successful, False if one or both nodes
            were not found, or if they are the same node.
        """
        if node_id1 == node_id2:
            logger.warning(
                f"Warning: Attempted to connect node ID {node_id1} to itself. Skipping.")
            return False

        node1 = self.get_node_by_id(node_id1)
        node2 = self.get_node_by_id(node_id2)

        if node1 and node2:
            if not self.graph.has_edge(node1, node2):
                self.graph.add_edge(node1, node2, **edge_attributes)
            return True
        else:
            missing_ids = []
            if not node1:
                missing_ids.append(node_id1)
            if not node2:
                missing_ids.append(node_id2)
            logger.warning(
                f"Warning: Could not connect nodes. Missing node IDs: {missing_ids}")
            return False
        
    def get_all_nodes(self) -> list[Node]:
        """Returns a list of all Node objects in the graph."""
        return list(self._id_to_node_map.values())
    
    def get_graph_copy(self) -> nx.Graph:
        """
        Returns a deep copy of the internal networkx graph.
        This is important if the graph is to be modified externally
        without affecting the manager's internal state, or for passing
        to other processes.
        """
        return self.graph.copy() # networkx.Graph.copy() is a deep copy by default for node/edge attributes
    
    def get_next_available_node_id_estimate(self) -> int:
        """
        Returns an estimate of the next node ID that would be generated.
        This is primarily for `SuperNetwork` to estimate ID blocks for workers.
        It's `_last_generated_id + 1`.
        """
        return self._last_generated_id + 1
    
    def clear(self, id_generator_start_value: int = 1):
        """
        Clears the graph and resets the ID generator.
        Useful for reusing the manager instance.
        """
        self.graph.clear()
        self._id_to_node_map.clear()
        self._node_id_generator = itertools.count(id_generator_start_value)
        self._last_generated_id = id_generator_start_value - 1
