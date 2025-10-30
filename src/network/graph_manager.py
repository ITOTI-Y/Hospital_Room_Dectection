"""Manages the network graph structure, node storage, and ID generation."""

import itertools
import networkx as nx
import numpy as np
from loguru import logger
from typing import Dict, Optional, List, Tuple, Any
from networkx.classes.reportviews import NodeDataView

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

    def get_all_nodes_data(self) -> NodeDataView:
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
