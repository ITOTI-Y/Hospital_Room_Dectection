"""
Orchestrates the construction of a single-floor network graph.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import networkx as nx
from loguru import logger
from typing import Tuple, List, Optional, Dict, TYPE_CHECKING, Sequence
from scipy.spatial import KDTree
import cv2

if TYPE_CHECKING:
    from .node_creators import BaseNodeCreator

from src.config import graph_config
from .graph_manager import GraphManager
from src.utils.processor import ImageProcessor
from .node_creators import (
    BaseNodeCreator,
    RoomNodeCreator,
    VerticalNodeCreator,
    PedestrianNodeCreator,
    OutsideNodeCreator,
    ConnectionNodeCreator,
)

logger = logger.bind(module=__name__)


class Network:
    """
    Manages the creation of a network graph for a single floor from an image.

    This class is now driven by the new configuration system and is decoupled
    from the old NetworkConfig class.
    """

    def __init__(self, id_generator_start_value: int):
        """
        Initializes the Network orchestrator.

        Args:
            id_generator_start_value: The starting ID for nodes in this network.
        """
        self.image_processor = ImageProcessor()
        self.graph_manager = GraphManager(id_generator_start_value)

        # Node creators are now initialized without passing config objects.
        # They will import and use the new config modules directly.
        self._node_creators: Sequence[BaseNodeCreator] = [
            RoomNodeCreator(self),
            VerticalNodeCreator(self),
            PedestrianNodeCreator(self),
            ConnectionNodeCreator(self),
        ]

        self._outside_node_creator = OutsideNodeCreator(self)

        self._current_image_data: Optional[np.ndarray] = None
        self._id_map: Optional[np.ndarray] = None
        self._image_height: Optional[int] = None
        self._image_width: Optional[int] = None
        self._mask_cache: Dict[str, np.ndarray] = {}

    def _get_mask(
        self,
        identifier: str | List[str],
        is_category: bool,
        apply_morphology: bool = True,
    ) -> np.ndarray:
        """Generates and caches a combined mask for a category or a list of node names."""
        if isinstance(identifier, list):
            key = "_".join(sorted(identifier))
        else:
            key = identifier
        cache_key = f"{key}_{is_category}_{apply_morphology}"

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        if self._image_height is None or self._image_width is None:
            raise RuntimeError("Image dimensions not set. Call _initialize_run first.")

        if is_category:
            if not isinstance(identifier, str):
                raise ValueError("Category identifier must be a string.")
            target_names = graph_config.get_nodes_by_category(identifier)
        else:
            target_names = identifier if isinstance(identifier, list) else [identifier]

        if not target_names:
            return np.zeros((self._image_height, self._image_width), dtype=np.uint8)

        combined_mask = np.zeros(
            (self._image_height, self._image_width), dtype=np.uint8
        )
        node_defs = graph_config.get_node_definitions()

        for name in target_names:
            node_props = node_defs.get(name, {})
            color_rgb = node_props.get("rgb")
            if not color_rgb or self._current_image_data is None:
                continue

            mask = np.all(
                self._current_image_data
                == np.array(color_rgb, dtype=np.uint8).reshape(1, 1, 3),
                axis=2,
            )
            mask = mask.astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        if apply_morphology:
            geometry_config = graph_config.get_geometry_config()
            kernel_size = geometry_config.get("morphology_kernel_size", (5, 5))
            logger.info(
                f"Kernel size before conversion: {kernel_size} (type: {type(kernel_size)})"
            )
            if isinstance(kernel_size, (float, int, str)):
                kernel_size = (int(kernel_size), int(kernel_size))
            combined_mask = self.image_processor.apply_morphology(
                combined_mask, operation="close_open", kernel_size=kernel_size
            )

        self._mask_cache[cache_key] = combined_mask
        return combined_mask

    def _initialize_run(self, image_path: Path, process_outside_nodes: bool) -> None:
        """Loads image, prepares internal data structures for a run."""
        # Load and preprocess image (quantize colors)
        raw_image_data = self.image_processor.load_and_prepare_image(image_path)
        self._current_image_data = self.image_processor.quantize_colors(raw_image_data)

        self._image_height, self._image_width = (
            self.image_processor.get_image_dimensions()
        )

        special_ids = graph_config.get_special_ids()
        self._id_map = np.full(
            (self._image_height, self._image_width),
            special_ids.get("background", -2),
            dtype=np.int32,
        )

        if process_outside_nodes:
            s_config = graph_config.get_super_network_config()
            outside_types = s_config.get("outside_types", [])
            if outside_types:
                for outside_type_name in outside_types:
                    outside_mask = self._outside_node_creator._create_mask_for_node(
                        self._current_image_data,
                        outside_type_name,
                        apply_morphology=True,
                    )
                    if outside_mask is not None:
                        self._id_map[outside_mask != 0] = special_ids.get("outside", -1)

    def _create_all_node_types(
        self, z_level: float, process_outside_nodes: bool, floor_num: int
    ) -> None:
        """Iterates through node creators to populate the graph."""
        if self._current_image_data is None or self._id_map is None:
            raise RuntimeError(
                "Network run not initialized properly. Call _initialize_run first."
            )

        # Execute creators in the predefined order
        for creator in self._node_creators:
            logger.info(f"Running creator: {creator.__class__.__name__}")
            creator.create_nodes(
                self._current_image_data, self._id_map, z_level, floor_num
            )

        if process_outside_nodes:
            logger.info(
                f"Running creator: {self._outside_node_creator.__class__.__name__}"
            )
            self._outside_node_creator.create_nodes(
                self._current_image_data, self._id_map, z_level, floor_num
            )

    def _refine_door_connections(self, z_level: float) -> None:
        """
        Refines the connections for all door nodes based on their 'door_type'.
        - EXTERIOR doors connect to the nearest 'Corridor'.
        - INTERIOR doors connect to the nearest non-door 'CONNECTOR' and 'Corridor'.
        - ROOM doors connect to the nearest 'FIXED' or 'SLOT' and 'Corridor'.
        - Doors with a null 'door_type' are removed.
        """
        if self.graph_manager.node_count() == 0:
            return

        all_nodes = list(self.graph_manager.get_all_nodes_data())
        door_nodes = [
            (nid, data)
            for nid, data in all_nodes
            if data.get("category") == "CONNECTOR"
            and data.get("name") == "Door"
            and data.get("pos_z") == z_level
        ]

        if not door_nodes:
            return

        # Helper to create a KD-tree for a list of nodes
        def build_kdtree(nodes):
            if not nodes:
                return None, []
            positions = np.array(
                [(n_data["pos_x"], n_data["pos_y"]) for _, n_data in nodes]
            )
            return KDTree(positions), nodes

        # Prepare KD-trees for target node types
        corridor_nodes = [
            (nid, data)
            for nid, data in all_nodes
            if data.get("name") == "Corridor" and data.get("pos_z") == z_level
        ]
        corridor_tree, corridor_nodes_list = build_kdtree(corridor_nodes)

        room_nodes = [
            (nid, data)
            for nid, data in all_nodes
            if data.get("category") in ["FIXED", "SLOT"]
            and data.get("pos_z") == z_level
        ]
        room_tree, room_nodes_list = build_kdtree(room_nodes)

        other_connector_nodes = [
            (nid, data)
            for nid, data in all_nodes
            if data.get("category") == "CONNECTOR"
            and data.get("name") != "Door"
            and data.get("pos_z") == z_level
        ]
        other_connector_tree, other_connector_nodes_list = build_kdtree(
            other_connector_nodes
        )

        nodes_to_remove = []

        for door_id, door_data in door_nodes:
            door_type = door_data.get("door_type")
            door_pos = (door_data["pos_x"], door_data["pos_y"])

            if door_type == "EXTERIOR":
                if corridor_tree:
                    _, idx = corridor_tree.query(door_pos)
                    nearest_node_id, _ = corridor_nodes_list[idx]
                    self.graph_manager.connect_nodes_by_ids(door_id, nearest_node_id)

            elif door_type == "INTERIOR":
                if corridor_tree:
                    _, idx = corridor_tree.query(door_pos)
                    nearest_node_id, _ = corridor_nodes_list[idx]
                    self.graph_manager.connect_nodes_by_ids(door_id, nearest_node_id)
                if other_connector_tree:
                    _, idx = other_connector_tree.query(door_pos)
                    nearest_node_id, _ = other_connector_nodes_list[idx]
                    self.graph_manager.connect_nodes_by_ids(door_id, nearest_node_id)

            elif door_type == "ROOM":
                # Connect to the colliding room node(s)
                colliding_ids = door_data.get("colliding_node_ids", [])
                for nid in colliding_ids:
                    node_data = self.graph_manager.get_node_attributes(nid)
                    if node_data and node_data.get("category") in ["FIXED", "SLOT"]:
                        self.graph_manager.connect_nodes_by_ids(door_id, nid)

                # Connect to the nearest passageway
                combined_passageway_nodes = (
                    corridor_nodes_list + other_connector_nodes_list
                )
                passageway_tree, passageway_nodes_list = build_kdtree(
                    combined_passageway_nodes
                )
                if passageway_tree:
                    _, idx = passageway_tree.query(door_pos)
                    nearest_passageway_id, _ = passageway_nodes_list[idx]
                    self.graph_manager.connect_nodes_by_ids(
                        door_id, nearest_passageway_id
                    )
            else:
                nodes_to_remove.append(door_id)

        # Remove nodes marked for deletion
        for node_id in nodes_to_remove:
            self.graph_manager.remove_node(node_id)
        if nodes_to_remove:
            logger.info(f"Removed {len(nodes_to_remove)} doors with null door_type.")

    def run(
        self,
        image_path: Path,
        z_level: float = 0.0,
        process_outside_nodes: bool = False,
        floor_num: int = 0,
    ) -> Tuple[nx.Graph, int, int, int]:
        """
        Executes the full network generation pipeline.
        Args:
            process_outside_nodes: If True, detailed mesh nodes for outside areas are created.
                                   If False, outside areas are only marked in id_map (for door typing)
                                   but no actual outside mesh nodes are generated by OutsideNodeCreator.
        """
        logger.info(
            f"Processing floor: {image_path} at z={z_level}, process_outside_nodes={process_outside_nodes}"
        )

        self._initialize_run(image_path, process_outside_nodes)

        self._create_all_node_types(z_level, process_outside_nodes, floor_num)

        self._refine_door_connections(z_level)

        logger.info(f"Finished floor. Nodes: {self.graph_manager.node_count()}")

        if self._image_width is None or self._image_height is None:
            raise RuntimeError("Image dimensions not set.")

        return (
            self.graph_manager.get_graph_copy(),
            self._image_width,
            self._image_height,
            self.graph_manager.get_next_available_node_id_estimate(),
        )
