"""
Orchestrates the construction of a single-floor network graph.
"""

import numpy as np
import networkx as nx
import logging
from typing import Dict, Tuple, Any, List, Optional
from scipy.spatial import KDTree

from src.config import NetworkConfig
from .graph_manager import GraphManager
from .node import Node
from src.image_processing.processor import ImageProcessor
from .node_creators import (  # 节点创建器模块
    BaseNodeCreator,
    RoomNodeCreator,
    VerticalNodeCreator,
    PedestrianNodeCreator,
    OutsideNodeCreator,
    ConnectionNodeCreator
)

logger = logging.getLogger(__name__)

class Network:
    """
    Manages the creation of a network graph for a single floor from an image.

    The process involves:
    1. Loading and preprocessing the image.
    2. Creating different types of nodes (rooms, doors, corridors, etc.) using
       specialized NodeCreator strategies.
    3. Establishing connections between nodes, including specific logic for
       connecting doors to pedestrian/outside mesh areas.
    """

    def __init__(self,
                 config: NetworkConfig,
                 color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]],
                 id_generator_start_value: int):
        """
        Initializes the Network orchestrator.

        Args:
            config: The main configuration object.
            color_map_data: The RGB color to type mapping.
            id_generator_start_value: The starting ID for nodes in this network.
                                      Crucial for `SuperNetwork` to ensure global ID uniqueness
                                      when processing multiple floors in parallel.
        """
        self.config = config
        self.color_map_data = color_map_data  # 传递给创建器的颜色映射数据

        self.image_processor = ImageProcessor(config, color_map_data)
        self.graph_manager = GraphManager(id_generator_start_value)

        # 初始化节点创建器
        self._node_creators: List[BaseNodeCreator] = [
            RoomNodeCreator(config, color_map_data,
                            self.image_processor, self.graph_manager),
            VerticalNodeCreator(config, color_map_data,
                                self.image_processor, self.graph_manager),
            # Pedestrian and Outside creators mark areas in id_map first, then create mesh.
            # Connection creator relies on these id_map markings.
            PedestrianNodeCreator(config, color_map_data,
                                  self.image_processor, self.graph_manager),
            # Outside creator might be conditional based on 'outside' flag in run()
            # 连接创建器应该在房间、垂直、行人、室外区域被标记/创建之后运行
            ConnectionNodeCreator(config, color_map_data,
                                  self.image_processor, self.graph_manager)
        ]

        # have an instance of OutsideNodeCreator for conditional use
        self._outside_node_creator = OutsideNodeCreator(
            config, color_map_data, self.image_processor, self.graph_manager)

        self._current_image_data: Optional[np.ndarray] = None
        self._id_map: Optional[np.ndarray] = None
        self._image_height: Optional[int] = None
        self._image_width: Optional[int] = None

    def _initialize_run(self, image_path: str) -> None:
        """Loads image, prepares internal data structures for a run."""
        # Load and preprocess image (quantize colors)
        raw_image_data = self.image_processor.load_and_prepare_image(
            image_path)
        self._current_image_data = self.image_processor.quantize_colors(
            raw_image_data)

        self._image_height, self._image_width = self.image_processor.get_image_dimensions()

        # id_map stores the ID of the node occupying each pixel, or special area IDs
        self._id_map = np.full((self._image_height, self._image_width),
                               self.config.BACKGROUND_ID_MAP_VALUE, dtype=np.int32)  # 使用int32类型存储ID

        # This ensures that doors of type 'out' are recognized correctly even if the OutsideNodeCreator is not run
        # to create detailed outside mesh nodes.
        if self.config.OUTSIDE_TYPES:  # 检查是否定义了室外类型
            for outside_type_name in self.config.OUTSIDE_TYPES:
                # 从量化图像中为此室外类型创建掩码
                # We don't need full morphology here, just the raw areas.
                # ConnectionNodeCreator will use its own dilation.
                outside_mask = self._outside_node_creator._create_mask_for_type(
                    self._current_image_data,
                    outside_type_name,
                    apply_morphology=True  # 应用基础形态学操作清理掩码
                )
                if outside_mask is not None:
                    self._id_map[outside_mask !=
                                 0] = self.config.OUTSIDE_ID_MAP_VALUE

    def _create_all_node_types(self, z_level: int, process_outside_nodes: bool) -> None:
        """Iterates through node creators to populate the graph."""
        if self._current_image_data is None or self._id_map is None:
            raise RuntimeError(
                "Network run not initialized properly. Call _initialize_run first.")

        # Specific order of creation can be important
        # 1. Rooms and Vertical transport (solid areas with own IDs)
        # 2. Pedestrian areas (mesh + special ID in id_map)
        # 3. Outside areas (mesh + special ID in id_map) - if requested
        # 4. Connections (doors - rely on previously set IDs in id_map)

        # Execute creators in the predefined order
        for creator in self._node_creators:
            logger.info(f"Running creator: {creator.__class__.__name__}")
            creator.create_nodes(self._current_image_data,
                                 self._id_map, z_level)

        # 注意：室外节点创建已被禁用，因为在医院室内布局优化中不需要室外节点
        # 如果将来需要处理室外区域，可以通过配置参数process_outside_nodes来启用
        # if process_outside_nodes:
        #     logger.info(f"运行节点创建器: {self._outside_node_creator.__class__.__name__} (mesh节点)")
        #     self._outside_node_creator.create_nodes(
        #         self._current_image_data, self._id_map, z_level)

    def _connect_doors_to_mesh_areas(self, z_level: int) -> None:
        """
        Connects door nodes (type 'in' or 'out') to the nearest pedestrian/outside
        mesh nodes respectively.
        """
        if not self.graph_manager.get_all_nodes():
            return  # Optimization

        connection_nodes = [
            node for node in self.graph_manager.get_all_nodes().values()
            if node.node_type in self.config.CONNECTION_TYPES and node.z == z_level
            # Only connect these
            and hasattr(node, 'door_type') and (node.door_type == 'in' or node.door_type == 'out')
        ]
        if not connection_nodes:
            return

        pedestrian_mesh_nodes = [
            node for node in self.graph_manager.get_all_nodes().values()
            if node.node_type in self.config.PEDESTRIAN_TYPES and node.z == z_level
        ]
        outside_mesh_nodes = [
            node for node in self.graph_manager.get_all_nodes().values()
            if node.node_type in self.config.OUTSIDE_TYPES and node.z == z_level
        ]

        ped_tree = None
        if pedestrian_mesh_nodes:
            ped_positions = np.array([(p_node.x, p_node.y)
                                     for p_node in pedestrian_mesh_nodes])
            if ped_positions.size > 0:  # Ensure not empty before creating KDTree
                ped_tree = KDTree(ped_positions)

        out_tree = None
        if outside_mesh_nodes:
            out_positions = np.array([(o_node.x, o_node.y)
                                     for o_node in outside_mesh_nodes])
            if out_positions.size > 0:
                out_tree = KDTree(out_positions)

        max_door_to_mesh_distance = self.config.GRID_SIZE * 3

        for conn_node in connection_nodes:
            door_pos_2d = (conn_node.x, conn_node.y)

            if conn_node.door_type == 'in' and ped_tree:
                dist, idx = ped_tree.query(door_pos_2d, k=1)
                # Check if idx is a valid index and not out of bounds (e.g. if ped_tree was empty for some reason)
                if idx < len(pedestrian_mesh_nodes) and dist <= max_door_to_mesh_distance:
                    nearest_ped_node = pedestrian_mesh_nodes[idx]
                    self.graph_manager.connect_nodes_by_ids(
                        conn_node.id, nearest_ped_node.id)

            elif conn_node.door_type == 'out':  # 'out' doors connect to outside AND potentially nearby pedestrian areas
                connected_to_main_outside = False
                if out_tree:
                    dist, idx = out_tree.query(door_pos_2d, k=1)
                    if idx < len(outside_mesh_nodes) and dist <= max_door_to_mesh_distance:
                        nearest_out_node = outside_mesh_nodes[idx]
                        self.graph_manager.connect_nodes_by_ids(
                            conn_node.id, nearest_out_node.id)
                        connected_to_main_outside = True

                # Also check for nearby pedestrian nodes if this 'out' door is on a path
                if ped_tree:
                    # Query for potentially multiple pedestrian nodes within a smaller radius
                    # This is for cases like an exit onto a patio (pedestrian) then to lawn (outside)
                    indices_in_ball = ped_tree.query_ball_point(
                        door_pos_2d, r=np.sqrt(2 * self.config.GRID_SIZE ** 2))
                    for ped_idx in indices_in_ball:
                        if ped_idx < len(pedestrian_mesh_nodes):
                            self.graph_manager.connect_nodes_by_ids(
                                conn_node.id, pedestrian_mesh_nodes[ped_idx].id)
                            # If it connects to outside mesh AND pedestrian mesh, that's fine.
                            # The pathfinding will choose the best route.

    def run(self, image_path: str, z_level: int = 0, process_outside_nodes: bool = False) \
            -> Tuple[nx.Graph, int, int, int]:
        """
        Executes the full network generation pipeline.
        Args:
            process_outside_nodes: If True, detailed mesh nodes for outside areas are created.
                                   If False, outside areas are only marked in id_map (for door typing)
                                   but no actual outside mesh nodes are generated by OutsideNodeCreator.
        """
        logger.info(f"--- Processing floor: {image_path} at z={z_level}, process_outside_nodes={process_outside_nodes} ---")
        
        # self.graph_manager.clear(...) # Only if reusing Network instance, typically not.

        self._initialize_run(image_path) # This now pre-marks OUTSIDE_ID_MAP_VALUE
        
        self._create_all_node_types(z_level, process_outside_nodes) # process_outside_nodes controls mesh creation
        
        self._connect_doors_to_mesh_areas(z_level)
        
        logger.info(f"--- Finished floor. Nodes: {len(self.graph_manager.get_all_nodes())} ---")
        
        if self._image_width is None or self._image_height is None:
            raise RuntimeError("Image dimensions not set.")

        return (
            self.graph_manager.get_graph_copy(),
            self._image_width,
            self._image_height,
            self.graph_manager.get_next_available_node_id_estimate()
        )