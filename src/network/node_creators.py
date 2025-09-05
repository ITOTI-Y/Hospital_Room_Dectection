"""
Defines strategies for creating different types of nodes in the network.

This module uses the Strategy design pattern where each node type (Room,
Connection, Pedestrian, etc.) has its own creator class derived from a
base class.
"""

import abc
import cv2
import numpy as np
from src.rl_optimizer.utils.setup import setup_logger
from scipy.spatial import KDTree
from typing import Dict, Tuple, List, Any, Optional

from src.config import NetworkConfig
from .node import Node
from .graph_manager import GraphManager
from src.image_processing.processor import ImageProcessor

logger = setup_logger(__name__)


class BaseNodeCreator(abc.ABC):
    """
    Abstract base class for node creators.
    """

    def __init__(self,
                 config: NetworkConfig,
                 color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]],
                 image_processor: ImageProcessor,
                 graph_manager: GraphManager):
        self.config = config
        self.color_map_data = color_map_data
        self.image_processor = image_processor
        self.graph_manager = graph_manager
        self.types_map_name_to_rgb: Dict[str, Tuple[int, int, int]] = \
            {details['name']: rgb for rgb, details in color_map_data.items()}
        self.types_map_name_to_time: Dict[str, float] = \
            {details['name']: details.get('time', 1.0)
             for rgb, details in color_map_data.items()}

    def _get_color_rgb_by_name(self, type_name: str) -> Optional[Tuple[int, int, int]]:
        return self.types_map_name_to_rgb.get(type_name)

    def _get_time_by_name(self, type_name: str) -> float:
        return self.types_map_name_to_time.get(type_name, self.config.PEDESTRIAN_TIME)

    def _get_ename_by_name(self, type_name: str) -> Optional[str]:
        return self.color_map_data.get(self._get_color_rgb_by_name(type_name), {}).get('e_name')

    def _get_code_by_name(self, type_name: str) -> Optional[Dict[str, Any]]:
        return self.color_map_data.get(self._get_color_rgb_by_name(type_name), {}).get('code')

    def _create_mask_for_type(self,
                              image_data: np.ndarray,
                              target_type_name: str,
                              apply_morphology: bool = True,
                              morphology_operation: str = 'close_open',
                              morphology_kernel_size: Optional[Tuple[int, int]] = None
                              ) -> Optional[np.ndarray]:
        """Creates a binary mask for a single specified node type."""
        color_rgb = self._get_color_rgb_by_name(target_type_name)
        if color_rgb is None:
            logger.warning(f"Warning: Color for type '{target_type_name}' not found. Cannot create mask.")
            return None

        mask = np.all(image_data == np.array(
            color_rgb, dtype=np.uint8).reshape(1, 1, 3), axis=2)
        mask = mask.astype(np.uint8) * 255

        if apply_morphology:
            kernel_size = morphology_kernel_size or self.config.MORPHOLOGY_KERNEL_SIZE
            mask = self.image_processor.apply_morphology(
                mask,
                operation=morphology_operation,
                kernel_size=kernel_size
            )
        return mask

    def _find_connected_components(self, mask: np.ndarray, connectivity: int = 4) \
            -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        return cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    @abc.abstractmethod
    def create_nodes(self,
                     processed_image_data: np.ndarray,
                     id_map: np.ndarray,
                     z_level: int):
        pass

class RoomNodeCreator(BaseNodeCreator):
    """Creates nodes for room-type areas."""
    def create_nodes(self, processed_image_data: np.ndarray, id_map: np.ndarray, z_level: int):
        target_room_types = self.config.ROOM_TYPES
        if not target_room_types: return

        for room_type_name in target_room_types:
            mask = self._create_mask_for_type(processed_image_data, room_type_name)
            if mask is None: continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1: continue

            node_time = self._get_time_by_name(room_type_name)
            e_name = self._get_ename_by_name(room_type_name)
            code = self._get_code_by_name(room_type_name)

            for i in range(1, retval):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < self.config.AREA_THRESHOLD: continue

                centroid_x, centroid_y = centroids[i]
                position = (int(centroid_x), int(centroid_y), z_level)
                node_id = self.graph_manager.generate_node_id()
                room_node = Node(node_id=node_id, node_type=room_type_name,
                                x=position[0], y=position[1], z=position[2],
                                time=node_time, area=area, e_name=e_name, code=code)
                self.graph_manager.add_node(room_node)
                id_map[labels == i] = room_node.id

class VerticalNodeCreator(BaseNodeCreator):
    """Creates nodes for vertical transport areas (stairs, elevators, escalators)."""
    def create_nodes(self, processed_image_data: np.ndarray, id_map: np.ndarray, z_level: int):
        target_vertical_types = self.config.VERTICAL_TYPES
        if not target_vertical_types: return

        for vertical_type_name in target_vertical_types:
            mask = self._create_mask_for_type(processed_image_data, vertical_type_name)
            if mask is None: continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1: continue

            node_time = self._get_time_by_name(vertical_type_name)
            e_name = self._get_ename_by_name(vertical_type_name)
            code = self._get_code_by_name(vertical_type_name)

            for i in range(1, retval):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < self.config.AREA_THRESHOLD: continue

                centroid_x, centroid_y = centroids[i]
                position = (int(centroid_x), int(centroid_y), z_level)
                node_id = self.graph_manager.generate_node_id()
                v_node = Node(node_id=node_id, node_type=vertical_type_name,
                              x=position[0], y=position[1], z=position[2],
                              time=node_time, area=area, e_name=e_name, code=code)
                self.graph_manager.add_node(v_node)
                id_map[labels == i] = v_node.id

class MeshBasedNodeCreator(BaseNodeCreator): # New base for Pedestrian and Outside
    """Base class for creators that generate a mesh of nodes within areas."""

    def _create_mesh_nodes_for_mask(self,
                                    mask: np.ndarray,
                                    region_type_name: str,
                                    id_map: np.ndarray, # Pass id_map to update
                                    id_map_value_for_area: int, # Value to mark the area in id_map
                                    z_level: int,
                                    node_time: float,
                                    e_name: Optional[str],
                                    code: Optional[str],
                                    grid_size_multiplier: int):
        """Helper to create mesh nodes within a given mask and connect them."""
        # First, mark the entire area in id_map with the special area identifier
        id_map[mask != 0] = id_map_value_for_area

        retval, labels, stats, _ = self._find_connected_components(mask, connectivity=8)
        grid_size = self.config.GRID_SIZE * grid_size_multiplier
        # Estimated area for a single mesh node
        mesh_node_area = float(1)


        for i in range(1, retval): # For each connected component
            component_area = stats[i, cv2.CC_STAT_AREA]
            if component_area < self.config.AREA_THRESHOLD: # Ensure component itself is large enough
                continue

            x_stat, y_stat, w_stat, h_stat, _ = stats[i]

            gx = np.arange(x_stat + grid_size / 2, x_stat + w_stat, grid_size) # Center points in grid cells
            gy = np.arange(y_stat + grid_size / 2, y_stat + h_stat, grid_size)
            if len(gx) == 0 or len(gy) == 0: continue # Avoid empty grid

            grid_points_x, grid_points_y = np.meshgrid(gx, gy)

            grid_points_y_int = grid_points_y.astype(int).clip(0, mask.shape[0] - 1)
            grid_points_x_int = grid_points_x.astype(int).clip(0, mask.shape[1] - 1)
            
            valid_mask_indices = labels[grid_points_y_int, grid_points_x_int] == i
            
            valid_x_coords = grid_points_x[valid_mask_indices]
            valid_y_coords = grid_points_y[valid_mask_indices]

            component_nodes: List[Node] = []
            for vx, vy in zip(valid_x_coords, valid_y_coords):
                pos = (int(vx), int(vy), z_level)
                node_id = self.graph_manager.generate_node_id()
                mesh_node = Node(node_id=node_id, node_type=region_type_name,
                                x=pos[0], y=pos[1], z=pos[2],
                                time=node_time, area=mesh_node_area,
                                e_name=e_name, code=code)
                self.graph_manager.add_node(mesh_node)
                component_nodes.append(mesh_node)
                # Optionally, mark the exact grid cell in id_map with the mesh_node.id
                # id_map[int(vy-grid_size/2):int(vy+grid_size/2), int(vx-grid_size/2):int(vx+grid_size/2)] = mesh_node.id
                # For now, the broader area is already marked.

            if not component_nodes or len(component_nodes) < 2:
                continue

            node_positions_2d = np.array([(node.x, node.y) for node in component_nodes])
            kdtree = KDTree(node_positions_2d)
            # Max distance to connect (diagonal of a grid cell, plus a small tolerance)
            max_distance_connect = np.sqrt(2) * grid_size * 1.05

            for j, current_node in enumerate(component_nodes):
                # Query for k-nearest, then filter by distance
                # k=9 includes self + 8 neighbors in a square grid
                distances, indices_k_nearest = kdtree.query(
                    (current_node.x, current_node.y),
                    k=min(len(component_nodes), self.config.MESH_NODE_CONNECTIVITY_K), # Ensure k is not > num_points
                    distance_upper_bound=max_distance_connect
                )

                for dist_val, neighbor_idx in zip(distances, indices_k_nearest):
                    if neighbor_idx >= len(component_nodes) or dist_val > max_distance_connect :
                        continue # Out of bounds or too far

                    neighbor_node = component_nodes[neighbor_idx]
                    if current_node.id == neighbor_node.id:
                        continue
                    
                    self.graph_manager.connect_nodes_by_ids(current_node.id, neighbor_node.id)

class PedestrianNodeCreator(MeshBasedNodeCreator):
    """Creates mesh nodes for pedestrian areas (e.g., corridors)."""
    def create_nodes(self, processed_image_data: np.ndarray, id_map: np.ndarray, z_level: int):
        target_pedestrian_types = self.config.PEDESTRIAN_TYPES
        if not target_pedestrian_types: return

        for ped_type_name in target_pedestrian_types:
            e_name = self._get_ename_by_name(ped_type_name)
            code = self._get_code_by_name(ped_type_name)
            mask = self._create_mask_for_type(processed_image_data, ped_type_name)
            if mask is None: continue
            
            self._create_mesh_nodes_for_mask(
                mask=mask,
                region_type_name=ped_type_name,
                id_map=id_map,
                id_map_value_for_area=self.config.PEDESTRIAN_ID_MAP_VALUE,
                z_level=z_level,
                node_time=self.config.PEDESTRIAN_TIME,
                grid_size_multiplier=1,
                e_name=e_name,
                code=code
            )

class OutsideNodeCreator(MeshBasedNodeCreator):
    """Creates mesh nodes for outside areas."""
    def create_nodes(self, processed_image_data: np.ndarray, id_map: np.ndarray, z_level: int):
        target_outside_types = self.config.OUTSIDE_TYPES
        if not target_outside_types: return

        for outside_type_name in target_outside_types: # Should typically be just one '室外'
            mask = self._create_mask_for_type(processed_image_data, outside_type_name)
            if mask is None: continue
            
            self._create_mesh_nodes_for_mask(
                mask=mask,
                region_type_name=outside_type_name,
                id_map=id_map,
                id_map_value_for_area=self.config.OUTSIDE_ID_MAP_VALUE,
                z_level=z_level,
                node_time=self._get_time_by_name(outside_type_name) * self.config.OUTSIDE_MESH_TIMES_FACTOR,
                grid_size_multiplier=self.config.OUTSIDE_MESH_TIMES_FACTOR
            )

class ConnectionNodeCreator(BaseNodeCreator):
    """Creates nodes for connections (e.g., doors) and links them to adjacent areas."""
    def create_nodes(self, processed_image_data: np.ndarray, id_map: np.ndarray, z_level: int):
        target_connection_types = self.config.CONNECTION_TYPES
        if not target_connection_types: return

        pass_through_ids_in_id_map = [self.config.BACKGROUND_ID_MAP_VALUE]
        dilation_kernel_np = np.ones(self.config.CONNECTION_DILATION_KERNEL_SIZE, np.uint8)

        for conn_type_name in target_connection_types:
            mask = self._create_mask_for_type(processed_image_data, conn_type_name)
            e_name = self._get_ename_by_name(conn_type_name)
            code = self._get_code_by_name(conn_type_name)
            if mask is None: continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1: continue

            for i in range(1, retval): # For each door component
                area = float(stats[i, cv2.CC_STAT_AREA])
                # Doors can be smaller, adjust threshold if needed, e.g., AREA_THRESHOLD / 4
                if area < self.config.AREA_THRESHOLD / 10 and area < 5: # Allow very small doors
                    continue

                centroid_x, centroid_y = centroids[i]
                position = (int(centroid_x), int(centroid_y), z_level)

                node_id = self.graph_manager.generate_node_id()
                conn_node = Node(node_id=node_id, node_type=conn_type_name,
                                x=position[0], y=position[1], z=position[2],
                                time=self.config.CONNECTION_TIME, area=area,
                                e_name=e_name, code=code)
                self.graph_manager.add_node(conn_node)

                component_mask_pixels = (labels == i)
                id_map[component_mask_pixels] = conn_node.id # Mark door pixels with its own ID

                # Dilate the door component mask to find neighboring regions/nodes in id_map
                # Need to convert boolean mask `component_mask_pixels` to uint8 for dilate
                uint8_component_mask = component_mask_pixels.astype(np.uint8) * 255
                dilated_component_mask = cv2.dilate(uint8_component_mask, dilation_kernel_np, iterations=1)
                
                neighbor_ids_in_map = np.unique(id_map[dilated_component_mask != 0])

                # Determine door type
                is_connected_to_outside = self.config.OUTSIDE_ID_MAP_VALUE in neighbor_ids_in_map
                is_connected_to_pedestrian = self.config.PEDESTRIAN_ID_MAP_VALUE in neighbor_ids_in_map

                if is_connected_to_outside:
                    conn_node.door_type = 'out'
                elif is_connected_to_pedestrian:
                    # If it connects to pedestrian and NOT to outside, it's an 'in' door (e.g. from corridor to room)
                    # Or if it connects pedestrian to room.
                    # If a door connects pedestrian area to an outside area, it's more like an 'out' door.
                    # This needs careful definition based on your use case.
                    # For now: if it sees pedestrian and not outside, consider it 'in' (towards rooms/internal).
                    # If it sees pedestrian AND outside, the 'out' takes precedence.
                    conn_node.door_type = 'in'
                else: # Connects only to actual nodes (rooms, vertical, other doors)
                    conn_node.door_type = 'room' # Default for internal doors

                # Connect the door node to the identified neighboring ACTUAL nodes
                for neighbor_id_val in neighbor_ids_in_map:
                    if neighbor_id_val == conn_node.id or neighbor_id_val in pass_through_ids_in_id_map:
                        continue
                    
                    # Check if it's an actual node ID (positive)
                    # Special area IDs (OUTSIDE_ID_MAP_VALUE, PEDESTRIAN_ID_MAP_VALUE) are negative or large positive.
                    if neighbor_id_val > 0: # Assuming actual node IDs are positive and start from 1
                        target_node = self.graph_manager.get_node_by_id(neighbor_id_val)
                        if target_node and target_node.id != conn_node.id:
                            self.graph_manager.connect_nodes_by_ids(conn_node.id, target_node.id)