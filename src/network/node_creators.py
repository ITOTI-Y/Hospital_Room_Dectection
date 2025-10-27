"""Defines strategies for creating different types of nodes in the network."""

from __future__ import annotations
import abc
import cv2
import numpy as np
from loguru import logger
from scipy.spatial import KDTree
from typing import Dict, Tuple, Any, Optional, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .network import Network

from src.config import graph_config

logger = logger.bind(module=__name__)


class BaseNodeCreator(abc.ABC):
    """Abstract base class for node creators."""

    def __init__(self, network: "Network"):
        self.network = network
        self.image_processor = network.image_processor
        self.graph_manager = network.graph_manager
        self.node_defs = graph_config.get_node_definitions()
        self.geometry_config = graph_config.get_geometry_config()
        self.special_ids = graph_config.get_special_ids()
        self.super_network_config = graph_config.get_super_network_config()

    def _get_node_properties(self, name: str) -> Dict[str, Any]:
        return self.node_defs.get(name, {})

    def _create_mask_for_node(
        self,
        image_data: np.ndarray,
        name: str,
        apply_morphology: bool = True,
        morphology_operation: str = "close_open",
        morphology_kernel_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        node_props = self._get_node_properties(name)
        color_rgb = node_props.get("rgb")
        if not color_rgb:
            logger.warning(f"Color for node '{name}' not found.")
            return None

        mask = np.all(
            image_data == np.array(color_rgb, dtype=np.uint8).reshape(1, 1, 3), axis=2
        )
        mask = mask.astype(np.uint8) * 255

        if apply_morphology:
            kernel_size = self.geometry_config.get("morphology_kernel_size", (5, 5))
            if not isinstance(kernel_size, Iterable):
                kernel_size = (int(kernel_size), int(kernel_size))
            kernel_size = morphology_kernel_size or kernel_size
            mask = self.image_processor.apply_morphology(
                mask, operation=morphology_operation, kernel_size=kernel_size
            )
        return mask

    def _find_connected_components(self, mask: np.ndarray, connectivity: int = 4):
        return cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    @abc.abstractmethod
    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        pass


class RoomNodeCreator(BaseNodeCreator):
    """Creates nodes for room-type areas."""

    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        target_names = graph_config.get_nodes_by_category(
            "SLOT"
        ) + graph_config.get_nodes_by_category("FIXED")
        area_threshold = self.geometry_config.get("area_threshold", 60)

        for name in target_names:
            mask = self._create_mask_for_node(processed_image_data, name)
            if mask is None:
                continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1:
                continue

            node_props = self._get_node_properties(name)
            for i in range(1, retval):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < area_threshold:
                    continue

                centroid_x, centroid_y = centroids[i]
                node_id = self.graph_manager.generate_node_id()

                self.graph_manager.add_node(
                    node_id,
                    name=name,
                    cname=node_props.get("cname"),
                    rgb=node_props.get("rgb"),
                    code=node_props.get("code"),
                    service_time=node_props.get("service_time", 0),
                    category=node_props.get("category"),
                    pos_x=centroid_x,
                    pos_y=centroid_y,
                    pos_z=z_level,
                    area=area,
                )
                id_map[labels == i] = node_id


class VerticalNodeCreator(BaseNodeCreator):
    """Creates nodes for vertical transport areas."""

    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        target_names = self.super_network_config.get("vertical_types", [])
        area_threshold = self.geometry_config.get("area_threshold", 60)

        for name in target_names:
            mask = self._create_mask_for_node(processed_image_data, name)
            if mask is None:
                continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1:
                continue

            node_props = self._get_node_properties(name)
            for i in range(1, retval):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < area_threshold:
                    continue

                centroid_x, centroid_y = centroids[i]
                node_id = self.graph_manager.generate_node_id()

                self.graph_manager.add_node(
                    node_id,
                    name=name,
                    cname=node_props.get("cname"),
                    rgb=node_props.get("rgb"),
                    code=node_props.get("code"),
                    service_time=node_props.get("service_time", 0),
                    time_per_floor=node_props.get("time_per_floor"),
                    category=node_props.get("category"),
                    pos_x=centroid_x,
                    pos_y=centroid_y,
                    pos_z=z_level,
                    area=area,
                )
                id_map[labels == i] = node_id


class MeshBasedNodeCreator(BaseNodeCreator):
    """Base class for creators that generate a mesh of nodes."""

    def _create_mesh_nodes_for_mask(
        self,
        mask: np.ndarray,
        name: str,
        id_map: np.ndarray,
        id_map_value: int,
        z_level: float,
        multiplier: int,
    ):
        id_map[mask != 0] = id_map_value
        retval, labels, stats, _ = self._find_connected_components(mask, connectivity=8)
        grid_size = self.geometry_config.get("grid_size", 40) * multiplier
        area_threshold = self.geometry_config.get("area_threshold", 60)
        node_props = self._get_node_properties(name)

        for i in range(1, retval):
            if stats[i, cv2.CC_STAT_AREA] < area_threshold:
                continue

            x, y, w, h, _ = stats[i]
            gx = np.arange(x + grid_size / 2, x + w, grid_size)
            gy = np.arange(y + grid_size / 2, y + h, grid_size)
            if len(gx) == 0 or len(gy) == 0:
                continue

            grid_x, grid_y = np.meshgrid(gx, gy)
            valid_indices = (
                labels[
                    grid_y.astype(int).clip(0, mask.shape[0] - 1),
                    grid_x.astype(int).clip(0, mask.shape[1] - 1),
                ]
                == i
            )

            valid_x, valid_y = grid_x[valid_indices], grid_y[valid_indices]
            node_ids, positions = [], []
            for vx, vy in zip(valid_x, valid_y):
                node_id = self.graph_manager.generate_node_id()
                self.graph_manager.add_node(
                    node_id,
                    name=name,
                    cname=node_props.get("cname"),
                    rgb=node_props.get("rgb"),
                    code=node_props.get("code"),
                    service_time=node_props.get("service_time", 0),
                    category=node_props.get("category"),
                    pos_x=vx,
                    pos_y=vy,
                    pos_z=z_level,
                    area=1.0,
                )
                node_ids.append(node_id)
                positions.append([vx, vy])

            if len(node_ids) < 2:
                continue
            kdtree = KDTree(positions)
            max_dist = np.sqrt(2) * grid_size * 1.05
            pairs = kdtree.query_pairs(r=max_dist)
            for i_idx, j_idx in pairs:
                self.graph_manager.connect_nodes_by_ids(
                    node_ids[i_idx], node_ids[j_idx]
                )


class PedestrianNodeCreator(MeshBasedNodeCreator):
    """Creates mesh nodes for pedestrian areas."""

    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        target_names = graph_config.get_nodes_by_category("PATH")
        for name in target_names:
            mask = self._create_mask_for_node(processed_image_data, name)
            if mask is None:
                continue
            self._create_mesh_nodes_for_mask(
                mask, name, id_map, self.special_ids.get("pedestrian", -3), z_level, 1
            )


class OutsideNodeCreator(MeshBasedNodeCreator):
    """Creates mesh nodes for outside areas."""

    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        ground_floor_num = self.super_network_config.get(
            "ground_floor_number_for_outside"
        )
        if floor_num != ground_floor_num:
            return

        target_names = self.super_network_config.get("outside_types", [])
        for name in target_names:
            mask = self._create_mask_for_node(processed_image_data, name)
            if mask is None:
                continue
            self._create_mesh_nodes_for_mask(
                mask, name, id_map, self.special_ids.get("outside", -1), z_level, 1
            )


class ConnectionNodeCreator(BaseNodeCreator):
    """Creates nodes for connections and links them to adjacent areas."""

    def create_nodes(
        self,
        processed_image_data: np.ndarray,
        id_map: np.ndarray,
        z_level: float,
        floor_num: int,
    ):
        door_names = [
            t
            for t in graph_config.get_nodes_by_category("CONNECTOR")
            if t not in self.super_network_config.get("vertical_types", [])
        ]
        if not door_names:
            return

        # Get masks for collision detection
        corridor_mask = self.network._get_mask("Corridor", is_category=False)
        outdoor_mask = self.network._get_mask("Outdoor", is_category=False)

        room_categories = ["FIXED", "SLOT"]
        room_mask = np.zeros_like(corridor_mask)
        for cat in room_categories:
            room_mask = cv2.bitwise_or(
                room_mask, self.network._get_mask(cat, is_category=True)
            )

        all_connector_names = graph_config.get_nodes_by_category("CONNECTOR")
        other_connector_names = [
            name for name in all_connector_names if name not in door_names
        ]
        other_connector_mask = self.network._get_mask(
            other_connector_names, is_category=False
        )

        kernel_list = self.geometry_config.get(
            "connection_dilation_kernel_size", [3, 3]
        )
        if not (isinstance(kernel_list, list) and len(kernel_list) == 2):
            kernel_list = [3, 3]
        dilation_kernel = np.ones(tuple(kernel_list), np.uint8)
        area_threshold = self.geometry_config.get("area_threshold", 60)

        for name in door_names:
            node_props = self._get_node_properties(name)
            mask = self._create_mask_for_node(processed_image_data, name)
            if mask is None:
                continue

            retval, labels, stats, centroids = self._find_connected_components(mask)
            if retval <= 1:
                continue

            for i in range(1, retval):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < area_threshold / 10 and area < 5:
                    continue

                component_mask = (labels == i).astype(np.uint8) * 255
                dilated_mask = cv2.dilate(
                    component_mask, dilation_kernel, iterations=1
                )

                # Check for collisions
                collides_outdoor = np.any(cv2.bitwise_and(dilated_mask, outdoor_mask))
                collides_corridor = np.any(
                    cv2.bitwise_and(dilated_mask, corridor_mask)
                )
                collides_other_connector = np.any(
                    cv2.bitwise_and(dilated_mask, other_connector_mask)
                )
                collides_room = np.any(cv2.bitwise_and(dilated_mask, room_mask))

                door_type = None
                colliding_node_ids = np.unique(id_map[dilated_mask != 0])
                colliding_node_ids = [
                    nid for nid in colliding_node_ids if nid > 0
                ]  # Filter out special IDs

                ground_floor_num = self.super_network_config.get(
                    "ground_floor_number_for_outside"
                )
                if (
                    collides_outdoor
                    and collides_corridor
                    and floor_num == ground_floor_num
                ):
                    door_type = "EXTERIOR"
                elif collides_other_connector and collides_corridor:
                    door_type = "INTERIOR"
                elif collides_room and (collides_corridor or collides_other_connector):
                    door_type = "ROOM"

                centroid_x, centroid_y = centroids[i]
                node_id = self.graph_manager.generate_node_id()

                self.graph_manager.add_node(
                    node_id,
                    name=name,
                    cname=node_props.get("cname"),
                    rgb=node_props.get("rgb"),
                    code=node_props.get("code"),
                    service_time=node_props.get("service_time", 0),
                    category=node_props.get("category"),
                    pos_x=centroid_x,
                    pos_y=centroid_y,
                    pos_z=z_level,
                    area=area,
                    door_type=door_type,
                    colliding_node_ids=colliding_node_ids,
                )
                id_map[labels == i] = node_id
