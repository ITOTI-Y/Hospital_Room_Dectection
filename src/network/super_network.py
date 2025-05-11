"""
Manages the construction of a multi-floor network by orchestrating
individual Network instances, potentially in parallel.
"""

import multiprocessing
import os  # For os.cpu_count()
import pathlib
import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial import KDTree

from src.config import NetworkConfig
from src.graph.node import Node
from .network import Network  # The single-floor network builder
from .floor_manager import FloorManager

logger = logging.getLogger(__name__)

# Worker function for multiprocessing - must be defined at the top-level or picklable


def _process_floor_worker(task_args: Tuple[pathlib.Path, float, int, Dict[str, Any], Dict[Tuple[int, int, int], Dict[str, Any]], bool]) \
        -> Tuple[Optional[nx.Graph], Optional[int], Optional[int], int, pathlib.Path, float]:
    """
    Worker function to process a single floor's network generation.

    Args:
        task_args: A tuple containing:
            - image_path (pathlib.Path): Path to the floor image.
            - z_level (float): Z-coordinate for this floor.
            - id_start_value (int): Starting node ID for this floor.
            - config_dict (Dict): Dictionary representation of NetworkConfig.
            - color_map_data (Dict): The color map.
            - process_outside_nodes (bool): Flag to process outside nodes.

    Returns:
        A tuple containing:
            - graph (Optional[nx.Graph]): Generated graph for the floor, or None on error.
            - width (Optional[int]): Image width, or None on error.
            - height (Optional[int]): Image height, or None on error.
            - next_id_val_from_worker (int): The next available ID from this worker's GraphManager.
            - image_path (pathlib.Path): Original image path (for result matching).
            - z_level (float): Original z_level (for result matching).
    """
    image_path, z_level, id_start_value, config_dict, color_map_data, process_outside_nodes = task_args
    try:
        # Reconstruct config from dict for the worker process
        # Note: This assumes NetworkConfig can be reconstructed from its __dict__
        # and COLOR_MAP is passed directly.
        # A more robust way might be to pass necessary primitive types or use a dedicated
        # config serialization if NetworkConfig becomes very complex.
        # Initialize with color_map
        worker_config = NetworkConfig(color_map_data=color_map_data)
        # Update other attributes from the passed dictionary
        for key, value in config_dict.items():
            # Avoid re-assigning COLOR_MAP
            if key != "COLOR_MAP" and hasattr(worker_config, key):
                setattr(worker_config, key, value)

        network_builder = Network(
            config=worker_config,
            color_map_data=color_map_data,
            id_generator_start_value=id_start_value
        )
        graph, width, height, next_id = network_builder.run(
            image_path=str(image_path),  # network.run expects str path
            z_level=z_level,
            process_outside_nodes=process_outside_nodes
        )
        return graph, width, height, next_id, image_path, z_level
    except Exception as e:
        logger.error(
            f"Error processing floor {image_path.name} in worker: {e}")
        # Return next_id_val as id_start_value + config.ESTIMATED_MAX_NODES_PER_FLOOR
        # to ensure main process ID allocation remains consistent even on worker failure.
        # A more sophisticated error handling might be needed.
        est_next_id = id_start_value + \
            config_dict.get("ESTIMATED_MAX_NODES_PER_FLOOR", 10000)
        return None, None, None, est_next_id, image_path, z_level


class SuperNetwork:
    """
    Orchestrates the creation of a multi-floor network graph.

    It manages multiple Network instances, one for each floor, and combines
    their graphs. It supports parallel processing of floors.
    """

    def __init__(self,
                 config: NetworkConfig,
                 color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]],
                 num_processes: Optional[int] = None,
                 base_floor: int = 0,
                 default_floor_height: Optional[float] = None,
                 vertical_connection_tolerance: Optional[int] = None):
        """
        Initializes the SuperNetwork.

        Args:
            config: The main configuration object.
            color_map_data: The RGB color to type mapping.
            num_processes: Number of processes to use for parallel floor processing.
                           Defaults to os.cpu_count().
            base_floor: Default base floor number if not detected from filename.
            default_floor_height: Default height between floors. Uses config value if None.
            vertical_connection_tolerance: Pixel distance tolerance for connecting
                                           vertical nodes between floors. Uses config if None.
        """
        self.config = config
        self.color_map_data = color_map_data
        self.super_graph: nx.Graph = nx.Graph()

        self.num_processes: int = num_processes if num_processes is not None else (
            os.cpu_count() or 1)

        _floor_height = default_floor_height if default_floor_height is not None else config.DEFAULT_FLOOR_HEIGHT
        self.floor_manager = FloorManager(
            base_floor_default=base_floor, default_floor_height=_floor_height)

        self.vertical_connection_tolerance: int = vertical_connection_tolerance \
            if vertical_connection_tolerance is not None else config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE

        self.floor_z_map: Dict[int, float] = {}  # floor_number -> z_coordinate
        self.path_to_floor_map: Dict[pathlib.Path,
                                     int] = {}  # image_path -> floor_number

        self.width: Optional[int] = None
        self.height: Optional[int] = None

    def _prepare_floor_data(self, image_file_paths: List[pathlib.Path],
                            z_levels_override: Optional[List[float]] = None) \
            -> List[Tuple[pathlib.Path, float, bool]]:
        """
        Determines floor numbers and Z-levels for each image path.

        Returns:
            A list of tuples: (image_path, z_level, process_outside_flag)
        """
        image_paths_as_pathlib = [pathlib.Path(p) for p in image_file_paths]

        self.path_to_floor_map, floor_to_path_map = self.floor_manager.auto_assign_floors(
            image_paths_as_pathlib)

        if z_levels_override and len(z_levels_override) == len(image_paths_as_pathlib):
            # Assign override Z-levels based on sorted floor numbers to maintain consistency
            # This assumes z_levels_override is provided in an order that corresponds to
            # how image_paths_as_pathlib would be sorted if floors were known.
            # A safer way is to map z_levels_override via path if a mapping is provided.
            # For now, let's assume z_levels_override corresponds to the sorted order of detected floors.
            sorted_paths_by_floor = sorted(
                self.path_to_floor_map.keys(), key=lambda p: self.path_to_floor_map[p])
            temp_path_to_z = {path: z for path, z in zip(
                sorted_paths_by_floor, z_levels_override)}
            self.floor_z_map = {
                self.path_to_floor_map[p]: temp_path_to_z[p] for p in sorted_paths_by_floor}

        else:
            self.floor_z_map = self.floor_manager.calculate_z_levels(
                floor_to_path_map)

        if not self.floor_z_map:
            raise ValueError("Could not determine Z-levels for floors.")

        # Prepare task list with (path, z_level, process_outside_flag)
        floor_tasks_data = []
        # Determine min and max floor numbers for process_outside logic
        all_floor_nums = list(self.floor_z_map.keys())
        min_floor_num = min(all_floor_nums) if all_floor_nums else 0
        max_floor_num = max(all_floor_nums) if all_floor_nums else 0

        for p_path, floor_num in self.path_to_floor_map.items():
            z_level = self.floor_z_map[floor_num]
            # Process outside nodes for ground floor (min_floor_num) and top floor (max_floor_num) by default,
            # or if DEFAULT_OUTSIDE_PROCESSING_IN_SUPERNETWORK is True for all.
            # This logic can be customized.
            process_outside = self.config.DEFAULT_OUTSIDE_PROCESSING_IN_SUPERNETWORK or \
                (floor_num == min_floor_num or floor_num == max_floor_num)
            floor_tasks_data.append((p_path, z_level, process_outside))

        # Sort tasks by Z-level to ensure somewhat deterministic ID assignment progression,
        # though ESTIMATED_MAX_NODES_PER_FLOOR should handle variations.
        floor_tasks_data.sort(key=lambda item: item[1])
        return floor_tasks_data

    def run(self,
            image_file_paths: List[str],  # List of string paths from main
            z_levels_override: Optional[List[float]] = None,
            force_vertical_tolerance: Optional[int] = None) -> nx.Graph:
        """
        Builds the multi-floor network.

        Args:
            image_file_paths: List of string paths to floor images.
            z_levels_override: Optional list to manually set Z-levels for each image.
                               Order should correspond to sorted floor order or be a path-to-z map.
            force_vertical_tolerance: Optionally override the vertical connection tolerance.

        Returns:
            The combined multi-floor NetworkX graph.
        """
        self.super_graph.clear()  # Clear previous graph if any
        image_paths_pl = [pathlib.Path(p) for p in image_file_paths]

        floor_run_data = self._prepare_floor_data(
            image_paths_pl, z_levels_override)
        if not floor_run_data:
            logger.warning("Warning: No floor data to process.")
            return self.super_graph

        tasks_for_pool = []
        current_id_start = 1
        config_dict_serializable = self.config.__dict__.copy()
        # COLOR_MAP is already part of config_dict_serializable if NetworkConfig init stores it.
        # If COLOR_MAP is global, it's fine for multiprocessing on systems where memory is copied (fork).
        # For spawn, it needs to be picklable or passed. Here, color_map_data is passed.

        for p_path, z_level, process_outside_flag in floor_run_data:
            tasks_for_pool.append((
                p_path, z_level, current_id_start,
                config_dict_serializable, self.color_map_data, process_outside_flag
            ))
            current_id_start += self.config.ESTIMATED_MAX_NODES_PER_FLOOR

        logging.info(
            f"Starting parallel processing of {len(tasks_for_pool)} floors using {self.num_processes} processes...")

        results = []
        # Use with statement for Pool to ensure proper cleanup
        # Only use pool if multiple tasks and processes
        if self.num_processes > 1 and len(tasks_for_pool) > 1:
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.map(_process_floor_worker, tasks_for_pool)
        else:  # Run sequentially for single process or single task
            logging.info("Running floor processing sequentially...")
            for task in tasks_for_pool:
                results.append(_process_floor_worker(task))

        first_floor_processed = True
        # To store (graph, width, height) for valid results
        processed_graphs_data = []

        for graph_result, width_res, height_res, _next_id, res_path, res_z in results:
            if graph_result is None or width_res is None or height_res is None:
                logger.warning(
                    f"Warning: Failed to process floor image {res_path.name} (z={res_z}). Skipping.")
                continue

            if first_floor_processed:
                self.width = width_res
                self.height = height_res
                first_floor_processed = False
            elif self.width != width_res or self.height != height_res:
                raise ValueError(
                    f"Image dimensions mismatch for {res_path.name}. "
                    f"Expected ({self.width},{self.height}), got ({width_res},{height_res}). "
                    "All floor images must have the same dimensions."
                )

            processed_graphs_data.append(
                graph_result)  # Store the graph itself

        # Combine graphs
        for floor_graph in processed_graphs_data:
            # Nodes in floor_graph should already have all attributes from Node class
            # and GraphManager.add_node should have added them to nx.Graph.
            # Make sure node objects themselves are added, not just IDs.
            self.super_graph.add_nodes_from(floor_graph.nodes(data=True))
            self.super_graph.add_edges_from(floor_graph.edges(data=True))

        if force_vertical_tolerance is not None:
            self.vertical_connection_tolerance = force_vertical_tolerance
        elif self.config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE == 0:  # 0 might mean auto-calculate
            self.vertical_connection_tolerance = self._auto_calculate_vertical_tolerance()

        self._connect_floors()

        logger.info(
            f"SuperNetwork construction complete. Total nodes: {self.super_graph.number_of_nodes()}")
        return self.super_graph

    def _auto_calculate_vertical_tolerance(self) -> int:
        """
        Automatically calculates a tolerance for connecting vertical nodes
        based on their typical proximity (if not specified).
        """
        vertical_nodes = [
            node for node in self.super_graph.nodes()  # Get node objects
            if isinstance(node, Node) and node.node_type in self.config.VERTICAL_TYPES
        ]
        if not vertical_nodes or len(vertical_nodes) < 2:
            return self.config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE  # Fallback

        # Consider only XY positions for tolerance calculation
        positions_xy = np.array([node.pos[:2] for node in vertical_nodes])
        if len(positions_xy) < 2:
            return self.config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE

        try:
            tree = KDTree(positions_xy)
            # Find distance to the nearest neighbor for each vertical node (excluding itself)
            # k=2 includes self and nearest
            distances, _ = tree.query(positions_xy, k=2)

            # Use distances to the actual nearest neighbor (second column)
            # Filter out zero distances if k=1 was used or if duplicates exist
            # Avoid self-match if k=1
            nearest_distances = distances[:, 1][distances[:, 1] > 1e-6]

            if nearest_distances.size == 0:
                return self.config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE

            avg_min_distance = np.mean(nearest_distances)
            # Tolerance could be a factor of this average minimum distance
            # Example: 50% of avg min distance
            calculated_tolerance = int(avg_min_distance * 0.5)
            logger.info(
                f"Auto-calculated vertical tolerance: {calculated_tolerance} (based on avg_min_dist: {avg_min_distance:.2f})")
            return max(10, calculated_tolerance)  # Ensure a minimum tolerance
        except Exception as e:
            logger.error(
                f"Error in auto-calculating tolerance: {e}. Using default.")
            return self.config.DEFAULT_VERTICAL_CONNECTION_TOLERANCE

    def _connect_floors(self) -> None:
        """
        Connects vertical transport nodes (e.g., stairs, elevators) between
        different floors if they are of the same type and spatially close in XY.
        """
        all_vertical_nodes_in_graph = [
            node for node in self.super_graph.nodes()  # Iterating actual Node objects
            if isinstance(node, Node) and node.node_type in self.config.VERTICAL_TYPES
        ]

        if not all_vertical_nodes_in_graph:
            logger.info("No vertical nodes found to connect between floors.")
            return

        # Group vertical nodes by their specific type (e.g., 'Stairs', 'Elevator')
        nodes_by_type: Dict[str, List[Node]] = {}
        for node in all_vertical_nodes_in_graph:
            nodes_by_type.setdefault(node.node_type, []).append(node)

        logger.info(
            f"Attempting to connect floors. Tolerance: {self.vertical_connection_tolerance} pixels.")
        connected_pairs_count = 0

        for node_type, nodes_of_this_type in nodes_by_type.items():
            if len(nodes_of_this_type) < 2:
                continue  # Not enough nodes of this type to form a connection

            # Sort nodes by Z-level, then by Y, then by X for potentially more stable pairing
            # Though KDTree approach doesn't strictly need pre-sorting.
            nodes_of_this_type.sort(
                key=lambda n: (n.pos[2], n.pos[1], n.pos[0]))

            # Build KDTree for XY positions of nodes of this specific type
            positions_xy = np.array([node.pos[:2]
                                    for node in nodes_of_this_type])
            if positions_xy.shape[0] < 2:
                continue  # Need at least 2 points for KDTree sensible query

            try:
                kdtree = KDTree(positions_xy)
            except Exception as e:
                logger.error(
                    f"Could not build KDTree for vertical node type {node_type}: {e}")
                continue

            processed_nodes_indices = set()  # To avoid redundant checks

            for i, current_node in enumerate(nodes_of_this_type):
                if i in processed_nodes_indices:
                    continue

                # Query for other nodes of the SAME TYPE within the XY tolerance
                # query_ball_point returns indices into the `positions_xy` array
                indices_in_ball = kdtree.query_ball_point(
                    current_node.pos[:2], r=self.vertical_connection_tolerance)

                for neighbor_idx in indices_in_ball:
                    if neighbor_idx == i:  # Don't connect to self
                        continue

                    neighbor_node = nodes_of_this_type[neighbor_idx]

                    # Crucial check: Ensure they are on different floors (Z-levels differ significantly)
                    # Z-levels are too close (same floor)
                    if abs(current_node.pos[2] - neighbor_node.pos[2]) < 1.0:
                        continue

                    # Connect if not already connected
                    if not self.super_graph.has_edge(current_node, neighbor_node):
                        self.super_graph.add_edge(
                            current_node, neighbor_node, type='vertical_connection')
                        connected_pairs_count += 1
                        # Mark both as processed for this type of pairing to avoid re-pairing B with A if A-B done
                        # This might be too aggressive if a node can connect to multiple above/below.
                        # A simpler approach is to just let KDTree find pairs.
                        # The has_edge check prevents duplicate edges.

                processed_nodes_indices.add(i)

        logger.info(
            f"Inter-floor connections made for {connected_pairs_count} pairs of vertical nodes.")
