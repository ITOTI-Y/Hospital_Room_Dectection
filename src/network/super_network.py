"""Manages the construction of a multi-floor network graph.

This module orchestrates the creation of a comprehensive, multi-floor
network by managing individual `Network` instances for each floor. It
supports parallel processing to accelerate the generation of large-scale
graphs and handles the vertical connections between floors.
"""

import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from scipy.spatial import KDTree

from src.config import graph_config

from .floor_manager import FloorManager
from .network import Network

logger = logger.bind(module=__name__)


def _process_floor_worker(
    task_args: tuple[Path, float, int, bool, int],
) -> tuple[nx.Graph | None, int | None, int | None, int, Path, float, int]:
    """Processes a single floor's network generation in a worker process.

    Args:
        task_args: A tuple containing the arguments for the worker:
            - image_path (Path): Path to the floor's image file.
            - z_level (float): The Z-coordinate for all nodes on this floor.
            - id_start_value (int): The starting ID for node generation.
            - process_outside_nodes (bool): Flag to process outside areas.
            - floor_num (int): The floor number.

    Returns:
        A tuple with the results:
            - graph (Optional[nx.Graph]): The generated graph for the floor.
            - width (Optional[int]): The width of the floor image.
            - height (Optional[int]): The height of the floor image.
            - next_id (int): The next available node ID after this worker.
            - image_path (Path): The original image path for identification.
            - z_level (float): The original Z-level for identification.
            - floor_num (int): The original floor number for identification.
    """
    image_path, z_level, id_start_value, process_outside_nodes, floor_num = task_args
    try:
        network_builder = Network(id_generator_start_value=id_start_value)
        graph, width, height, next_id = network_builder.run(
            image_path=image_path,
            z_level=z_level,
            process_outside_nodes=process_outside_nodes,
            floor_num=floor_num,
        )
        return graph, width, height, next_id, image_path, z_level, floor_num
    except Exception as e:
        logger.error(f'Error processing floor {image_path.name} in worker: {e}')
        s_config = graph_config.get_super_network_config()
        est_next_id = id_start_value + s_config.get(
            'estimated_max_nodes_per_floor', 10000
        )
        return None, None, None, est_next_id, image_path, z_level, floor_num


class SuperNetwork:
    """Orchestrates the creation of a multi-floor network graph."""

    def __init__(self, num_processes: int | None = None, base_floor: int = 0):
        """Initializes the SuperNetwork.

        Args:
            num_processes: Number of processes for parallel floor processing.
                           Defaults to the number of CPU cores.
            base_floor: The default base floor number if it cannot be
                        detected from the image filename.
        """
        self.s_config = graph_config.get_super_network_config()
        self.super_graph: nx.Graph = nx.Graph()
        self.designated_ground_floor_number: int | None = None
        self.designated_ground_floor_z: float | None = None
        self.num_processes: int = (
            num_processes if num_processes is not None else (os.cpu_count() or 1)
        )

        floor_height = self.s_config.get('default_floor_height', 10.0)
        self.floor_manager = FloorManager(
            base_floor_default=base_floor, default_floor_height=floor_height
        )

        self.vertical_connection_tolerance: int = self.s_config.get(
            'default_vertical_connection_tolerance', 0
        )
        self.floor_z_map: dict[int, float] = {}
        self.path_to_floor_map: dict[Path, int] = {}
        self.width: int | None = None
        self.height: int | None = None

    def _should_process_outside_nodes(
        self, floor_num: int, designated_ground_floor_num: int | None
    ) -> bool:
        """Determines if outside nodes should be processed for a specific floor."""
        generate_outside = self.s_config.get('generate_outside_nodes', False)
        if not generate_outside:
            return False

        if (
            designated_ground_floor_num is None
            or floor_num != designated_ground_floor_num
        ):
            return False
        return bool(self.s_config.get('outside_types'))

    def _prepare_floor_data(
        self,
        image_file_paths: list[Path],
        z_levels_override: list[float] | None = None,
    ) -> list[tuple[Path, float, bool, int]]:
        """Determines floor numbers and Z-levels for each image path."""
        self.path_to_floor_map, floor_to_path_map = (
            self.floor_manager.auto_assign_floors(image_file_paths)
        )

        if z_levels_override and len(z_levels_override) == len(image_file_paths):
            sorted_paths = sorted(
                self.path_to_floor_map.keys(), key=lambda p: self.path_to_floor_map[p]
            )
            path_to_z = dict(zip(sorted_paths, z_levels_override, strict=True))
            self.floor_z_map = {
                self.path_to_floor_map[p]: z for p, z in path_to_z.items()
            }
        else:
            self.floor_z_map = self.floor_manager.calculate_z_levels(floor_to_path_map)

        if not self.floor_z_map:
            raise ValueError('Could not determine Z-levels for floors.')

        all_floor_nums = list(self.floor_z_map.keys())
        if not all_floor_nums:
            return []

        designated_ground_floor_num = self.s_config.get(
            'ground_floor_number_for_outside'
        )
        if designated_ground_floor_num is None:
            positive_or_zero_floors = sorted([fn for fn in all_floor_nums if fn >= 0])
            if 0 in all_floor_nums:
                designated_ground_floor_num = 0
            elif 1 in all_floor_nums and not any(0 <= fn < 1 for fn in all_floor_nums):
                designated_ground_floor_num = 1
            elif positive_or_zero_floors:
                designated_ground_floor_num = positive_or_zero_floors[0]

        self.designated_ground_floor_number = designated_ground_floor_num
        if designated_ground_floor_num is not None:
            self.designated_ground_floor_z = self.floor_z_map.get(
                designated_ground_floor_num
            )

        tasks = []
        for p, floor_num in self.path_to_floor_map.items():
            if (z_level := self.floor_z_map.get(floor_num)) is not None:
                process_outside = self._should_process_outside_nodes(
                    floor_num, designated_ground_floor_num
                )
                tasks.append((p, z_level, process_outside, floor_num))

        return sorted(tasks, key=lambda item: item[1])

    def run(
        self,
        image_file_paths: list[Path],
        z_levels_override: list[float] | None = None,
        force_vertical_tolerance: int | None = None,
    ) -> nx.Graph:
        """Builds the multi-floor network."""
        self.super_graph.clear()
        floor_run_data = self._prepare_floor_data(image_file_paths, z_levels_override)
        if not floor_run_data:
            logger.warning('No floor data to process.')
            return self.super_graph

        max_nodes = self.s_config.get('estimated_max_nodes_per_floor', 10000)
        tasks = [
            (p, z, i * max_nodes + 1, outside, floor_num)
            for i, (p, z, outside, floor_num) in enumerate(floor_run_data, start=1)
        ]

        self.num_processes = min(self.num_processes, len(tasks))
        logger.info(
            f'Processing {len(tasks)} floors using {self.num_processes} processes...'
        )
        results: list[Any] = []
        if self.num_processes > 1 and len(tasks) > 1:
            results = list(
                Parallel(n_jobs=self.num_processes)(
                    delayed(_process_floor_worker)(task) for task in tasks
                )
            )
        else:
            results = [_process_floor_worker(task) for task in tasks]

        first_floor = True
        for graph, width, height, _, path, _, _ in results:
            if graph is None:
                logger.warning(f'Failed to process floor image {path.name}. Skipping.')
                continue
            if first_floor:
                self.width, self.height, first_floor = width, height, False
            elif (self.width, self.height) != (width, height):
                raise ValueError(f'Image dimensions mismatch for {path.name}.')

            self.super_graph.add_nodes_from(graph.nodes(data=True))
            self.super_graph.add_edges_from(graph.edges(data=True))

        if force_vertical_tolerance is not None:
            self.vertical_connection_tolerance = force_vertical_tolerance
        elif self.s_config.get('default_vertical_connection_tolerance') == 0:
            self.vertical_connection_tolerance = (
                self._auto_calculate_vertical_tolerance()
            )

        self._connect_floors()
        logger.info(
            f'SuperNetwork construction complete. Total nodes: {self.super_graph.number_of_nodes()}'
        )
        return self.super_graph

    def _auto_calculate_vertical_tolerance(self) -> int:
        """Automatically calculates a tolerance for connecting vertical nodes."""
        vertical_types = self.s_config.get('vertical_types', [])
        vertical_nodes_data = [
            data
            for _, data in self.super_graph.nodes(data=True)
            if data.get('type') in vertical_types
        ]
        if len(vertical_nodes_data) < 2:
            return self.s_config.get('default_vertical_connection_tolerance', 0)

        positions_xy = np.array(
            [(data['pos_x'], data['pos_y']) for data in vertical_nodes_data]
        )
        if len(positions_xy) < 2:
            return self.s_config.get('default_vertical_connection_tolerance', 0)

        try:
            tree = KDTree(positions_xy)
            distances, _ = tree.query(positions_xy, k=2)
            nearest_distances = distances[:, 1][distances[:, 1] > 1e-6]
            if nearest_distances.size == 0:
                return self.s_config.get('default_vertical_connection_tolerance', 0)

            avg_min_dist = np.mean(nearest_distances)
            factor = self.s_config.get('vertical_tolerance_factor', 0.5)
            min_tol = self.s_config.get('min_vertical_tolerance', 10)
            calculated_tolerance = int(avg_min_dist * factor)
            logger.info(
                f'Auto-calculated vertical tolerance: {calculated_tolerance} (based on avg_min_dist: {avg_min_dist:.2f})'
            )
            return max(min_tol, calculated_tolerance)
        except Exception as e:
            logger.error(f'Error in auto-calculating tolerance: {e}. Using default.')
            return self.s_config.get('default_vertical_connection_tolerance', 0)

    def _connect_floors(self) -> None:
        """Connects vertical transport nodes between different floors."""
        vertical_types = self.s_config.get('vertical_types', [])
        all_vertical_nodes = [
            (node_id, data)
            for node_id, data in self.super_graph.nodes(data=True)
            if data.get('name') in vertical_types
        ]
        if not all_vertical_nodes:
            logger.info('No vertical nodes found to connect between floors.')
            return

        nodes_by_name: dict[str, list[dict[str, Any]]] = {}
        for node_id, data in all_vertical_nodes:
            data['id'] = node_id
            nodes_by_name.setdefault(data.get('name', 'unknown'), []).append(data)

        logger.info(
            f'Attempting to connect floors. Tolerance: {self.vertical_connection_tolerance} pixels.'
        )
        connected_pairs_count = 0
        z_diff_threshold = self.s_config.get('z_level_diff_threshold', 1.0)

        for _, nodes in nodes_by_name.items():
            if len(nodes) < 2:
                continue

            nodes.sort(key=lambda n: (n['pos_z'], n['pos_y'], n['pos_x']))
            positions = np.array([(n['pos_x'], n['pos_y']) for n in nodes])
            if len(positions) < 2:
                continue

            kdtree = KDTree(positions)
            pairs = kdtree.query_pairs(r=self.vertical_connection_tolerance)

            for i, j in pairs:
                node_i, node_j = nodes[i], nodes[j]
                if abs(
                    node_i['pos_z'] - node_j['pos_z']
                ) > z_diff_threshold and not self.super_graph.has_edge(
                    node_i['id'], node_j['id']
                ):
                    time_per_floor = node_i.get('time_per_floor', 10.0)
                    self.super_graph.add_edge(
                        node_i['id'],
                        node_j['id'],
                        type='vertical_connection',
                        weight=time_per_floor,
                    )
                    connected_pairs_count += 1

        logger.info(
            f'Inter-floor connections made for {connected_pairs_count} pairs of vertical nodes.'
        )
