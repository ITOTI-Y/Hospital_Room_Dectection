"""
Calculates travel times between specified room-like nodes in the graph.
"""
import csv
import pathlib
import networkx as nx
import logging  # Added for logging
from typing import Dict, List, Union

from src.config import NetworkConfig
from src.graph.node import Node

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_room_travel_times(
    graph: nx.Graph,
    config: NetworkConfig,
    output_dir: pathlib.Path,
    output_filename: str = "room_travel_times.csv"
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Calculates the shortest travel times between all pairs of "room" nodes
    and "outward-facing door" nodes in the graph.
    ... (docstring remains the same) ...
    """
    if not graph.nodes:
        logger.warning("Graph is empty. Cannot calculate travel times.")
        return {}

    room_nodes: List[Node] = []
    out_door_nodes: List[Node] = []

    for G_node_id, G_node_data in graph.nodes(data=True):
        node_obj = G_node_data.get('node_obj', G_node_id)
        if not isinstance(node_obj, Node):
            # logger.debug(f"Node {G_node_id} is not a Node object or lacks 'node_obj'. Skipping in travel time calc.")
            continue

        if node_obj.node_type in config.ROOM_TYPES:
            room_nodes.append(node_obj)
        elif node_obj.node_type in config.CONNECTION_TYPES and node_obj.door_type == 'out':
            out_door_nodes.append(node_obj)

    location_nodes: List[Node] = room_nodes + out_door_nodes
    location_names_map: Dict[Node, str] = {}

    for node in room_nodes:
        location_names_map[node] = node.node_type
    for i, node in enumerate(out_door_nodes):
        location_names_map[node] = f"OutDoor_{node.id}"

    if not location_nodes:
        logger.warning(
            "No room or outward-facing door nodes found to calculate travel times.")
        return {}

    def weight_function(u_id, v_id, edge_data):
        v_node_obj = v_id
        if not isinstance(v_node_obj, Node):
            v_node_data = graph.nodes[v_id]
            v_node_obj = v_node_data.get('node_obj', v_id)
            if not isinstance(v_node_obj, Node):
                logger.error(
                    f"Target node {v_id} in edge ({u_id}-{v_id}) is not a valid Node object for weight func.")
                raise ValueError(
                    f"Invalid node object for weight function: {v_id}")
        return v_node_obj.time

    travel_times_data: Dict[str, Dict[str, Union[float, str]]] = {}
    logger.info(
        f"Calculating travel times for {len(location_nodes)} locations...")

    for start_node_obj in location_nodes:
        start_location_name = location_names_map[start_node_obj]
        travel_times_data.setdefault(start_location_name, {})

        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                source=start_node_obj,
                weight=weight_function
            )
        except nx.NodeNotFound:
            logger.warning(
                f"Start node {start_location_name} (ID: {start_node_obj.id}) not found in graph for Dijkstra. Skipping.")
            continue
        except Exception as e:  # Catch other potential Dijkstra errors
            logger.error(
                f"Error during Dijkstra for start node {start_location_name} (ID: {start_node_obj.id}): {e}")
            continue

        for target_node_obj in location_nodes:
            target_location_name = location_names_map[target_node_obj]

            if start_node_obj == target_node_obj:
                travel_times_data[start_location_name][target_location_name] = round(
                    start_node_obj.time, 2)
                continue

            if target_node_obj in lengths:
                total_time = start_node_obj.time + lengths[target_node_obj]
                travel_times_data[start_location_name][target_location_name] = round(
                    total_time, 2)
            else:
                travel_times_data[start_location_name][target_location_name] = '∞'

    logger.info("Travel time calculation complete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = output_dir / output_filename
    all_location_names = sorted(list(set(location_names_map.values())))

    if not all_location_names:
        logger.warning("No location names to write to CSV for travel times.")
        return travel_times_data

    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['来源/目标'] + all_location_names
            writer.writerow(header)

            for source_name in all_location_names:
                row_data = [source_name]
                for target_name in all_location_names:
                    time_val = travel_times_data.get(
                        source_name, {}).get(target_name, 'N/A')
                    row_data.append(time_val)
                writer.writerow(row_data)
        logger.info(f"Travel times successfully saved to {csv_file_path}")
    except IOError as e:
        logger.error(
            f"Failed to write travel times CSV to {csv_file_path}: {e}")

    return travel_times_data
