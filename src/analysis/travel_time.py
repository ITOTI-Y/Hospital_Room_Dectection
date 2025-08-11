"""
Calculates travel times between specified room-like nodes in the graph.
"""
import csv
import pathlib
import networkx as nx
import logging  # Added for logging
from typing import Dict, List, Union, Optional

from src.config import NetworkConfig
from src.network.node import Node

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_room_travel_times(
    graph: nx.Graph,
    config: NetworkConfig,
    output_dir: pathlib.Path,
    output_filename: str = "room_travel_times.csv",
    ground_floor_z: Optional[float] = None
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Calculates the shortest travel times between all pairs of individual "room" instances
    and designated "outward-facing door" nodes in the graph.

    Each room instance and each relevant outward-facing door is treated as a unique location.
    Room instance names will be 'NodeType_NodeID'.
    Outward-facing door names will be 'OutDoor_NodeID'.
    The output CSV will also include a final row with the area of each unique location.

    Args:
        graph: The input NetworkX graph. Nodes are expected to be `Node` objects
               or have a `node_obj` attribute pointing to a `Node` object.
        config: The NetworkConfig object.
        output_dir: The directory to save the resulting CSV file.
        output_filename: The name of the output CSV file.
        ground_floor_z: The Z-coordinate of the designated ground floor.
                        Only 'out' doors on this floor will be considered.

    Returns:
        A dictionary where keys are source location names and values are
        dictionaries mapping target location names to travel times.
    """
    if not graph.nodes:
        logger.warning("Graph is empty. Cannot calculate travel times.")
        return {}

    room_nodes: List[Node] = []
    out_door_nodes: List[Node] = []

    if ground_floor_z is None:
        logger.warning(
            "ground_floor_z not provided to calculate_room_travel_times. "
            "No 'out' doors will be specifically included as distinct locations for travel time analysis "
            "unless this behavior is changed in the filtering logic below."
        )

    for G_node_id in graph.nodes():
        # 获取节点数据
        G_node_data = graph.nodes[G_node_id]
        node_obj = G_node_data.get('node_obj')
        
        if not isinstance(node_obj, Node):
            logger.warning(f"Node {G_node_id} does not have a valid node_obj attribute")
            continue

        if node_obj.node_type in config.ROOM_TYPES:
            room_nodes.append(node_obj)
        elif node_obj.node_type in config.CONNECTION_TYPES and getattr(node_obj, 'door_type', None) == 'out':
            if ground_floor_z is not None:
                # Tolerance for Z comparison
                if abs(node_obj.z - ground_floor_z) < 0.1:
                    out_door_nodes.append(node_obj)
            # If ground_floor_z is None, no out_door_nodes are added from this path based on current logic.
            # If you want a fallback, it would be here. For "only ground floor", this is correct.

    location_nodes: List[Node] = room_nodes + out_door_nodes
    # Map Node object to its unique name
    location_names_map: Dict[Node, str] = {}
    # Map unique location name to its area
    location_areas_map: Dict[str, float] = {}

    for node_obj in room_nodes:
        unique_name = f"{node_obj.node_type}_{node_obj.id}"
        location_names_map[node_obj] = unique_name
        location_areas_map[unique_name] = getattr(node_obj, 'area', 0)

    for node_obj in out_door_nodes:  # These are already filtered for ground floor
        unique_name = f"门_{node_obj.id}"
        location_names_map[node_obj] = unique_name
        location_areas_map[unique_name] = getattr(node_obj, 'area', 0)

    if not location_nodes:
        logger.warning(
            "No room instances or designated (ground floor) outward-facing door nodes found to calculate travel times.")
        return {}

    def weight_function(u_node_id, v_node_id, edge_data):  # u,v are node IDs
        # 从图中获取节点对象
        if v_node_id not in graph.nodes:
            logger.error(f"Node {v_node_id} not found in graph")
            return float('inf')
        
        v_node_data = graph.nodes[v_node_id]
        v_node_obj = v_node_data.get('node_obj')
        
        if not isinstance(v_node_obj, Node):
            logger.error(
                f"Target node {v_node_id} in edge is not a valid Node object for weight func.")
            return float('inf')  # 返回无穷大而不是抛出异常
        
        return v_node_obj.time

    travel_times_data: Dict[str, Dict[str, Union[float, str]]] = {}
    logger.info(
        f"Calculating travel times for {len(location_nodes)} unique locations...")

    for start_node_obj in location_nodes:
        start_location_name = location_names_map[start_node_obj]
        travel_times_data.setdefault(start_location_name, {})

        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                source=start_node_obj.id,  # 使用节点ID而不是节点对象
                weight=weight_function
            )
        except nx.NodeNotFound:
            logger.warning(
                f"Start node {start_location_name} (ID: {start_node_obj.id}) not in graph for Dijkstra. Skipping.")
            continue
        except Exception as e:
            logger.error(
                f"Error during Dijkstra for {start_location_name} (ID: {start_node_obj.id}): {e}")
            continue

        for target_node_obj in location_nodes:
            target_location_name = location_names_map[target_node_obj]

            if start_node_obj == target_node_obj:
                travel_times_data[start_location_name][target_location_name] = round(
                    start_node_obj.time, 2)
                continue

            if target_node_obj.id in lengths:
                total_time = start_node_obj.time + lengths[target_node_obj.id]
                travel_times_data[start_location_name][target_location_name] = round(
                    total_time, 2)
            else:
                travel_times_data[start_location_name][target_location_name] = '∞'

    logger.info("Travel time calculation complete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = output_dir / output_filename

    # all_location_names will now be unique identifiers like "RoomType_ID" or "OutDoor_ID"
    all_location_names = sorted(list(location_names_map.values()))

    if not all_location_names:
        logger.warning(
            "No unique location names generated to write to CSV for travel times and areas.")
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

            # "Area (px²)" or "面积 (m²)" depending on your unit
            area_row_label = "面积"
            area_row_data = [area_row_label]
            for loc_name in all_location_names:  # loc_name is now unique
                area_val = location_areas_map.get(
                    loc_name)  # Get area by unique name
                if area_val is not None:
                    area_row_data.append(f"{area_val:.2f}")
                else:
                    area_row_data.append("N/A")  # Should ideally not happen
            writer.writerow(area_row_data)

        logger.info(f"Travel times and areas saved to {csv_file_path}")
    except IOError as e:
        logger.error(
            f"Failed to write travel times and areas CSV to {csv_file_path}: {e}")

    return travel_times_data
