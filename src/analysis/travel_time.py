"""
Calculates travel times between specified room-like nodes in the graph.
"""

import pathlib
import networkx as nx
import logging
import pandas as pd
from typing import Dict, Union

logger = logging.getLogger(__name__)


def calculate_room_travel_times(
    graph: nx.Graph,
    output_dir: pathlib.Path,
    output_filename: str = "hospital_travel_times.csv",
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Calculates shortest travel times between all pairs of service locations.

    This function identifies all nodes categorized as 'SLOT' or 'FIXED',
    calculates the travel time between them using Dijkstra's algorithm based
    on the 'weight' attribute of edges, and saves the results to a CSV file
    using pandas.

    Args:
        graph: The input NetworkX graph with nodes containing attributes.
        output_dir: The directory to save the resulting CSV file.
        output_filename: The name of the output CSV file.

    Returns:
        A dictionary where keys are source location names and values are
        dictionaries mapping target location names to travel times.
    """
    if not graph.nodes:
        logger.warning("Graph is empty. Cannot calculate travel times.")
        return {}

    location_nodes = {
        node_id: data
        for node_id, data in graph.nodes(data=True)
        if data.get("category") in ["SLOT", "FIXED"]
    }

    if not location_nodes:
        logger.warning("No 'SLOT' or 'FIXED' nodes found to calculate travel times.")
        return {}

    location_names = sorted(
        [f"{data['name']}_{node_id}" for node_id, data in location_nodes.items()]
    )
    name_to_id = {name: int(name.split("_")[-1]) for name in location_names}

    logger.info(
        f"Calculating travel times for {len(location_names)} unique locations..."
    )

    all_pairs_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))

    df_data = {}
    for start_name in location_names:
        start_id = name_to_id[start_name]
        row = {}
        if start_id in all_pairs_lengths:
            for target_name in location_names:
                target_id = name_to_id[target_name]
                time = all_pairs_lengths[start_id].get(target_id)
                row[target_name] = round(time, 2) if time is not None else float("inf")
        else:
            logger.warning(f"Node {start_name} (ID: {start_id}) is disconnected.")
            for target_name in location_names:
                row[target_name] = float("inf")
        df_data[start_name] = row

    df = pd.DataFrame.from_dict(df_data, orient="index")
    df.index.name = "Source/Target"

    # areas = pd.Series(
    #     {
    #         name: location_nodes[name_to_id[name]].get("area", 0)
    #         for name in location_names
    #     },
    #     name="Area",
    # )
    # df = pd.concat([df, areas.to_frame().T])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = output_dir / output_filename
    try:
        df.to_csv(csv_file_path, float_format="%.2f")
        logger.info(f"Travel times and areas saved to {csv_file_path}")
    except IOError as e:
        logger.error(f"Failed to write travel times CSV to {csv_file_path}: {e}")

    return df.to_dict()
