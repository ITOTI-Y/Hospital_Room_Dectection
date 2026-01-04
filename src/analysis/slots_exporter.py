"""Exports service time information for SLOT nodes to a CSV file."""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def export_slots_to_csv(
    graph: nx.Graph, output_dir: Path, output_filename: str = 'slots.csv'
) -> None:
    """
    Extracts nodes with the 'SLOT' category and writes all their attributes
    to a CSV file.

    Args:
        graph: The input NetworkX graph.
        output_dir: The directory to save the CSV file.
        output_filename: The name of the output CSV file.
    """
    if not graph.nodes:
        logger.warning('Graph is empty. Cannot export slots.')
        return

    slot_nodes: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        if data.get('category') == 'SLOT':
            node_data = data.copy()
            node_data['id'] = node_id
            slot_nodes.append(node_data)

    if not slot_nodes:
        logger.warning("No nodes with category 'SLOT' found in the graph.")
        return

    df = pd.DataFrame(slot_nodes)

    output_path = output_dir / output_filename
    try:
        # The columns will be dynamically determined by the keys in the node data
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f'Successfully exported {len(df)} slots to {output_path}')
    except OSError as e:
        logger.error(f'Failed to write slots CSV to {output_path}: {e}')
