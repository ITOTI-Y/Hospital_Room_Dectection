import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class CostManager:
    def __init__(self, config: ConfigLoader):
        self.logger = setup_logger(__name__)
        self.config = config
        self.travel_times = self._load_travel_times()
        self.pathways: Dict[str, Dict[str, Any]] = {}
        self.pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.service_weights: Dict[str, float] = defaultdict(float)


    def _load_travel_times(self):
        self.travel_times = pd.read_csv(self.config.paths.travel_times_csv)
        return self.travel_times

    def initialize(self, pathways: Dict[str, Dict[str, Any]], **kwargs):
        self.pathways = pathways
        self._precompute_pair_weights()

    def _precompute_pair_weights(self):
        self.logger.info("Precomputing pair weights for all pathways.")

        for pathway in self.pathways.values():
            sequence = pathway.get('core_sequence', [])
            weight = pathway.get('weight', 1.0)
            start_nodes = pathway.get('start_nodes', [])
            end_nodes = pathway.get('end_nodes', [])

            for i, dept in enumerate(sequence):
                self.service_weights[dept] += weight
                if i > 0:
                    prev_dept = sequence[i - 1]
                    pair = (prev_dept, dept)
                    self.pair_weights[pair] += weight

            for start_node in start_nodes:
                pair = (start_node, sequence[0])
                self.pair_weights[pair] += weight

            for i, end_node in enumerate(end_nodes):
                if i == 0:
                    pair = (sequence[-1], end_node)
                else:
                    pair = (end_nodes[i - 1], end_node)
                self.pair_weights[pair] += weight

        self.logger.info(f"Total {len(self.pair_weights)} pairs computed.")