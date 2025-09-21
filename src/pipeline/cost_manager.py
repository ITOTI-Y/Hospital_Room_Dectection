import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, cast
from collections import defaultdict
from itertools import product, combinations_with_replacement

from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class CostManager:
    def __init__(self, config: ConfigLoader):
        self.logger = setup_logger(__name__)
        self.config = config

        self.node_def: Dict[str, Any] = {}
        self.cname_to_name: Dict[str, str] = {}
        self.name_to_name_id: Dict[str, List[str]] = defaultdict(list)
        self.initial_layout: Dict[str, str] = {}
        self.origin_service_cost: float = 0.0
        self.origin_travel_cost: float = 0.0
        self.travel_times: pd.DataFrame = pd.DataFrame(dtype=float)
        self.pair_times: Dict[Tuple[str, str], float] = defaultdict(float)
        self.pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.np_times: np.ndarray = np.array([])
        self.np_weights: np.ndarray = np.array([])
        self.layout: Dict[str, str] = {}

        self.id_to_name_id: Dict[int, str] = {}
        self.name_id_to_id: Dict[str, int] = {}
        self.name_id_to_area: Dict[str, float] = {}
        self.id_to_area: Dict[int, float] = {}
        self.area_dict: Dict[Tuple[int,int], bool] = {}

        self._load_travel_times()
        self._load_slots_information()
        self._load_node_definitions()

        self.pathways: Dict[str, Dict[str, Any]] = {}
        self.service_weights: Dict[str, float] = defaultdict(float)

    @property
    def origin_total_cost(self) -> float:
        return self.origin_service_cost + self.origin_travel_cost
    
    @property
    def current_total_cost(self) -> float:
        travel_time = np.dot(self.np_times[:, 2], self.np_weights[:, 2])
        return self.origin_service_cost + travel_time
    
    @property
    def current_travel_cost(self) -> float:
        travel_time = np.dot(self.np_times[:, 2], self.np_weights[:, 2])
        return travel_time
    
    @property
    def current_service_cost(self) -> float:
        return self.origin_service_cost

    def _load_travel_times(self) -> None:
        df = pd.read_csv(self.config.paths.travel_times_csv, index_col=0)
        for name_id in df.index:
            name, id_num = name_id.rsplit('_', 1)
            self.name_to_name_id[name].append(name_id)
            self.id_to_name_id[int(id_num)] = name_id
            self.name_id_to_id[name_id] = int(id_num)
        self.travel_times = df.copy()
    
    def _load_slots_information(self) -> None:
        df = pd.read_csv(self.config.paths.slots_csv)
        for _, row in df[['name', 'id', 'area']].iterrows():
            name_id = row['name'] + '_' + str(row['id'])
            self.name_id_to_area[name_id] = float(row['area'])
            self.id_to_area[int(row['id'])] = float(row['area'])
        self.slots = df.copy()

    def _load_node_definitions(self) -> None:
        self.node_def = self.config.graph_config.node_definitions
        self.cname_to_name: Dict[str, str] = {v['cname']: k for k, v in self.node_def.items()}

    def _get_initial_layout(self) -> None:
        for _, row in self.slots[['name', 'id']].iterrows():
            name = row['name'] + '_' + str(row['id'])
            self.initial_layout[name] = name
        self.layout = self.initial_layout.copy()

    def _sort_np_matrices(self):
        sorted_indices = np.lexsort((self.np_times[:, 1], self.np_times[:, 0]))
        self.np_times = self.np_times[sorted_indices]
        sorted_indices = np.lexsort((self.np_weights[:, 1], self.np_weights[:, 0]))
        self.np_weights = self.np_weights[sorted_indices]

    def _precompute_np_matrices(self):

        time_list = []
        for (dept1, dept2), time in self.pair_times.items():
            id1 = self.name_id_to_id.get(dept1)
            id2 = self.name_id_to_id.get(dept2)
            if id1 is None or id2 is None:
                continue
            time_list.append((id1, id2, time))
        self.np_times = np.array(time_list)

        weight_list = []
        for (dept1, dept2), weight in self.pair_weights.items():
            id1 = self.name_id_to_id.get(dept1)
            id2 = self.name_id_to_id.get(dept2)
            if id1 is None or id2 is None:
                continue
            weight_list.append((id1, id2, weight))
        self.np_weights = np.array(weight_list)

    def initialize(self, pathways: Dict[str, Dict[str, Any]], **kwargs):
        self.pathways = pathways
        self._get_initial_layout()
        self._precompute_pair_weights()
        self._precompute_travel_time_cost(self.initial_layout)
        self._precompute_np_matrices()
        self._precompute_area_dict()
        self._sort_np_matrices()
        self.origin_service_cost = self._precompute_service_time_cost()
        self.origin_travel_cost = np.dot(self.np_times[:, 2], self.np_weights[:, 2])
        self.logger.info(f"Initial Service Cost: {self.origin_service_cost:.2f}")
        self.logger.info(f"Initial Travel Cost: {self.origin_travel_cost:.2f}")
        self.logger.info(f"Initial Total Cost: {self.origin_total_cost:.2f}")
        pass

    def _precompute_area_dict(self):
        tolerance = cast(float, self.config.constraints.area_compatibility_tolerance)
        for id1, id2 in combinations_with_replacement(self.id_to_area.keys(), 2):
            area1 = self.id_to_area[id1]
            area2 = self.id_to_area[id2]
            is_compatible = abs(area1 - area2) <= tolerance * min(area1, area2)
            self.area_dict[(id1, id2)] = is_compatible

    def _precompute_pair_weights(self):
        self.logger.info("Precomputing pair weights for all pathways.")

        for dept1 in self.travel_times.index:
            for dept2 in self.travel_times.columns:
                if dept1 == dept2:
                    continue
                elif (dept2, dept1) in self.pair_weights:
                    continue
                self.pair_weights[(dept1, dept2)] = 0.0

        for pathway in self.pathways.values():
            sequence: List[str] = pathway.get('core_sequence', [])
            weight = pathway.get('weight', 1.0)
            start_nodes: List[str] = pathway.get('start_nodes', [])
            end_nodes: List[str] = pathway.get('end_nodes', [])

            for i, dept in enumerate(sequence):
                dept = self.cname_to_name.get(dept, dept)
                self.service_weights[dept] += weight
                if i > 0:
                    prev_dept = self.cname_to_name.get(sequence[i - 1], sequence[i - 1])
                    pairs = list(product(self.name_to_name_id.get(prev_dept, [prev_dept]), self.name_to_name_id.get(dept, [dept])))
                    for pair in pairs:
                        reversed_pair = (pair[1], pair[0])
                        if reversed_pair in self.pair_weights:
                            self.pair_weights[reversed_pair] += weight / len(pairs)
                            continue
                        self.pair_weights[pair] += weight / len(pairs)

            for start_node in start_nodes:
                start_node = self.cname_to_name.get(start_node, start_node)
                dept = self.cname_to_name.get(sequence[0], sequence[0])
                pairs: List[Tuple[str, str]] = list(product(self.name_to_name_id.get(start_node, [start_node]), self.name_to_name_id.get(dept, [dept])))
                for pair in pairs:
                    reversed_pair = (pair[1], pair[0])
                    if reversed_pair in self.pair_weights:
                        self.pair_weights[reversed_pair] += weight / len(pairs)
                        continue
                    self.pair_weights[pair] += weight / len(pairs)

            for i, end_node in enumerate(end_nodes):
                end_node = self.cname_to_name.get(end_node, end_node)
                if i == 0:
                    dept = self.cname_to_name.get(sequence[-1], sequence[-1])
                    pairs = list(product(self.name_to_name_id.get(dept, [dept]), self.name_to_name_id.get(end_node, [end_node])))
                else:
                    dept = self.cname_to_name.get(end_nodes[i - 1], end_nodes[i - 1])
                    pairs = list(product(self.name_to_name_id.get(dept, [dept]), self.name_to_name_id.get(end_node, [end_node])))
                for pair in pairs:
                    reversed_pair = (pair[1], pair[0])
                    if reversed_pair in self.pair_weights:
                        self.pair_weights[reversed_pair] += weight / len(pairs)
                        continue
                    self.pair_weights[pair] += weight / len(pairs)
        self.logger.info(f"Total {len(self.pair_weights)} pairs computed.")

    def _precompute_service_time_cost(self) -> float:

        service_cost = 0.0
        for dept in self.service_weights:
            service_time = self.node_def.get(dept, {}).get('service_time', 0)
            service_cost += service_time * self.service_weights[dept]
        return service_cost

    def _precompute_travel_time_cost(self, layout: Dict[str, str]) -> None:
        for (dept1, dept2), _ in self.pair_weights.items():
            slot1 = layout.get(dept1, dept1)
            slot2 = layout.get(dept2, dept2)
            time = cast(float, self.travel_times.loc[slot1, slot2])
            self.pair_times[(dept1, dept2)] += time

    

    def swap(self, dept1: str, dept2: str) -> float:
        slot1 = self.layout[dept1]
        slot2 = self.layout[dept2]

        id1 = self.name_id_to_id.get(slot1, -1)
        id2 = self.name_id_to_id.get(slot2, -1)

        if self.area_dict.get((id1, id2), False) is False:
            self.layout[dept1] = slot1
            self.layout[dept2] = slot2
            self.logger.warning(f"Swap between {dept1} and {dept2} violates area compatibility constraint. Swap reverted.")
            return -np.inf

        mask_id1 = self.np_times[:, :2] == id1
        mask_id2 = self.np_times[:, :2] == id2

        self.np_times[:, :2][mask_id1] = id2
        self.np_times[:, :2][mask_id2] = id1

        self._sort_np_matrices()

        self.layout[dept1] = slot2
        self.layout[dept2] = slot1

        return self.current_travel_cost