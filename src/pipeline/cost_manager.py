import pandas as pd
import numpy as np
import copy
from typing import Dict, Any, Tuple, List, cast, Optional
from collections import defaultdict
from itertools import product, combinations_with_replacement, combinations

from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger

class CostManager:
    def __init__(self, config: ConfigLoader, is_shuffle: bool = False):
        self.logger = setup_logger(__name__)
        self.config = config
        
        self.pathways: Dict[str, Dict[str, Any]] = {}
        self.max_travel_time: float = 0.0
        self.min_travel_time: float = float('inf')
        self.travel_times: pd.DataFrame = pd.DataFrame(dtype=float)
        self.name_to_name_id: Dict[str, List[str]] = defaultdict(list)
        self.id_to_name_id: Dict[int, str] = {}
        self.slots: pd.DataFrame = pd.DataFrame()
        self.name_id_to_id: Dict[str, int] = {}
        self.name_id_to_area: Dict[str, float] = {}
        self.id_to_area: Dict[int, float] = {}
        self.node_def: Dict[str, Any] = {}
        self.cname_to_name: Dict[str, str] = {}
        self.adjacency_preferences: Dict[Tuple[str, str], float] = {}

        self._load_travel_times(is_shuffle)
        self._load_slots_information()
        self._load_node_definitions()
        self._load_adjacency_preferences()

        self.initial_layout: Dict[str, str] = {}
        self.pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.pair_times: Dict[Tuple[str, str], float] = defaultdict(float)
        self.service_weights: Dict[str, float] = defaultdict(float)
        self.origin_service_cost: float = 0.0
        self.area_dict: Dict[Tuple[int,int], bool] = {}

        self.shared_data = {}

    def _load_travel_times(self, is_shuffle: bool) -> None:
        df = pd.read_csv(self.config.paths.travel_times_csv, index_col=0)
        if is_shuffle:
            df.columns = np.random.permutation(df.columns)
            df.index = df.columns
        for name_id in df.index:
            name, id_num = name_id.rsplit('_', 1)
            self.name_to_name_id[name].append(name_id)
            self.id_to_name_id[int(id_num)] = name_id
            self.name_id_to_id[name_id] = int(id_num)
        self.max_travel_time = df.values.max()
        self.min_travel_time = df.values[df.values > 0].min()
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
        self.cname_to_name = {v['cname']: k for k, v in self.node_def.items()}

    def _load_adjacency_preferences(self) -> None:
        if adjacency_preferences := self.config.constraints.adjacency_preferences:
            for pref in adjacency_preferences:
                depts: List[str] = []
                dept_names = [self.cname_to_name[i] for i in pref.depts]
                for i in dept_names:
                    depts.extend(self.name_to_name_id[i])
                weight = float(pref.weight)
                self.adjacency_preferences.update({(dept1, dept2): weight for dept1, dept2 in combinations(depts, 2)})

    def _precompute_initial_layout(self) -> None:
        for _, row in self.slots[['name', 'id']].iterrows():
            name = row['name'] + '_' + str(row['id'])
            self.initial_layout[name] = name

    def _precompute_pair_weights(self):

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

    def _precompute_service_time_cost(self) -> None:

        service_cost = 0.0
        for dept in self.service_weights:
            service_time = self.node_def.get(dept, {}).get('service_time', 0)
            service_cost += service_time * self.service_weights[dept]
        self.origin_service_cost = service_cost

    def _precompute_travel_time_cost(self) -> None:
        for (dept1, dept2), _ in self.pair_weights.items():
            slot1 = self.initial_layout.get(dept1, dept1)
            slot2 = self.initial_layout.get(dept2, dept2)
            time = cast(float, self.travel_times.loc[slot1, slot2])
            self.pair_times[(dept1, dept2)] += time
    
    def _precompute_area_dict(self):
        tolerance = cast(float, self.config.constraints.area_compatibility_tolerance)
        for id1, id2 in combinations_with_replacement(self.id_to_area.keys(), 2):
            area1 = self.id_to_area[id1]
            area2 = self.id_to_area[id2]
            is_compatible = (abs(area1 - area2) / max(area1, area2)) <= tolerance
            self.area_dict[(id1, id2)] = is_compatible

    def initialize(self, pathways: Dict[str, Dict[str, Any]] = {}) -> None:
        self.pathways = pathways
        self._precompute_initial_layout()
        self._precompute_pair_weights()
        self._precompute_service_time_cost()
        self._precompute_travel_time_cost()
        self._precompute_area_dict()
        self.shared_data = {
            "max_travel_time": self.max_travel_time,
            "min_travel_time": self.min_travel_time,
            "travel_times": self.travel_times,
            "pair_weights": self.pair_weights,
            "pair_times": self.pair_times,
            "service_weights": self.service_weights,
            "node_def": self.node_def,
            "name_id_to_id": self.name_id_to_id,
            "area_dict": self.area_dict,
            "initial_layout": self.initial_layout,
            "origin_service_cost": self.origin_service_cost,
            "adjacency_preferences": self.adjacency_preferences,
            "id_to_area": self.id_to_area
        }

    def create_cost_engine(self) -> 'CostEngine':
        return CostEngine(self.shared_data)

class CostEngine():
    def __init__(self, shared_data: Dict[str, Any]):
        self.logger = setup_logger(__name__)
        self.max_travel_time = shared_data["max_travel_time"]
        self.min_travel_time = shared_data["min_travel_time"]
        self.travel_times = shared_data["travel_times"]
        self.pair_weights = shared_data["pair_weights"]
        self.pair_times = shared_data["pair_times"]
        self.name_id_to_id = shared_data["name_id_to_id"]
        self.area_dict = shared_data["area_dict"]
        self.origin_service_cost = shared_data["origin_service_cost"]
        self.initial_layout = shared_data["initial_layout"]
        self.adjacency_preferences = shared_data["adjacency_preferences"]
        self.id_to_area = shared_data["id_to_area"]

        self._layout = copy.deepcopy(self.initial_layout)
        self._slot_layout = {v: k for k, v in self._layout.items()}

        self.np_times: np.ndarray = np.array([])
        self.np_weights: np.ndarray = np.array([])
        self._precompute_np_times()
        self._precompute_np_weights()
        self._sort_np_matrices()

    def _precompute_np_times(self) -> None:
        time_list = []
        for (dept1, dept2), time in self.pair_times.items():
            id1 = self.name_id_to_id.get(dept1)
            id2 = self.name_id_to_id.get(dept2)
            if id1 is None or id2 is None:
                continue
            time_list.append((id1, id2, time))
        self.np_times = np.array(time_list)

    def _precompute_np_weights(self) -> None:
        weight_list = []
        for (dept1, dept2), weight in self.pair_weights.items():
            id1 = self.name_id_to_id.get(dept1)
            id2 = self.name_id_to_id.get(dept2)
            if id1 is None or id2 is None:
                continue
            weight_list.append((id1, id2, weight))
        self.np_weights = np.array(weight_list)

    def _sort_np_matrices(self):
        self.np_times[:, :2] = np.sort(self.np_times[:, :2], axis=1)
        self.np_weights[:, :2] = np.sort(self.np_weights[:, :2], axis=1)
        sorted_indices = np.lexsort((self.np_times[:, 1], self.np_times[:, 0]))
        self.np_times = self.np_times[sorted_indices]
        sorted_indices = np.lexsort((self.np_weights[:, 1], self.np_weights[:, 0]))
        self.np_weights = self.np_weights[sorted_indices]

    @property
    def layout(self) -> Dict[str, str]:
        return self._layout
    
    @property
    def slot_layout(self) -> Dict[str, str]:
        return self._slot_layout

    @property
    def current_travel_cost(self) -> float:
        travel_time = np.dot(self.np_times[:, 2], self.np_weights[:, 2])
        return travel_time
    
    @property
    def current_total_cost(self) -> float:
        return self.origin_service_cost + self.current_travel_cost
    
    @property
    def current_adjacency_cost(self) -> float:
        adjacency_cost = 0.0
        for (dept1, dept2), weight in self.adjacency_preferences.items():
            time = self.dept_to_dept_cost(dept1, dept2)
            adjacency_cost += time / self.max_travel_time * weight
        return adjacency_cost
    
    
    def dept_to_dept_cost(self, dept1: str, dept2: str) -> Optional[float]:
        id1 = self.name_id_to_id[dept1]
        id2 = self.name_id_to_id[dept2]
        mask = self.np_times[:, :2] == np.array([id1, id2])
        if np.any(np.all(mask, axis=1)):
            time = self.np_times[np.all(mask, axis=1), 2][0]
            return time
        else:
            mask = self.np_times[:, :2] == np.array([id2, id1])
            if np.any(np.all(mask, axis=1)):
                time = self.np_times[np.all(mask, axis=1), 2][0]
                return time

    def reset(self) -> None:
        self._layout = copy.deepcopy(self.initial_layout)
        self._precompute_np_times()
        self._sort_np_matrices()

    def swap(self, dept1: str, dept2: str) -> Optional[float]:
        slot1 = self._layout[dept1]
        slot2 = self._layout[dept2]

        d_id1 = self.name_id_to_id.get(dept1, -1)
        d_id2 = self.name_id_to_id.get(dept2, -1)

        is_swapable:bool = (self.area_dict.get((d_id1, d_id2), False) or self.area_dict.get((d_id2, d_id1), False))
        if not is_swapable:
            self.logger.debug(f"Swap between {dept1} and {dept2} violates area compatibility constraint. Swap reverted.")
            return None

        mask_id1 = self.np_times[:, :2] == d_id1
        mask_id2 = self.np_times[:, :2] == d_id2

        self.np_times[:, :2][mask_id1] = d_id2
        self.np_times[:, :2][mask_id2] = d_id1

        self._sort_np_matrices()
        
        self._layout[dept1] = slot2
        self._layout[dept2] = slot1

        self._slot_layout[slot1] = dept2
        self._slot_layout[slot2] = dept1

        return self.current_travel_cost