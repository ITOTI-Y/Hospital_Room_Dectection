"""
Hospital Layout Optimization: CostManager and CostEngine

=== CRITICAL CONCEPTUAL DISTINCTION ===

This module carefully distinguishes between three related but different concepts:

1. SLOT: A physical location in the building.
   - Has fixed properties: area, (x, y, z) coordinates
   - Has fixed distances to other slots (building geometry)
   - Indexed by slot_idx: 0, 1, 2, ...
   - The distance_matrix[slot_a, slot_b] NEVER changes

2. DEPARTMENT INSTANCE: A functional unit that can be assigned to a slot.
   - Has properties: service_time, department type
   - In this problem, we have exactly n departments and n slots (1:1 mapping)
   - Indexed by dept_idx: 0, 1, 2, ...
   - Initially, dept_idx == slot_idx (identity mapping)

3. PATIENT FLOW: Movement of patients between departments.
   - Defined by clinical pathways (patient flow)
   - flow_matrix[dept_i, dept_j] represents patient traffic between departments
   - This is FIXED regardless of where departments are physically located
   - Patients go "from Pharmacy to Lab", not "from Location A to Location B"

=== THE OPTIMIZATION PROBLEM ===

We want to find the best assignment of departments to slots.

    Total Cost = Σ flow[dept_i, dept_j] * distance[slot_of(dept_i), slot_of(dept_j)]

Where:
- flow[dept_i, dept_j]: Patient flow between departments (FIXED, from pathways)
- distance[slot_a, slot_b]: Physical distance between slots (FIXED, from building)
- slot_of(dept_i): Which slot department i is assigned to (VARIABLE, what we optimize)

=== DATA FLOW ===

    travel_times.csv     →  distance_matrix (slot * slot)     [STATIC]
    slots.csv            →  slot properties (area, position)  [STATIC]
    pathways (generated) →  flow_matrix (dept * dept)         [PER-EPISODE]

    dept_to_slot mapping →  current layout                    [DYNAMIC, changes on swap]
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config.config_loader import ConfigLoader


@dataclass
class LayoutState:
    """
    Represents the current assignment of departments to slots.

    This is the ONLY thing that changes during optimization.

    Attributes:
        dept_to_slot: dept_to_slot[dept_idx] = slot_idx
            "Department i is currently located at slot j"
        slot_to_dept: slot_to_dept[slot_idx] = dept_idx
            "Slot j is currently occupied by department i" (reverse index)

    Example:
        Initial state (identity mapping):
            dept_to_slot = [0, 1, 2, 3]  # dept 0 → slot 0, dept 1 → slot 1, ...
            slot_to_dept = [0, 1, 2, 3]  # slot 0 ← dept 0, slot 1 ← dept 1, ...

        After swapping dept 0 and dept 2:
            dept_to_slot = [2, 1, 0, 3]  # dept 0 → slot 2, dept 2 → slot 0
            slot_to_dept = [2, 1, 0, 3]  # slot 0 ← dept 2, slot 2 ← dept 0
    """

    dept_to_slot: np.ndarray  # shape: (n_depts,), dtype: int32
    slot_to_dept: np.ndarray  # shape: (n_slots,), dtype: int32

    def swap(self, dept_idx1: int, dept_idx2: int) -> None:
        """
        Swap the slot assignments of two departments.

        Before: dept1 → slot_a, dept2 → slot_b
        After:  dept1 → slot_b, dept2 → slot_a
        """

        slot_a = self.dept_to_slot[dept_idx1]
        slot_b = self.dept_to_slot[dept_idx2]

        self.dept_to_slot[dept_idx1] = slot_b
        self.dept_to_slot[dept_idx2] = slot_a

        self.slot_to_dept[slot_a] = dept_idx2
        self.slot_to_dept[slot_b] = dept_idx1

    def copy(self) -> 'LayoutState':
        return LayoutState(
            dept_to_slot=self.dept_to_slot.copy(),
            slot_to_dept=self.slot_to_dept.copy(),
        )


@dataclass
class SlotData:
    """
    Static physical properties of slots (building structure).

    This data is immutable and shared across all CostEngine instances.

    Attributes:
        distance_matrix: distance_matrix[slot_a, slot_b] = travel time in seconds
            Physical distance between any two slots. NEVER changes.

        area_vector: area_vector[slot_idx] = area in square meters
            Physical area of each slot. Used for compatibility checking.

        position_matrix: position_matrix[slot_idx] = (x, y, z) coordinates
            Physical location of each slot center.

        slot_names: Human-readable names like "Pharmacy_20001"
            Used for debugging and output, NOT for indexing in hot path.
    """

    distance_matrix: np.ndarray  # (n_slots, n_slots), dtype: float32
    area_vector: np.ndarray  # (n_slots,), dtype: float32
    position_matrix: np.ndarray  # (n_slots, 3), dtype: float32
    slot_names: list[str]  # (n_slots,), dtype: str
    slot_name_to_idx: dict[str, int]  # "Pharmacy_20001" → slot_idx

    max_distance: float
    min_distance: float

    @property
    def n_slots(self) -> int:
        return len(self.slot_names)


@dataclass
class DepartmentData:
    """
    Static properties of department instances.

    In this problem, we have n departments that map 1:1 with n slots.
    Each department has intrinsic properties that don't depend on location.

    Attributes:
        service_times: service_times[dept_idx] = service time in seconds
            How long patients spend at this department.

        dept_types: dept_types[dept_idx] = type name (e.g., "Pharmacy")
            The functional type of each department instance.

        dept_names: Full identifier like "Pharmacy_20001"
            In this problem, dept_names[i] == slot_names[i] initially.
    """

    service_times: np.ndarray  # # (n_depts,), dtype: float32
    dept_types: list[str]  # (n_depts,), dtype: str
    dept_names: list[str]  # (n_depts,), dtype: str
    dept_name_to_idx: dict[str, int]  # "Pharmacy_20001" → dept_idx

    # Mapping from type name to list of department indices
    # e.g., {"Pharmacy": [0], "Radiology": [3, 15], ...}
    type_to_dept_indices: dict[str, list[int]]  # (n_types,), dtype: list[int]

    @property
    def n_depts(self) -> int:
        return len(self.dept_names)


@dataclass
class FlowData:
    """
    Patient flow data computed from clinical pathways.

    CRITICAL: This is indexed by DEPARTMENT indices, NOT slot indices!
    The flow between departments is determined by clinical processes,
    not by where departments are physically located.

    Attributes:
        flow_matrix: flow_matrix[dept_i, dept_j] = patient flow weight
            How many patients (weighted) travel from dept_i to dept_j.
            This matrix is SYMMETRIC and FIXED once computed.

        service_weights: service_weights[dept_idx] = total visits
            How many patient visits each department receives.

        nonzero_dept_pairs: List of (dept_i, dept_j, weight) for non-zero flows
            Precomputed for efficient iteration. Only upper triangle (i < j).

        total_service_cost: Sum of (service_time × service_weight) for all depts
            This is constant regardless of layout.
    """

    flow_matrix: np.ndarray  # (n_depts, n_depts), dtype: float32
    service_weights: np.ndarray  # (n_depts,), dtype: float32
    nonzero_dept_pairs: np.ndarray  # (n_edges, 3): [dept_i, dept_j, weight]
    total_service_cost: float  # dtype: float32

    @property
    def n_edges(self) -> int:
        return len(self.nonzero_dept_pairs)


@dataclass
class ConstraintData:
    """
    Constraint matrices for the optimization.

    Attributes:
        area_compatibility: area_compatibility[dept_idx, slot_idx]
            0.0 if department can fit in slot, negative penalty otherwise.
            Note: This is (dept x slot), not (slot x slot)!

        adjacency_preference: adjacency_preference[dept_i, dept_j]
            Positive weight if these departments should be close together.
    """

    area_compatibility: np.ndarray  # (n_depts, n_slots), dtype: float32
    adjacency_preference: np.ndarray  # (n_depts, n_depts), dtype: float32


class CostManager:
    """
    Manages data loading, preprocessing, and CostEngine creation.

    Responsibilities:
    1. Load travel_times.csv → SlotData (physical building properties)
    2. Load slots.csv → DepartmentData (department properties)
    3. Process pathways → FlowData (patient flow patterns)
    4. Compute constraints → ConstraintData
    5. Create CostEngine instances for RL environments

    Usage:
        manager = CostManager(config)
        manager.initialize(pathways)  # Call once per episode
        engine = manager.create_cost_engine()  # Create per environment
    """

    def __init__(self, config: ConfigLoader, shuffle_initial_layout: bool = False):
        self.logger = logger
        self.config = config
        self.shuffle_initial_layout = shuffle_initial_layout

        # Load Static data (once, shared across all episodes)
        self._slot_data, self._dept_data = self._load_static_data()
        self._constraint_data = self._compute_constraints()

        # Flow data is dynamic, computed per epoch or episode
        self._flow_data: FlowData | None = None

        # Initial Layout (optional, shuffled or identity)
        self._initial_dept_to_slot: np.ndarray = self._create_intitial_layout()

        self._cname_to_ename = self._build_name_mapping()

        self.logger.info(
            f'CostManager initialized with {self._slot_data.n_slots} slots, '
            f'shuffle={self.shuffle_initial_layout}'
        )

    def _load_static_data(self) -> tuple[SlotData, DepartmentData]:
        """
        Load slot and department data from CSV files.

        Returns:
            tuple[SlotData, DepartmentData]: Tuple containing SlotData and DepartmentData objects.
        """
        df_times = pd.read_csv(self.config.paths.travel_times_csv, index_col=0)

        slot_names = list(df_times.index)
        slot_name_to_idx = {name: idx for idx, name in enumerate(slot_names)}
        n = len(slot_names)

        distance_matrix = df_times.values.astype(np.float32)

        positive_distances = distance_matrix[distance_matrix > 0]
        max_distance = (
            float(distance_matrix.max()) if len(positive_distances) > 0 else 1.0
        )
        min_distance = (
            float(positive_distances.min()) if len(positive_distances) > 0 else 0.0
        )

        df_slots = pd.read_csv(self.config.paths.slots_csv)

        area_vector = np.zeros(n, dtype=np.float32)
        position_matrix = np.zeros((n, 3), dtype=np.float32)
        service_times = np.zeros(n, dtype=np.float32)
        dept_types = [''] * n

        node_def = self.config.graph_config.node_definitions

        for _, row in df_slots.iterrows():
            name = f'{row["name"]}_{row["id"]}'
            if name not in slot_name_to_idx:
                self.logger.warning(f'Slot {name} not found in travel_times file')
                continue

            idx = slot_name_to_idx[name]
            area_vector[idx] = float(row['area'])
            position_matrix[idx] = [float(row['x']), float(row['y']), float(row['z'])]
            type_name = str(row['name'])
            dept_types[idx] = type_name

            service_times[idx] = float(node_def[type_name]['service_time'])

        type_to_dept_indices: dict[str, list[int]] = defaultdict(list)
        for idx, dept_type in enumerate(dept_types):
            type_to_dept_indices[dept_type].append(idx)

        slot_data = SlotData(
            distance_matrix=distance_matrix,
            area_vector=area_vector,
            position_matrix=position_matrix,
            slot_names=slot_names,
            slot_name_to_idx=slot_name_to_idx,
            max_distance=max_distance,
            min_distance=min_distance,
        )

        dept_data = DepartmentData(
            service_times=service_times,
            dept_types=dept_types,
            dept_names=slot_names.copy(),
            dept_name_to_idx=slot_name_to_idx,
            type_to_dept_indices=type_to_dept_indices,
        )

        return slot_data, dept_data

    def _compute_constraints(self) -> ConstraintData:
        n = self._slot_data.n_slots
        tolerance = float(self.config.constraints.area_compatibility_tolerance)

        area_compatibility = np.zeros((n, n), dtype=np.float32)
        dept_areas = self._slot_data.area_vector  # Original area of each department
        slot_areas = self._slot_data.area_vector  # Area of each slot

        for dept_i in range(n):
            for slot_j in range(n):
                dept_area = dept_areas[dept_i]
                slot_area = slot_areas[slot_j]
                diff_ratio = abs(dept_area - slot_area) / max(dept_area, slot_area)
                if diff_ratio > tolerance:
                    area_compatibility[dept_i, slot_j] = -diff_ratio

        adj_pref = np.zeros((n, n), dtype=np.float32)
        if adj_prefs := self.config.constraints.adjacency_preferences:
            for pref in adj_prefs:
                weight = float(pref.weight)
                dept_cnames = list(pref.depts)

                all_indices: list[int] = []
                for name in dept_cnames:
                    indices = self._dept_data.type_to_dept_indices[name]
                    all_indices.extend(indices)

                for i in all_indices:
                    for j in all_indices:
                        if i != j:
                            adj_pref[i, j] = weight

        return ConstraintData(
            area_compatibility=area_compatibility,
            adjacency_preference=adj_pref,
        )

    def _create_intitial_layout(self) -> np.ndarray:
        n = self._slot_data.n_slots

        if self.shuffle_initial_layout:
            return np.random.permutation(n).astype(np.int32)
        else:
            return np.arange(n, dtype=np.int32)

    def _build_name_mapping(self) -> dict[str, int]:
        node_def = dict(self.config.graph_config.node_definitions)
        return {v['cname']: k for k, v in node_def.items()}

    def _compute_flow_data(self, pathways: dict[str, dict[str, Any]]) -> FlowData:
        """
        Compute patient flow matrix from pathways

        The flow matrix is indexed by department indices.

        Args:
            pathways (dict[str, dict[str, Any]]): Patient flow data from clinical pathways

        Returns:
            FlowData: FlowData object containing the flow matrix, service weights, and total service cost
        """
        n = self._dept_data.n_depts
        flow_matrix = np.zeros((n, n), dtype=np.float32)
        service_weights = np.zeros(n, dtype=np.float32)

        type_to_indices = self._dept_data.type_to_dept_indices

        for pathway in pathways.values():
            sequence: list[str] = pathway['core_sequence']
            weight = float(pathway['base_weight'])
            start_nodes: list[str] = pathway['start_nodes']
            end_nodes: list[str] = pathway['end_nodes']

            full_sequence: list[str] = start_nodes + sequence + end_nodes

            for i in range(len(full_sequence) - 1):
                from_type = full_sequence[i]
                to_type = full_sequence[i + 1]

                from_dept_indices = type_to_indices.get(from_type, [])
                to_dept_indices = type_to_indices.get(to_type, [])

                n_pairs = len(from_dept_indices) * len(to_dept_indices)
                pair_weight = weight / n_pairs

                for from_idx in from_dept_indices:
                    service_weights[from_idx] += weight
                    for to_idx in to_dept_indices:
                        flow_matrix[from_idx, to_idx] += pair_weight
                        flow_matrix[to_idx, from_idx] += pair_weight

            for idx in type_to_indices.get(full_sequence[-1], []):
                service_weights[idx] += weight

        nonzero_i, nonzero_j = np.nonzero(flow_matrix)
        mask = nonzero_i < nonzero_j
        nonzero_dept_pairs = np.column_stack(
            [
                nonzero_i[mask],
                nonzero_j[mask],
                flow_matrix[nonzero_i[mask], nonzero_j[mask]],
            ]
        ).astype(np.int32)

        total_service_cost = float(
            np.dot(service_weights, self._dept_data.service_times)
        )

        return FlowData(
            flow_matrix=flow_matrix,
            service_weights=service_weights,
            nonzero_dept_pairs=nonzero_dept_pairs,
            total_service_cost=total_service_cost,
        )

    def initialize(self, pathways: dict[str, dict[str, Any]]) -> None:
        self._flow_data = self._compute_flow_data(pathways)

        if self.shuffle_initial_layout:
            self._initial_dept_to_slot = self._create_intitial_layout()

        self.logger.info(
            f'Initialized with {len(pathways)} pathways, '
            f'{self._flow_data.n_edges} non-zero flow pairs'
        )

    def create_cost_engine(self) -> 'CostEngine':
        if self._flow_data is None:
            raise ValueError('Flow data not initialized')

        return CostEngine(
            slot_data=self._slot_data,
            dept_data=self._dept_data,
            constraint_data=self._constraint_data,
            flow_data=self._flow_data,
            initial_dept_to_slot=self._initial_dept_to_slot.copy(),
        )

    @property
    def slot_data(self) -> SlotData:
        return self._slot_data

    @property
    def dept_data(self) -> DepartmentData:
        return self._dept_data

    @property
    def flow_data(self) -> FlowData | None:
        return self._flow_data

    @property
    def constraint_data(self) -> ConstraintData:
        return self._constraint_data

    @property
    def n_slots(self) -> int:
        return self._slot_data.n_slots

    @property
    def n_depts(self) -> int:
        return self._dept_data.n_depts


class CostEngine:
    r"""
    Efficient cost calculation engine for layout optimization.

    Key insight: The cost function is

        Cost = Σ flow[dept_i, dept_j] * distance[slot_of(dept_i), slot_of(dept_j)]
              \_____________________/   \______________________________________/
                     FIXED                      DEPENDS ON LAYOUT

    The flow between departments is fixed (from pathways).
    The distance between slots is fixed (from building geometry).
    Only the dept -> slot mapping changes when we swap departments.

    Operations:
        - swap(i, j): O(1) - Just update the mapping
        - travel_cost: O(E) - Iterate over non-zero flows
        - swap_incremental(i, j): O(D) - Update only affected pairs

    Usage:
        engine = cost_manager.create_cost_engine()

        # Get initial cost
        cost = engine.travel_cost

        # Try a swap
        new_cost, is_valid = engine.swap(dept_i, dept_j)

        # Or use incremental update (faster for sequential swaps)
        new_cost, is_valid = engine.swap_incremental(dept_i, dept_j)

        # Reset to initial layout
        engine.reset()
    """

    def __init__(
        self,
        slot_data: SlotData,
        dept_data: DepartmentData,
        constraint_data: ConstraintData,
        flow_data: FlowData,
        initial_dept_to_slot: np.ndarray,
    ):
        self._slots = slot_data
        self._depts = dept_data
        self._constraints = constraint_data
        self._flow = flow_data

        self._initial_state = LayoutState(
            dept_to_slot=initial_dept_to_slot.copy(),
            slot_to_dept=self._invert_mapping(initial_dept_to_slot),
        )

        self._state = self._initial_state.copy()

        self._cached_travel_cost: float | None = None
        self._last_swap: tuple[int, int] | None = None

    @staticmethod
    def _invert_mapping(dept_to_slot: np.ndarray) -> np.ndarray:
        n = len(dept_to_slot)
        slot_to_dept = np.zeros(n, dtype=np.int32)
        for dept_idx, slot_idx in enumerate(dept_to_slot):
            slot_to_dept[slot_idx] = dept_idx
        return slot_to_dept

    def reset(self) -> None:
        self._state = self._initial_state.copy()
        self._cached_travel_cost = None
        self._last_swap = None

    def swap(self, dept_i: int, dept_j: int) -> tuple[float, bool, bool]:
        """
        Swap two departments's slot assignments.

        Args:
            dept_i (int): Index of first department
            dept_j (int): Index of second department

        Returns:
            tuple[float, bool]: (new_travel_cost, is_area_compatible)
        """

        target_slot_for_i = self._state.dept_to_slot[dept_i]
        target_slot_for_j = self._state.dept_to_slot[dept_j]

        is_compatible = (
            self._constraints.area_compatibility[dept_i, target_slot_for_j] == 0.0
            and self._constraints.area_compatibility[dept_j, target_slot_for_i] == 0.0
        )
        is_repeat = (dept_i, dept_j) == self._last_swap or (
            dept_j,
            dept_i,
        ) == self._last_swap

        self._last_swap = (dept_i, dept_j)
        if not is_compatible:
            return self.travel_cost, is_compatible, is_repeat

        self._state.swap(dept_i, dept_j)
        self._cached_travel_cost = None
        return self.travel_cost, is_compatible, is_repeat

    def swap_incremental(self, dept_i: int, dept_j: int) -> tuple[float, bool, bool]:
        if self._cached_travel_cost is None:
            self._cached_travel_cost = self._compute_travel_cost_full()

        target_slot_for_i = self._state.dept_to_slot[dept_i]
        target_slot_for_j = self._state.dept_to_slot[dept_j]

        is_compatible = (
            self._constraints.area_compatibility[dept_i, target_slot_for_j] == 0.0
            and self._constraints.area_compatibility[dept_j, target_slot_for_i] == 0.0
        )
        is_repeat = (dept_i, dept_j) == self._last_swap or (
            dept_j,
            dept_i,
        ) == self._last_swap

        self._last_swap = (dept_i, dept_j)
        if not is_compatible:
            return self._cached_travel_cost, is_compatible, is_repeat

        old_contribution = (
            self._compute_dept_travel_contribution(dept_i)
            + self._compute_dept_travel_contribution(dept_j)
            - self._compute_pair_travel_cost(dept_i, dept_j)
        )

        self._state.swap(dept_i, dept_j)

        new_contribution = (
            self._compute_dept_travel_contribution(dept_i)
            + self._compute_dept_travel_contribution(dept_j)
            - self._compute_pair_travel_cost(dept_i, dept_j)
        )

        self._cached_travel_cost += new_contribution - old_contribution
        return self._cached_travel_cost, is_compatible, is_repeat

    def _compute_dept_travel_contribution(self, dept_idx: int) -> float:
        """
        Compute total travel cost contribution from one department.

        This is the sum of (flow x distance) for all edges involving dept_idx.
        """
        contribution = 0.0
        slot_of_dept = self._state.dept_to_slot[dept_idx]
        for other_dept in range(self._depts.n_depts):
            flow = self._flow.flow_matrix[dept_idx, other_dept]
            if flow > 0.0:
                slot_of_other = self._state.dept_to_slot[other_dept]
                distance = self._slots.distance_matrix[slot_of_dept, slot_of_other]
                contribution += flow * distance
        return contribution

    def _compute_pair_travel_cost(self, dept_i: int, dept_j: int) -> float:
        """
        Compute travel cost for a single pair of departments.
        """
        flow = self._flow.flow_matrix[dept_i, dept_j]
        if flow == 0.0:
            return 0.0

        slot_i = self._state.dept_to_slot[dept_i]
        slot_j = self._state.dept_to_slot[dept_j]
        distance = self._slots.distance_matrix[slot_i, slot_j]
        return flow * distance

    def _compute_travel_cost_full(self) -> float:
        dept_to_slot = self._state.dept_to_slot
        reordered_distances_matrix = self._slots.distance_matrix[
            np.ix_(dept_to_slot, dept_to_slot)
        ]
        return float(np.sum(self._flow.flow_matrix * reordered_distances_matrix) / 2)

    @property
    def travel_cost(self) -> float:
        """
        Get total weighted travel cost under current layout.

        Cost = Σ flow[dept_i, dept_j] x distance[slot_of(dept_i), slot_of(dept_j)]

        Cached for efficiency. Invalidated on swap().
        """
        if self._cached_travel_cost is None:
            self._cached_travel_cost = self._compute_travel_cost_full()
        return self._cached_travel_cost

    @property
    def area_penalty(self) -> float:
        dept_indices = np.arange(self._depts.n_depts)
        slot_indices = self._state.dept_to_slot
        return float(
            self._constraints.area_compatibility[dept_indices, slot_indices].sum()
        )

    @property
    def adjacency_score(self) -> float:
        score = 0.0
        n = self._depts.n_depts

        for dept_i in range(n):
            for dept_j in range(dept_i + 1, n):
                pref = self._constraints.adjacency_preference[dept_i, dept_j]
                if pref > 0.0:
                    slot_i = self._state.dept_to_slot[dept_i]
                    slot_j = self._state.dept_to_slot[dept_j]
                    distance = self._slots.distance_matrix[slot_i, slot_j]
                    score += pref * pref * (self._slots.max_distance / distance)
        return score
