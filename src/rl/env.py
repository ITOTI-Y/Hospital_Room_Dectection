import uuid
from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.config.config_loader import ConfigLoader
from src.pipeline import CostManager, PathwayGenerator


@dataclass
class GraphObservation:
    x_numerical: np.ndarray  # Shape: (max_departments, numerical_feature_dim)
    x_categorical: np.ndarray  # Shape: (max_departments,)
    edge_index: np.ndarray  # Shape: (2, E_max)
    edge_weight: np.ndarray  # Shape: (E_max,)
    node_mask: np.ndarray  # Shape: (max_departments,)
    edge_mask: np.ndarray  # Shape: (E_max,)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "x_numerical": self.x_numerical,
            "x_categorical": self.x_categorical,
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
            "node_mask": self.node_mask,
            "edge_mask": self.edge_mask,
        }

    def __repr__(self) -> str:
        return (
            f"GraphObservation(\n"
            f" x_numerical shape: {self.x_numerical.shape},\n"
            f" x_categorical shape: {self.x_categorical.shape},\n"
            f" edge_index shape: {self.edge_index.shape},\n"
            f" edge_weight shape: {self.edge_weight.shape},\n"
            f" node_mask shape: {self.node_mask.shape},\n"
            f" edge_mask shape: {self.edge_mask.shape},\n"
            f")"
        )


class LayoutEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}

    def __init__(self, config: ConfigLoader, max_departments: int, max_step: int, patience: int = 50):
        super().__init__()
        self.env_id = str(uuid.uuid4())[:8]

        self.config = config
        self.logger = logger.bind(module=__name__)
        self.max_departments = max_departments
        self.PADDING_IDX = max_departments - 1
        self.max_step = max_step
        self.patience = patience  # Early stopping: terminate if no improvement for N steps

        # [service_time, service_weight, area, x, y, z, action_flag]
        self.categorical_feature_dim = 1  # [name]
        self.fixable_features: np.ndarray = np.zeros(
            (self.max_departments, 3), dtype=np.float32
        )
        self.moveable_features: np.ndarray = np.zeros(
            (self.max_departments, 3), dtype=np.float32
        )
        self.action_flag_feature: np.ndarray = np.zeros(
            (self.max_departments, 1), dtype=np.float32
        )
        self.numerical_feature_dim = (
            self.fixable_features.shape[1]
            + self.moveable_features.shape[1]
            + self.action_flag_feature.shape[1]
        )

        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config, is_shuffle=True)

        self.norm_numerical_feature: np.ndarray | None = None
        self.index_to_dept_id: dict[int, str] = {}
        self.dept_id_to_index: dict[str, int] = {}
        self.num_total_slot: int = 0
        self.num_total_travel_node: int = 0
        self.scaler: StandardScaler = StandardScaler()
        self._precompute_normalization_stats()
        self._precompute_categorical_features()

        self.E_max = self.max_departments * (self.max_departments - 1) // 2
        self.observation_space = spaces.Dict(
            {
                "x_numerical": spaces.Box(
                    low=-5,
                    high=5,
                    shape=(self.max_departments, self.numerical_feature_dim),
                    dtype=np.float32,
                ),
                "x_categorical": spaces.Box(
                    low=0,
                    high=self.max_departments - 1,
                    shape=(self.max_departments,),
                    dtype=np.int32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=-1,
                    shape=(2, self.E_max),
                    dtype=np.int32,
                ),
                "edge_weight": spaces.Box(
                    low=-1, high=1, shape=(self.E_max,), dtype=np.float32
                ),
                "node_mask": spaces.MultiBinary(self.max_departments),
                "edge_mask": spaces.MultiBinary(self.E_max),
                # New: flow matrix for dynamic adaptation
                "flow_matrix": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_departments, self.max_departments),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(
            [self.max_departments, self.max_departments]
        )

        self.current_step = 0
        self.current_cost = 0.0
        self.initial_cost = 0.0
        self.best_cost = float('inf')  # Track best cost for early stopping
        self._no_improvement_count = 0  # Count steps without improvement

        self.last_failed_action: np.ndarray | None = None

        # Cost normalizer for reward scaling (computed on first reset)
        self.cost_normalizer: float | None = None

        # Flow matrix for dynamic adaptation (injected in reset or manually)
        self.flow_matrix: np.ndarray | None = None

    def _precompute_normalization_stats(self) -> None:
        pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways=pathways)

        fixable_features = self.cost_manager.slots[
            ["service_time", "service_weight", "area"]
        ].to_numpy()
        moveable_features = self.cost_manager.slots[
            ["pos_x", "pos_y", "pos_z"]
        ].to_numpy()

        self.fixable_features[: len(fixable_features)] = fixable_features
        self.moveable_features[: len(moveable_features)] = moveable_features

        self.scaler.fit(
            np.concatenate(
                [
                    self.fixable_features,
                    self.moveable_features,
                    self.action_flag_feature,
                ],
                axis=1,
            )
        )

    def _precompute_categorical_features(self):
        slots_name_ids = self.cost_manager.slots_name_id
        self.num_total_travel_node = len(self.cost_manager.travel_times.index)
        self.num_total_slot = len(slots_name_ids)
        self.index_to_dept_id = dict(enumerate(slots_name_ids))
        self.dept_id_to_index = {name: i for i, name in enumerate(slots_name_ids)}

    def _compute_numerical_features(self):
        pass

    def _estimate_cost_scale(self) -> float:
        """Estimate cost scale for reward normalization using travel time statistics."""
        times = self.cost_manager.travel_times.values
        valid_times = times[times > 0]
        if len(valid_times) == 0:
            return 1.0
        # Use standard deviation of travel times scaled by number of departments
        return float(np.std(valid_times) * self.num_total_slot)

    def _extract_flow_matrix(self) -> np.ndarray:
        """Extract normalized flow matrix from pair_weights.

        Returns:
            Flow matrix (num_total_slot, num_total_slot) with values in [0, 1]
        """
        n = self.num_total_slot
        flow = np.zeros((n, n), dtype=np.float32)

        for (dept1, dept2), weight in self.cost_manager.pair_weights.items():
            if dept1 in self.dept_id_to_index and dept2 in self.dept_id_to_index:
                i = self.dept_id_to_index[dept1]
                j = self.dept_id_to_index[dept2]
                flow[i, j] = weight
                flow[j, i] = weight  # Ensure symmetry

        # Normalize to [0, 1]
        if flow.max() > 0:
            flow = flow / flow.max()

        return flow

    def _update_cost_engine_with_flow(self, flow_matrix: np.ndarray) -> None:
        """Update cost engine's pair_weights from flow matrix.

        Args:
            flow_matrix: Flow matrix (num_total_slot, num_total_slot)
        """
        # Update pair_weights based on flow matrix
        for (dept1, dept2) in self.cost_manager.pair_weights.keys():
            if dept1 in self.dept_id_to_index and dept2 in self.dept_id_to_index:
                i = self.dept_id_to_index[dept1]
                j = self.dept_id_to_index[dept2]
                self.cost_manager.pair_weights[(dept1, dept2)] = flow_matrix[i, j]

        # Recreate cost engine with updated weights
        self.cost_engine = self.cost_manager.create_cost_engine()

    def step(self, action: np.ndarray):
        self.current_step += 1
        self.logger.info(
            f"Env {self.env_id} Step {self.current_step}, Action taken: {action}"
        )

        total_reward: float = 0.0

        idx1: int = action[0].astype(int)
        idx2: int = action[1].astype(int)

        if idx1 >= self.num_total_slot or idx2 >= self.num_total_slot or idx1 == idx2:
            reward: float = self.config.constraints.invalid_action  # type: ignore
            self.last_failed_action = action.astype(dtype=int)
            self.action_flag_feature[idx1, 0] += 1.0
            self.action_flag_feature[idx2, 0] += 1.0
            observation = self._get_observation()
            terminated = False
            truncated = self.current_step >= self.max_step
            info = self._get_info()
            self.logger.warning(f"Invalid action: {action}, reward: {reward}")
            return observation, total_reward + reward, terminated, truncated, info

        dept1 = self.index_to_dept_id[idx1]
        dept2 = self.index_to_dept_id[idx2]

        previous_cost: float = self.current_cost
        new_cost, is_swapable = self.cost_engine.swap(dept1, dept2)
        if not is_swapable:
            reward: float = self.config.constraints.invalid_action  # type: ignore
            self.last_failed_action = action.astype(dtype=int)
            self.action_flag_feature[idx1, 0] += 1.0
            self.action_flag_feature[idx2, 0] += 1.0
        else:
            # Valid swap: use normalized improvement as reward
            self.action_flag_feature[:, 0] = 0.0
            improvement = previous_cost - new_cost
            # Normalize by cost scale for better gradient stability
            reward = improvement / (self.cost_normalizer + 1e-6)  # type: ignore

        self.current_cost = new_cost

        # Track best cost for early stopping
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        total_reward += reward

        self.logger.info(
            f"Step {self.current_step}: Swapped {dept1} <-> {dept2}, reward: {reward:.4f}, current_cost: {self.current_cost:.2f}, best_cost: {self.best_cost:.2f}"
        )

        # Early stopping if no improvement for patience steps
        terminated = self._no_improvement_count >= self.patience
        truncated = self.current_step >= self.max_step

        if terminated:
            self.logger.info(f"Early stopping: no improvement for {self.patience} steps")

        observation = self._get_observation()
        info = self._get_info()

        return observation, total_reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        flow_matrix: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset environment with optional custom flow matrix.

        Args:
            seed: Random seed
            flow_matrix: Optional custom flow matrix for dynamic adaptation.
                         If None, uses default flow from pathways.

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self.logger.info("Resetting environment")

        pathways = self.pathway_generator.generate_all()
        # self.cost_manager = CostManager(self.config, is_shuffle=True) # If needed to shuffle travel_matrix
        self.cost_manager.initialize(pathways=pathways)
        self.cost_engine = self.cost_manager.create_cost_engine()

        # Handle custom flow matrix injection
        if flow_matrix is not None:
            self.logger.info("Injecting custom flow matrix")
            self.flow_matrix = flow_matrix[:self.num_total_slot, :self.num_total_slot]
            self._update_cost_engine_with_flow(self.flow_matrix)
        else:
            # Extract default flow from pathways
            self.flow_matrix = self._extract_flow_matrix()

        self.current_step = 0
        self.initial_cost = self.cost_engine.current_travel_cost
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost
        self._no_improvement_count = 0

        # Compute cost normalizer on first reset
        if self.cost_normalizer is None:
            self.cost_normalizer = self._estimate_cost_scale()
            self.logger.info(f"Cost normalizer initialized: {self.cost_normalizer:.2f}")

        self.logger.info(
            f"New instance created. Active departments: {self.num_total_slot}, Initial travel cost: {self.initial_cost}"
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get observation including flow matrix for dynamic adaptation.

        Returns:
            Dictionary observation with all required fields
        """
        x_categorical = (
            np.ones((self.max_departments,), dtype=np.int32) * self.PADDING_IDX
        )
        node_mask = np.zeros((self.max_departments,), dtype=np.int32)

        shuffled_indexes = [
            self.dept_id_to_index[slot_id]
            for slot_id in self.cost_engine.layout.values()
        ]
        moveable_feature = np.zeros_like(self.moveable_features)
        moveable_feature[: self.num_total_slot, :] = self.moveable_features[
            shuffled_indexes
        ]

        x_norm_numerical = np.asarray(
            self.scaler.transform(
                np.concatenate(
                    (self.fixable_features, moveable_feature, self.action_flag_feature),
                    axis=1,
                )
            ),
            dtype=np.float32,
        )

        x_categorical[: self.num_total_slot] = np.array(
            list(self.index_to_dept_id.keys())
        )
        node_mask[: self.num_total_slot] = 1

        edge_index = np.ones((2, self.E_max), dtype=np.int32) * -1
        edge_weight = np.zeros((self.E_max,), dtype=np.float32)
        edge_mask = np.zeros((self.E_max,), dtype=np.int32)

        slot_name_id_edge_weights = self.cost_engine.slot_name_id_edge_weights
        for i, (name_id1, name_id2, weight) in enumerate(slot_name_id_edge_weights):
            if name_id1 in self.dept_id_to_index and name_id2 in self.dept_id_to_index:
                idx1 = self.dept_id_to_index[name_id1]
                idx2 = self.dept_id_to_index[name_id2]
                edge_index[0, i] = idx1
                edge_index[1, i] = idx2
                edge_weight[i] = weight
                edge_mask[i] = 1

        # Add flow matrix to observation
        padded_flow = np.zeros((self.max_departments, self.max_departments), dtype=np.float32)
        if self.flow_matrix is not None:
            n = min(self.flow_matrix.shape[0], self.max_departments)
            padded_flow[:n, :n] = self.flow_matrix[:n, :n]

        return {
            "x_numerical": x_norm_numerical,
            "x_categorical": x_categorical,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "flow_matrix": padded_flow,
        }

    def _get_info(self) -> dict:
        return {
            "current_cost": self.current_cost,
            "initial_cost": self.initial_cost,
            "step": self.current_step,
            "num_departments": self.num_total_slot,
        }

    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(
                f"Current Total Travel Cost: {self.current_cost:.2f} (Initial: {self.initial_cost:.2f})"
            )
            print(
                f"Improvement: {(self.initial_cost - self.current_cost) / (self.initial_cost + 1e-6) * 100:.2f}%"
            )
