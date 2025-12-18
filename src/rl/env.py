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

    def __init__(
        self,
        config: ConfigLoader,
        max_departments: int,
        max_step: int,
        is_training: bool = True,
    ):
        super().__init__()
        self.env_id = str(uuid.uuid4())[:8]

        self.config = config
        self.logger = logger.bind(module=__name__)
        self.max_departments = max_departments
        self.PADDING_IDX = max_departments - 1
        self.max_step = max_step
        self.is_training = is_training

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

        self.pathway_generator = PathwayGenerator(self.config, is_training=is_training)
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
            }
        )
        self.action_space = spaces.MultiDiscrete(
            [self.max_departments, self.max_departments]
        )

        self.current_step = 0
        self.current_cost = 0.0
        self.initial_cost = 0.0
        self.best_cost = 0.0  # Track best cost achieved in episode
        self.total_swaps = 0
        self.invalid_swaps = 0
        self.no_change_swaps = 0
        self.cumulative_reward = 0.0

        self.last_failed_action: np.ndarray | None = None
        self.last_action: tuple[int, int] | None = None  # Track last action for repetition penalty

        # Extended tracking for action history (to prevent cycling)
        self.action_history: list[frozenset[int]] = []  # Track all actions as frozensets
        self.action_history_window: int = 10  # Window size for repetition check
        self.steps_without_improvement: int = 0  # For early termination

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

    def _compute_reward(
        self,
        cost_diff: float,
        is_swapable: bool,
        is_invalid: bool,
        action: tuple[int, int],
    ) -> tuple[float, dict[str, float]]:
        """Compute shaped reward for the current step.

        Args:
            cost_diff: previous_cost - new_cost (positive means improvement)
            is_swapable: Whether the swap is valid (area compatible)
            is_invalid: Whether the action was invalid (same dept or out of bounds)
            action: The (idx1, idx2) action taken

        Returns:
            total_reward: The total reward for this step
            reward_components: Dict with breakdown of reward components
        """
        constraints = self.config.constraints

        # Get reward parameters with defaults
        step_penalty: float = getattr(constraints, "step_penalty", -0.01)
        invalid_penalty: float = getattr(constraints, "invalid_action", -1.0)
        no_change_penalty: float = getattr(constraints, "no_improvement_penalty", -0.5)
        improvement_scale: float = getattr(constraints, "improvement_scale", 100.0)
        repetition_penalty: float = getattr(constraints, "repetition_penalty", -1.0)
        exploration_bonus: float = getattr(constraints, "exploration_bonus", 0.1)

        reward_components = {
            "improvement": 0.0,
            "step_penalty": step_penalty,
            "invalid_penalty": 0.0,
            "no_change_penalty": 0.0,
            "area_penalty": 0.0,
            "repetition_penalty": 0.0,
            "exploration_bonus": 0.0,
        }

        # Invalid action penalty
        if is_invalid:
            reward_components["invalid_penalty"] = invalid_penalty
            self.invalid_swaps += 1
            total_reward = step_penalty + invalid_penalty
            return total_reward, reward_components

        # Create action set for comparison (order-independent)
        action_set = frozenset(action)

        # Calculate improvement reward
        # Use relative improvement: cost_diff / initial_cost
        improvement_ratio = cost_diff / (self.initial_cost + 1e-6)
        improvement_reward = improvement_ratio * improvement_scale
        reward_components["improvement"] = improvement_reward

        # No change penalty (swap happened but no cost change)
        if abs(cost_diff) < 1e-6:
            reward_components["no_change_penalty"] = no_change_penalty
            self.no_change_swaps += 1

        # Extended repetition penalty: check if action was in recent history window
        recent_history = self.action_history[-self.action_history_window:]
        action_count_in_window = recent_history.count(action_set)
        if action_count_in_window > 0:
            # Stronger penalty for more repetitions
            reward_components["repetition_penalty"] = repetition_penalty * action_count_in_window

        # Exploration bonus: reward for trying new action pairs
        if action_set not in self.action_history:
            reward_components["exploration_bonus"] = exploration_bonus

        # Add current action to history
        self.action_history.append(action_set)

        # Area incompatibility penalty (swap executed but areas don't match well)
        if not is_swapable:
            area_penalty = self.cost_engine.area_compatibility_cost * 0.1
            reward_components["area_penalty"] = area_penalty

        self.total_swaps += 1

        total_reward = sum(reward_components.values())
        return total_reward, reward_components

    def _compute_terminal_reward(self) -> float:
        """Compute terminal bonus/penalty based on overall episode performance."""
        terminal_scale: float = getattr(
            self.config.constraints, "terminal_bonus_scale", 50.0
        )

        # Calculate overall improvement from initial state
        total_improvement = (self.initial_cost - self.best_cost) / (
            self.initial_cost + 1e-6
        )

        # Bonus for positive improvement, penalty for negative
        terminal_reward = total_improvement * terminal_scale

        return terminal_reward

    def step(self, action: np.ndarray):
        self.current_step += 1

        idx1: int = action[0].astype(int)
        idx2: int = action[1].astype(int)

        # Check for invalid action
        is_invalid = (
            idx1 >= self.num_total_slot or idx2 >= self.num_total_slot or idx1 == idx2
        )

        if is_invalid:
            self.last_failed_action = action.astype(dtype=int)
            self.action_flag_feature[idx1, 0] += 1.0
            self.action_flag_feature[idx2, 0] += 1.0

            reward, components = self._compute_reward(
                cost_diff=0.0, is_swapable=True, is_invalid=True, action=(idx1, idx2)
            )
            self.last_action = (idx1, idx2)
            self.cumulative_reward += reward

            observation = self._get_observation()
            terminated = False
            truncated = self.current_step >= self.max_step

            if truncated:
                terminal_reward = self._compute_terminal_reward()
                reward += terminal_reward
                self._log_episode_end()

            info = self._get_info()
            info["reward_components"] = components

            self.logger.debug(
                f"Invalid action: {action}, reward: {reward:.3f}, "
                f"components: {components}"
            )

            return (
                observation.to_dict(),
                reward,
                terminated,
                truncated,
                info,
            )

        # Valid action - perform swap
        dept1 = self.index_to_dept_id[idx1]
        dept2 = self.index_to_dept_id[idx2]

        previous_cost: float = self.current_cost
        new_cost, is_swapable = self.cost_engine.swap(dept1, dept2)

        if not is_swapable:
            self.last_failed_action = action.astype(dtype=int)
            self.action_flag_feature[idx1, 0] += 1.0
            self.action_flag_feature[idx2, 0] += 1.0
        else:
            self.action_flag_feature[:, 0] = 0.0

        self.current_cost = new_cost
        self.best_cost = min(self.best_cost, new_cost)

        cost_diff: float = previous_cost - new_cost

        # Track steps without improvement for early termination
        if cost_diff > 1e-6:
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        reward, components = self._compute_reward(
            cost_diff=cost_diff, is_swapable=is_swapable, is_invalid=False, action=(idx1, idx2)
        )
        self.last_action = (idx1, idx2)
        self.cumulative_reward += reward

        # Early termination: if no improvement for too long, terminate
        early_termination_steps: int = getattr(
            self.config.constraints, "early_termination_steps", 20
        )
        terminated = self.steps_without_improvement >= early_termination_steps
        truncated = self.current_step >= self.max_step

        if terminated or truncated:
            terminal_reward = self._compute_terminal_reward()
            reward += terminal_reward
            components["terminal"] = terminal_reward
            self._log_episode_end()

        observation = self._get_observation()
        info = self._get_info()
        info["reward_components"] = components

        # Log every 10 steps for debugging
        if self.current_step % 10 == 0:
            self.logger.info(
                f"Step {self.current_step}: {dept1} <-> {dept2}, "
                f"reward: {reward:.3f}, cost: {new_cost:.1f} (change: {cost_diff:.1f})"
            )

        return observation.to_dict(), reward, terminated, truncated, info

    def _log_episode_end(self):
        """Log episode summary statistics."""
        improvement = (self.initial_cost - self.best_cost) / (self.initial_cost + 1e-6)
        early_term = self.steps_without_improvement >= getattr(
            self.config.constraints, "early_termination_steps", 20
        )
        self.logger.info(
            f"Episode end: total_reward={self.cumulative_reward:.2f}, "
            f"improvement={improvement * 100:.2f}%, "
            f"swaps={self.total_swaps}, invalid={self.invalid_swaps}, "
            f"no_change={self.no_change_swaps}, "
            f"unique_actions={len(set(self.action_history))}, "
            f"early_term={early_term}"
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)

        pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways=pathways)
        self.cost_engine = self.cost_manager.create_cost_engine()

        self.current_step = 0
        self.initial_cost = self.cost_engine.current_travel_cost
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost
        self.total_swaps = 0
        self.invalid_swaps = 0
        self.no_change_swaps = 0
        self.cumulative_reward = 0.0
        self.last_action = None  # Reset action tracking
        self.action_history = []  # Reset action history
        self.steps_without_improvement = 0  # Reset early termination counter
        self.action_flag_feature[:, 0] = 0.0

        self.logger.info(
            f"Reset: departments={self.num_total_slot}, initial_cost={self.initial_cost:.1f}"
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation.to_dict(), info

    def _get_observation(self) -> GraphObservation:
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

        return GraphObservation(
            x_numerical=x_norm_numerical,
            x_categorical=x_categorical,
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

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
