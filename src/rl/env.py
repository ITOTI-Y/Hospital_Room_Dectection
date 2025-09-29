import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Optional

from src.config.config_loader import ConfigLoader
from src.pipeline import PathwayGenerator, CostManager
from src.utils.logger import setup_logger


class LayoutEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: ConfigLoader, max_departments: int, max_step: int):
        super().__init__()

        self.config = config
        self.logger = setup_logger(__name__)
        self.max_departments = max_departments
        self.max_step = max_step

        # [service_time, service_weight, area, x, y, z]
        self.numerical_feature_dim = 6
        self.categorical_feature_dim = 1  # [name]

        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config, is_shuffle=True)

        self.norm_numerical_feature: Optional[np.ndarray] = None
        self.index_to_dept_id: Dict[int, str] = {}
        self.dept_id_to_index: Dict[str, int] = {}
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
                    high=self.num_total_slot - 1,
                    shape=(self.max_departments,),
                    dtype=np.int32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=self.max_departments - 1,
                    shape=(2, self.E_max),
                    dtype=np.int32,
                ),
                "edge_weight": spaces.Box(
                    low=0, high=1, shape=(self.E_max,), dtype=np.float32
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

    def _precompute_normalization_stats(self) -> None:
        pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways=pathways)

        features = self.cost_manager.slots[
            ["service_time", "service_weight", "area", "pos_x", "pos_y", "pos_z"]
        ]
        self.scaler.fit(features)

    def _precompute_categorical_features(self):
        slots_name_ids = self.cost_manager.slots_name_id
        self.num_total_travel_node = len(self.cost_manager.travel_times.index)
        self.num_total_slot = len(slots_name_ids)
        self.index_to_dept_id = {i: name for i, name in enumerate(slots_name_ids)}
        self.dept_id_to_index = {name: i for i, name in enumerate(slots_name_ids)}

    def step(self, action: np.ndarray):
        self.current_step += 1

        idx1: int = action[0].astype(int)
        idx2: int = action[1].astype(int)

        if idx1 >= self.num_total_slot or idx2 >= self.num_total_slot or idx1 == idx2:
            reward: float = self.config.constraints.invalid_action
            observation = self._get_observation()
            done = self.current_step >= self.max_step
            info = self._get_info()
            self.logger.warning(f"Invalid action: {action}, reward: {reward}")
            return observation, reward, done, False, info
        
        dept1 = self.index_to_dept_id[idx1]
        dept2 = self.index_to_dept_id[idx2]

        previous_cost = self.current_cost
        new_cost = self.cost_engine.swap(dept1, dept2)
        
        step_penalty = self.config.constraints.step_penalty
        if new_cost is None:
            reward: float = self.config.constraints.invalid_action
        else:
            cost_diff = previous_cost - new_cost
            reward = cost_diff / (self.initial_cost + 1e-6)
            self.current_cost = new_cost
        
        reward += step_penalty

        terminated = False
        truncated = self.current_step >= self.max_step

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def reset(self, seed: Optional[int] = None) -> tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.logger.info("Resetting environment")

        pathways = self.pathway_generator.generate_all()
        # self.cost_manager = CostManager(self.config, is_shuffle=True) # If needed to shuffle travel_matrix
        self.cost_manager.initialize(pathways=pathways)
        self.cost_engine = self.cost_manager.create_cost_engine()

        self.current_step = 0
        self.initial_cost = self.cost_engine.current_travel_cost
        self.current_cost = self.initial_cost

        self.logger.info(
            f"New instance created. Active departments: {self.num_total_slot}, Initial travel cost: {self.initial_cost}"
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        x_numerical = np.zeros(
            (self.max_departments, self.numerical_feature_dim), dtype=np.float32
        )
        x_categorical = np.zeros((self.max_departments,), dtype=np.int32)
        node_mask = np.zeros((self.max_departments,), dtype=np.int32)

        x_norm_numerical = self.scaler.transform(
            self.cost_manager.slots[
                ["service_time", "service_weight", "area", "pos_x", "pos_y", "pos_z"]
            ]
        )
        x_numerical[: self.num_total_slot, :] = x_norm_numerical

        x_categorical[: self.num_total_slot] = np.array(
            [i for i in self.index_to_dept_id.keys()]
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
        
        return {
            "x_numerical": x_numerical,
            "x_categorical": x_categorical,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }

    def _get_info(self) -> Dict:
        return {
            "current_cost": self.current_cost,
            "initial_cost": self.initial_cost,
            "step": self.current_step,
            "num_departments": self.num_total_slot,
        }

    def render(self, mode:str="human"):
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Current Total Travel Cost: {self.current_cost:.2f} (Initial: {self.initial_cost:.2f})")
            print(f"Improvement: {(self.initial_cost - self.current_cost) / (self.initial_cost + 1e-6) * 100:.2f}%")