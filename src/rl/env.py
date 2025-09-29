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
        self.index_to_dept: Dict[int, str] = {}
        self.dept_to_index: Dict[str, int] = {}
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
        names = self.cost_manager.slots["name"].tolist()
        self.num_total_travel_node = len(self.cost_manager.travel_times.index)
        self.num_total_slot = len(names)
        self.index_to_dept = {i: name for i, name in enumerate(names)}
        self.dept_to_index = {name: i for i, name in enumerate(names)}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
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

    def _get_observation(self):
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
            [i for i in self.index_to_dept.keys()]
        )
        node_mask[: self.num_total_slot] = 1

        edge_index = np.zeros((2, self.E_max), dtype=np.int32)
        edge_weight = np.zeros((self.E_max,), dtype=np.float32)
        edge_mask = np.zeros((self.E_max,), dtype=np.int32)

        slot_name_id_edge_weights = self.cost_engine.slot_name_id_edge_weights
        for i, (name_id1, name_id2, weight) in enumerate(slot_name_id_edge_weights):
            name1 = name_id1.split("_")[0]
            name2 = name_id2.split("_")[0]
            if name1 in self.dept_to_index and name2 in self.dept_to_index:
                idx1 = self.dept_to_index[name1]
                idx2 = self.dept_to_index[name2]
                edge_index[0, i] = idx1
                edge_index[1, i] = idx2
                edge_weight[i] = weight
                edge_mask[i] = 1
        pass

    def _get_info(self):
        pass
