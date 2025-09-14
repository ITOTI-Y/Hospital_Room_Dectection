"""Configuration module for the optimization project."""

import pathlib
from typing import List


class RLConfig:
    """Stores configuration parameters for RL and other optimization algorithms."""

    def __init__(self):
        """Initializes all configuration parameters."""
        # Path configurations
        self.ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent
        self.RL_OPTIMIZER_PATH: pathlib.Path = self.ROOT_PATH / "src" / "rl_optimizer"
        self.DATA_PATH: pathlib.Path = self.RL_OPTIMIZER_PATH / "data"
        self.CACHE_PATH: pathlib.Path = self.DATA_PATH / "cache"
        self.LOG_PATH: pathlib.Path = self.ROOT_PATH / "logs"
        self.RESULT_PATH: pathlib.Path = self.ROOT_PATH / "results"

        # Input files
        self.TRAVEL_TIMES_CSV: pathlib.Path = (
            self.RESULT_PATH / "network" / "hospital_travel_times.csv"
        )
        self.PROCESS_TEMPLATES_JSON: pathlib.Path = (
            self.DATA_PATH / "process_templates_traditional.json"
        )

        # Auto-generated/cached intermediate files
        self.NODE_VARIANTS_JSON: pathlib.Path = self.CACHE_PATH / "node_variants.json"
        self.TRAFFIC_DISTRIBUTION_JSON: pathlib.Path = (
            self.CACHE_PATH / "traffic_distribution.json"
        )
        self.RESOLVED_PATHWAYS_PKL: pathlib.Path = (
            self.CACHE_PATH / "resolved_pathways.pkl"
        )
        self.COST_MATRIX_CACHE: pathlib.Path = (
            self.CACHE_PATH / "cost_precomputation.npz"
        )

        # Constraint configurations
        self.AREA_SCALING_FACTOR: float = 0.1
        self.EMPTY_SLOT_PENALTY_FACTOR: float = 10000.0
        self.ALLOW_PARTIAL_LAYOUT: bool = True
        self.MANDATORY_ADJACENCY: List[List[str]] = []
        self.FIXED_NODE_TYPES: List[str] = [
            "门",
            "楼梯",
            "电梯",
            "扶梯",
            "走廊",
            "墙",
            "栏杆",
            "室外",
            "绿化",
            "中庭",
            "空房间",
            "急诊科",
            "挂号收费",
        ]

        # --- Heuristic Algorithm Defaults ---
        # Simulated Annealing
        self.SA_DEFAULT_INITIAL_TEMP: float = 1000.0
        self.SA_DEFAULT_FINAL_TEMP: float = 0.1
        self.SA_DEFAULT_COOLING_RATE: float = 0.95
        self.SA_DEFAULT_TEMPERATURE_LENGTH: int = 100
        self.SA_DEFAULT_MAX_ITERATIONS: int = 10000
        self.SA_MAX_REPAIR_ATTEMPTS: int = 10

        # Genetic Algorithm
        self.GA_DEFAULT_POPULATION_SIZE: int = 300
        self.GA_DEFAULT_ELITE_SIZE: int = 20
        self.GA_DEFAULT_MUTATION_RATE: float = 0.15
        self.GA_DEFAULT_CROSSOVER_RATE: float = 0.85
        self.GA_DEFAULT_TOURNAMENT_SIZE: int = 5
        self.GA_DEFAULT_MAX_AGE: int = 100
        self.GA_DEFAULT_CONVERGENCE_THRESHOLD: int = 300
        self.GA_DEFAULT_MAX_ITERATIONS: int = 10000
        self.GA_MAX_REPAIR_ATTEMPTS: int = 5

        # --- PPO & RL Configurations ---
        # Transformer model
        self.EMBEDDING_DIM: int = 128
        self.FEATURES_DIM: int = 128
        self.TRANSFORMER_HEADS: int = 4
        self.TRANSFORMER_LAYERS: int = 4
        self.TRANSFORMER_DROPOUT: float = 0.1

        # Policy network
        self.POLICY_NET_ARCH: int = 128
        self.POLICY_NET_LAYERS: int = 2
        self.VALUE_NET_ARCH: int = 128
        self.VALUE_NET_LAYERS: int = 2

        # Learning rate scheduler
        self.LEARNING_RATE_SCHEDULE_TYPE: str = "linear"
        self.LEARNING_RATE_INITIAL: float = 1e-4
        self.LEARNING_RATE_FINAL: float = 1e-8
        self.LEARNING_RATE: float = 1e-4

        # PPO training hyperparameters
        self.NUM_ENVS: int = 8
        self.N_STEPS: int = 512
        self.TOTAL_TIMESTEPS: int = 5_000_000
        self.GAMMA: float = 0.99
        self.GAE_LAMBDA: float = 0.95
        self.CLIP_RANGE: float = 0.2
        self.ENT_COEF: float = 0.05
        self.VF_COEF: float = 0.5
        self.MAX_GRAD_NORM: float = 0.5
        self.BATCH_SIZE: int = 64
        self.N_EPOCHS: int = 10

        # Checkpoint and evaluation
        self.EVAL_FREQUENCY: int = 10000
        self.RESUME_TRAINING: bool = False
        self.PRETRAINED_MODEL_PATH: str = "data/model"
        self.CHECKPOINT_FREQUENCY: int = 50000
        self.SAVE_TRAINING_STATE: bool = True

        # Ensure critical paths exist
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_PATH.mkdir(parents=True, exist_ok=True)
