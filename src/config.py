"""Configuration module for the network generation project."""

import pathlib
from typing import Dict, Tuple, List, Any, Optional

# Global COLOR_MAP - Consider encapsulating or making it part of a config loader
COLOR_MAP: Dict[Tuple[int, int, int], Dict[str, Any]] = {
    (244, 67, 54): {'name': '内诊药房', 'time': 46},
    (0, 150, 136): {'name': '挂号收费', 'time': 79},
    (103, 58, 183): {'name': '急诊科', 'time': 9000},
    (145, 102, 86): {'name': '中心供应室', 'time': 1},
    (33, 150, 243): {'name': '全科', 'time': 435},
    (3, 169, 244): {'name': '放射科', 'time': 250},
    (0, 188, 212): {'name': '儿科', 'time': 640},
    (207, 216, 220): {'name': '走廊', 'time': 1},
    (117, 117, 117): {'name': '楼梯', 'time': 1},
    (189, 189, 189): {'name': '电梯', 'time': 1},
    (158, 158, 158): {'name': '扶梯', 'time': 1},
    (76, 175, 80): {'name': '绿化', 'time': 1},
    (255, 235, 59): {'name': '墙', 'time': 1},
    (121, 85, 72): {'name': '门', 'time': 1},
    (156, 39, 176): {'name': '室外', 'time': 1},
    (139, 195, 74): {'name': '内镜中心', 'time': 960},
    (205, 220, 57): {'name': '检验中心', 'time': 180},
    (255, 193, 7): {'name': '消化内科', 'time': 245},
    (255, 152, 0): {'name': '甲状腺外科', 'time': 256},
    (254, 87, 34): {'name': '肾内科', 'time': 315},
    (169, 238, 90): {'name': '心血管内科', 'time': 886},
    (88, 67, 60): {'name': '采血处', 'time': 184},
    (239, 199, 78): {'name': '眼科', 'time': 564},
    (253, 186, 87): {'name': '中医科', 'time': 1486},
    (250, 133, 96): {'name': '耳鼻喉科', 'time': 737},
    (197, 254, 130): {'name': '口腔一区', 'time': 1004},
    (173, 133, 11): {'name': '超声科', 'time': 670},
    (119, 90, 10): {'name': '病理科', 'time': 1},
    (250, 146, 138): {'name': '骨科', 'time': 223},
    (255, 128, 171): {'name': '泌尿外科', 'time': 337},
    (33, 250, 230): {'name': '肝胆胰外科', 'time': 397},
    (82, 108, 255): {'name': '皮肤科', 'time': 462},
    (226, 58, 255): {'name': '妇科', 'time': 442},
    (100, 139, 55): {'name': '产科', 'time': 404},
    (188, 246, 126): {'name': '产房', 'time': 1},
    (113, 134, 91): {'name': '手术室', 'time': 1},
    (175, 207, 142): {'name': '门诊手术室', 'time': 1},
    (179, 116, 190): {'name': '中庭', 'time': 1},
    (232, 137, 248): {'name': '口腔科二区', 'time': 1004},
    (63, 100, 23): {'name': '神经内科', 'time': 428},
    (240, 222, 165): {'name': '呼吸内科', 'time': 359},
    (187, 24, 80): {'name': '综合激光科', 'time': 462},
    (150, 133, 179): {'name': '透析中心', 'time': 14400},
    (112, 40, 236): {'name': '肿瘤科', 'time': 2320},
    (241, 190, 186): {'name': '产前诊断门诊', 'time': 442},
    (186, 146, 160): {'name': '体检科', 'time': 1260},
    (71, 195, 180): {'name': '生殖医学科', 'time': 425},
    (187, 152, 247): {'name': '烧伤整形科', 'time': 256},
    (254, 210, 145): {'name': '介入科', 'time': 3240},
    (251, 242, 159): {'name': '栏杆', 'time': 1},
    (240, 61, 123): {'name': 'NICU', 'time': 1},
    (250, 162, 193): {'name': 'ICU', 'time': 1},
    (252, 201, 126): {'name': '静配中心', 'time': 1},
    (255, 255, 255): {'name': '空房间', 'time': 1}
}

class NetworkConfig:
    """Stores configuration parameters for network generation and plotting."""

    def __init__(self, color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]] = COLOR_MAP):
        self.RESULT_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'result'
        self.DEBUG_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'debug'
        self.IMAGE_ROTATE: int = 180
        self.AREA_THRESHOLD: int = 60  # Minimum area for a component to be considered a node

        # Node Type Definitions (derived from COLOR_MAP)
        self.ALL_TYPES: List[str] = [v['name'] for v in color_map_data.values()]

        self.CONNECTION_TYPES: List[str] = ['门']
        _ban_type_base: List[str] = ['墙', '栏杆', '室外', '走廊', '电梯', '扶梯', '楼梯', '空房间', '绿化', '中庭']
        self.BAN_TYPES: List[str] = [name for name in _ban_type_base if name in self.ALL_TYPES]

        self.ROOM_TYPES: List[str] = [
            v['name'] for v in color_map_data.values()
            if v['name'] not in self.BAN_TYPES and v['name'] not in self.CONNECTION_TYPES
        ]
        self.VERTICAL_TYPES: List[str] = [name for name in ['电梯', '扶梯', '楼梯'] if name in self.ALL_TYPES]
        self.PEDESTRIAN_TYPES: List[str] = [name for name in ['走廊'] if name in self.ALL_TYPES]
        self.OUTSIDE_TYPES: List[str] = [name for name in ['室外'] if name in self.ALL_TYPES]

        # Grid and Special IDs for pixel-level identification in id_map
        self.GRID_SIZE: int = 40  # Base grid size for mesh node generation
        self.OUTSIDE_ID_MAP_VALUE: int = -1  # Special ID for 'outside' areas in the id_map
        self.BACKGROUND_ID_MAP_VALUE: int = -2 # Special ID for 'background' in the id_map
        self.PEDESTRIAN_ID_MAP_VALUE: int = -3 # Special ID for 'pedestrian' areas in the id_map

        # Node Property Times
        self.OUTSIDE_MESH_TIMES_FACTOR: int = 2  # Multiplier for grid size and time for outside nodes
        self.PEDESTRIAN_TIME: float = 1.75  # Default time for pedestrian nodes
        self.CONNECTION_TIME: float = 3.0  # Default time for connection nodes (e.g., doors)

        # Plotting and Visualization
        self.IMAGE_MIRROR: bool = True  # Whether to mirror the image horizontally in plots
        self.NODE_COLOR_FROM_MAP: bool = True  # Use colors from COLOR_MAP for nodes in plots

        self.NODE_SIZE_DEFAULT: int = 10
        self.NODE_SIZE_PEDESTRIAN: int = 5
        self.NODE_SIZE_CONNECTION: int = 8
        self.NODE_SIZE_VERTICAL: int = 10
        self.NODE_SIZE_ROOM: int = 7
        self.NODE_SIZE_OUTSIDE: int = 4
        self.NODE_OPACITY: float = 0.8
        self.SHOW_PEDESTRIAN_LABELS: bool = False

        self.HORIZONTAL_EDGE_COLOR: str = "#1f77b4"
        self.VERTICAL_EDGE_COLOR: str = "#ff7f0e"
        self.EDGE_WIDTH: float = 0.5

        # SuperNetwork Specific
        self.DEFAULT_FLOOR_HEIGHT: float = 10.0
        self.DEFAULT_VERTICAL_CONNECTION_TOLERANCE: int = 0 # Default pixel distance for connecting vertical nodes across floors
        # Estimated max nodes per floor. Used for pre-allocating ID ranges in multi-processing.
        # Should be an overestimate to avoid ID collisions.
        self.ESTIMATED_MAX_NODES_PER_FLOOR: int = 10000
        self.DEFAULT_OUTSIDE_PROCESSING_IN_SUPERNETWORK: bool = False # Default for processing outside nodes per floor in SuperNetwork
        self.GROUND_FLOOR_NUMBER_FOR_OUTSIDE: Optional[int] = None # Or 0, or None to rely on auto-detection

        # Morphology Kernel
        self.MORPHOLOGY_KERNEL_SIZE: Tuple[int, int] = (5, 5)
        self.CONNECTION_DILATION_KERNEL_SIZE: Tuple[int, int] = (3,3)

        # KDTree query parameters
        self.MESH_NODE_CONNECTIVITY_K: int = 9 # k-nearest neighbors for mesh node connection

        # Ensure paths exist
        self.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        self.DEBUG_PATH.mkdir(parents=True, exist_ok=True)

class RLConfig:
    """存储用于基于强化学习的布局优化的所有配置参数。

    Attributes:
        ROOT_PATH (pathlib.Path): 项目的根目录路径。
        RL_OPTIMIZER_PATH (pathlib.Path): RL优化器模块的根目录。
        DATA_PATH (pathlib.Path): RL优化器的数据目录。
        CACHE_PATH (pathlib.Path): 用于存放所有自动生成的中间文件的缓存目录。
        LOG_PATH (pathlib.Path): 用于存放训练日志和模型的目录。
        TRAVEL_TIMES_CSV (pathlib.Path): 原始通行时间矩阵的CSV文件路径。
        PROCESS_TEMPLATES_JSON (pathlib.Path): 用户定义的就医流程模板文件路径。
        NODE_VARIANTS_JSON (pathlib.Path): 自动生成的节点变体缓存文件路径。
        TRAFFIC_DISTRIBUTION_JSON (pathlib.Path): 自动生成的流量分布缓存文件路径。
        RESOLVED_PATHWAYS_PKL (pathlib.Path): 最终解析出的流线数据缓存文件路径。
        COST_MATRIX_CACHE (pathlib.Path): 预计算成本矩阵的缓存文件路径。
        AREA_SCALING_FACTOR (float): 科室面积允许的缩放容差。
        MANDATORY_ADJACENCY (List[List[str]]): 强制相邻的科室对列表。
        PREFERRED_ADJACENCY (Dict[str, List[List[str]]]): 偏好相邻（软约束）的科室对字典。
        FIXED_NODE_TYPES (List[str]): 在布局中位置固定、不参与优化的节点类型列表。
        EMBEDDING_DIM (int): 节点嵌入向量的维度。
        TRANSFORMER_HEADS (int): Transformer编码器中的多头注意力头数。
        TRANSFORMER_LAYERS (int): Transformer编码器的层数。
        FEATURES_DIM (int): 特征提取器输出的特征维度。
        LEARNING_RATE (float): 优化器的学习率。
        NUM_ENVS (int): 用于训练的并行环境数量。
        NUM_STEPS (int): 每个环境在每次更新前收集的数据步数。
        TOTAL_TIMESTEPS (int): 训练的总时间步数。
        GAMMA (float): 奖励的折扣因子。
        GAE_LAMBDA (float): 通用优势估计(GAE)的lambda参数。
        CLIP_EPS (float): PPO中的裁剪范围。
        ENT_COEF (float): 熵损失的系数，用于鼓励探索。
        BATCH_SIZE (int): 每个优化轮次中使用的批大小。
        NUM_EPOCHS (int): 每次收集数据后，对数据进行优化的轮次。
        REWARD_TIME_WEIGHT (float): 奖励函数中通行时间成本的权重。
        REWARD_ADJACENCY_WEIGHT (float): 奖励函数中相邻性偏好的权重。
    """

    def __init__(self):
        # --- 路径配置 (使用Pathlib) ---
        self.ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent
        self.RL_OPTIMIZER_PATH: pathlib.Path = self.ROOT_PATH / 'src' / 'rl_optimizer'
        self.DATA_PATH: pathlib.Path = self.RL_OPTIMIZER_PATH / 'data'
        self.CACHE_PATH: pathlib.Path = self.DATA_PATH / 'cache'
        self.LOG_PATH: pathlib.Path = self.ROOT_PATH / 'logs'
        self.RESULT_PATH: pathlib.Path = self.ROOT_PATH / 'result'

        # --- 输入文件 ---
        self.TRAVEL_TIMES_CSV: pathlib.Path = self.ROOT_PATH / 'result' / 'super_network_travel_times.csv'
        self.PROCESS_TEMPLATES_JSON: pathlib.Path = self.DATA_PATH / 'process_templates.json'

        # --- 自动生成/缓存的中间文件 ---
        self.NODE_VARIANTS_JSON: pathlib.Path = self.CACHE_PATH / 'node_variants.json'
        self.TRAFFIC_DISTRIBUTION_JSON: pathlib.Path = self.CACHE_PATH / 'traffic_distribution.json'
        self.RESOLVED_PATHWAYS_PKL: pathlib.Path = self.CACHE_PATH / 'resolved_pathways.pkl'
        self.COST_MATRIX_CACHE: pathlib.Path = self.CACHE_PATH / 'cost_precomputation.npz'

        # --- 约束配置 ---
        self.AREA_SCALING_FACTOR: float = 0.1
        self.MANDATORY_ADJACENCY: List[List[str]] = []  # 例如: [['手术室_30007', '中心供应室_10003']]
        self.PREFERRED_ADJACENCY: Dict[str, List[List[str]]] = {
            'positive': [], # 例如: [['检验中心_10007', '采血处_20007']]
            'negative': []  # 例如: [['儿科_10006', '急诊科_1']]
        }
        self.FIXED_NODE_TYPES: List[str] = [
            '门', '楼梯', '电梯', '扶梯', '走廊', '墙', '栏杆', 
            '室外', '绿化', '中庭', '空房间'
        ]

        # --- 模型超参数 ---
        self.EMBEDDING_DIM: int = 128
        self.TRANSFORMER_HEADS: int = 4
        self.TRANSFORMER_LAYERS: int = 4
        self.FEATURES_DIM: int = 256
        self.LEARNING_RATE: float = 3e-4

        # --- PPO 训练超参数 ---
        self.NUM_ENVS: int = 8
        self.NUM_STEPS: int = 512
        self.TOTAL_TIMESTEPS: int = 5_000_000
        self.GAMMA: float = 0.99
        self.GAE_LAMBDA: float = 0.95
        self.CLIP_EPS: float = 0.2
        self.ENT_COEF: float = 0.01
        self.BATCH_SIZE: int = 64
        self.NUM_EPOCHS: int = 10

        # --- 软约束奖励权重 ---
        self.REWARD_TIME_WEIGHT: float = 1.0
        self.REWARD_ADJACENCY_WEIGHT: float = 0.1

        # 确保关键路径存在
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_PATH.mkdir(parents=True, exist_ok=True)