"""Configuration module for the network generation project."""

import pathlib
from typing import Dict, Tuple, List, Any, Optional

# 全局颜色映射表 - 医院区域类型的RGB颜色编码
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
        self.RESULT_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'results' / 'network'
        self.DEBUG_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'debug'
        self.IMAGE_ROTATE: int = 180
        self.AREA_THRESHOLD: int = 60  # 被视为节点的连通区域最小面积阈值

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

        # 网格和特殊ID用于在id_map中进行像素级别的识别
        self.GRID_SIZE: int = 40  # mesh节点生成的基础网格大小
        self.OUTSIDE_ID_MAP_VALUE: int = -1  # id_map中'室外'区域的特殊ID
        self.BACKGROUND_ID_MAP_VALUE: int = -2  # id_map中'背景'区域的特殊ID
        self.PEDESTRIAN_ID_MAP_VALUE: int = -3  # id_map中'行人通道'区域的特殊ID

        # 节点属性时间配置
        self.OUTSIDE_MESH_TIMES_FACTOR: int = 2  # 室外节点的网格大小和时间乘数因子
        self.PEDESTRIAN_TIME: float = 1.75  # 行人节点的默认通行时间
        self.CONNECTION_TIME: float = 3.0  # Default time for connection nodes (e.g., doors)

        # 绘图和可视化配置
        self.IMAGE_MIRROR: bool = True  # 是否在绘图中水平镜像图像
        self.NODE_COLOR_FROM_MAP: bool = True  # 在绘图中是否使用COLOR_MAP的颜色显示节点

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

        # SuperNetwork多层网络专用配置
        self.DEFAULT_FLOOR_HEIGHT: float = 10.0
        self.DEFAULT_VERTICAL_CONNECTION_TOLERANCE: int = 0  # 跨楼层连接垂直节点的默认像素距离容差
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
        RESUME_TRAINING (bool): 是否启用断点续训功能。
        PRETRAINED_MODEL_PATH (str): 预训练模型路径，用于断点续训。
        CHECKPOINT_FREQUENCY (int): checkpoint保存频率（按训练步数计算）。
        SAVE_TRAINING_STATE (bool): 是否保存完整训练状态（包括优化器、学习率调度器状态）。
    """

    def __init__(self):
        # --- 路径配置 (使用Pathlib) ---
        """
        初始化 RLConfig 类，设置强化学习布局优化所需的所有路径、输入文件、缓存文件、约束参数及模型训练超参数。
        
        该方法会自动创建缓存目录和日志目录（如不存在）。
        """
        self.ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent
        self.RL_OPTIMIZER_PATH: pathlib.Path = self.ROOT_PATH / 'src' / 'rl_optimizer'
        self.DATA_PATH: pathlib.Path = self.RL_OPTIMIZER_PATH / 'data'
        self.CACHE_PATH: pathlib.Path = self.DATA_PATH / 'cache'
        self.LOG_PATH: pathlib.Path = self.ROOT_PATH / 'logs'
        self.RESULT_PATH: pathlib.Path = self.ROOT_PATH / 'results'

        # --- 输入文件 ---
        self.TRAVEL_TIMES_CSV: pathlib.Path = self.RESULT_PATH / 'network' / 'hospital_travel_times.csv'
        self.PROCESS_TEMPLATES_JSON: pathlib.Path = self.DATA_PATH / 'process_templates.json'

        # --- 自动生成/缓存的中间文件 ---
        self.NODE_VARIANTS_JSON: pathlib.Path = self.CACHE_PATH / 'node_variants.json'
        self.TRAFFIC_DISTRIBUTION_JSON: pathlib.Path = self.CACHE_PATH / 'traffic_distribution.json'
        self.RESOLVED_PATHWAYS_PKL: pathlib.Path = self.CACHE_PATH / 'resolved_pathways.pkl'
        self.COST_MATRIX_CACHE: pathlib.Path = self.CACHE_PATH / 'cost_precomputation.npz'

        # --- 约束配置 ---
        self.AREA_SCALING_FACTOR: float = 0.1
        self.EMPTY_SLOT_PENALTY_FACTOR: float = 10000.0  # 每个空槽位的惩罚系数
        self.ALLOW_PARTIAL_LAYOUT: bool = True  # 是否允许部分布局（跳过槽位）
        self.MANDATORY_ADJACENCY: List[List[str]] = []  # 例如: [['手术室_30007', '中心供应室_10003']]
        self.PREFERRED_ADJACENCY: Dict[str, List[List[str]]] = {
            'positive': [], # 例如: [['检验中心_10007', '采血处_20007']]
            'negative': []  # 例如: [['儿科_10006', '急诊科_1']]
        }
        self.FIXED_NODE_TYPES: List[str] = [
            '门', '楼梯', '电梯', '扶梯', '走廊', '墙', '栏杆', 
            '室外', '绿化', '中庭', '空房间'
        ]

        # --- Transformer模型配置 ---
        self.EMBEDDING_DIM: int = 128  # 嵌入维度
        self.FEATURES_DIM: int = 128  # 特征提取器输出的特征维度
        self.TRANSFORMER_HEADS: int = 4  # 多头注意力头数
        self.TRANSFORMER_LAYERS: int = 4  # Transformer层数
        self.TRANSFORMER_DROPOUT: float = 0.1  # Dropout比例
        
        # --- 策略网络配置 ---
        self.POLICY_NET_ARCH: int = 128
        self.POLICY_NET_LAYERS: int = 2
        self.VALUE_NET_ARCH: int = 128
        self.VALUE_NET_LAYERS: int = 2
        
        # --- 学习率调度器配置 ---
        self.LEARNING_RATE_SCHEDULE_TYPE: str = "linear"  # "linear", "constant"
        self.LEARNING_RATE_INITIAL: float = 3e-4  # 初始学习率
        self.LEARNING_RATE_FINAL: float = 1e-5   # 最终学习率（线性衰减的目标值）
        self.LEARNING_RATE: float = 3e-4  # 保持向后兼容性

        # --- PPO 训练超参数 ---
        self.NUM_ENVS: int = 8
        self.N_STEPS: int = 512  # 修正为N_STEPS以匹配PPO参数
        self.TOTAL_TIMESTEPS: int = 5_000_000
        self.GAMMA: float = 0.99
        self.GAE_LAMBDA: float = 0.95
        self.CLIP_RANGE: float = 0.2  # 修正为CLIP_RANGE以匹配PPO参数
        self.ENT_COEF: float = 0.01
        self.VF_COEF: float = 0.5  # 添加值函数损失系数
        self.MAX_GRAD_NORM: float = 0.5  # 添加梯度裁剪参数
        self.BATCH_SIZE: int = 64
        self.N_EPOCHS: int = 10  # 修正为N_EPOCHS以匹配PPO参数
        
        # --- 性能优化参数 ---
        self.COST_CACHE_SIZE: int = 1000  # LRU缓存大小
        self.MAX_PATHWAY_COMBINATIONS: int = 10000  # 最大流线组合数
        
        # --- 算法默认参数 ---
        self.DEFAULT_PENALTY: float = 1000.0  # 默认惩罚值
        self.LARGE_PENALTY: float = 10000.0  # 大惩罚值
        self.INVALID_ACTION_PENALTY: float = -100.0  # 无效动作惩罚
        
        # --- 模拟退火默认参数 ---
        self.SA_MAX_REPAIR_ATTEMPTS: int = 10
        self.SA_DEFAULT_INITIAL_TEMP: float = 1000.0
        self.SA_DEFAULT_FINAL_TEMP: float = 1.0
        self.SA_DEFAULT_COOLING_RATE: float = 0.95
        
        # --- 遗传算法默认参数 ---
        self.GA_DEFAULT_POPULATION_SIZE: int = 50
        self.GA_DEFAULT_ELITE_SIZE: int = 5
        self.GA_DEFAULT_MUTATION_RATE: float = 0.1
        self.GA_DEFAULT_CROSSOVER_RATE: float = 0.8
        self.GA_DEFAULT_TOURNAMENT_SIZE: int = 3
        
        # --- 并发控制 ---
        self.MAX_PARALLEL_ALGORITHMS: int = 3
        
        # --- 评估和检查点配置 ---
        self.EVAL_FREQUENCY: int = 10000  # 添加评估频率

        # --- 软约束奖励权重 ---
        self.REWARD_TIME_WEIGHT: float = 1.0
        self.REWARD_ADJACENCY_WEIGHT: float = 0.1
        self.REWARD_PLACEMENT_BONUS: float = 1.0  # 成功放置一个科室的即时奖励
        self.REWARD_EMPTY_SLOT_PENALTY: float = 5.0  # 每个空槽位的最终惩罚
        self.REWARD_SCALE_FACTOR: float = 10000.0  # 奖励缩放因子, 仅对加权总时间成本有效
        
        # --- 势函数奖励配置 ---
        self.ENABLE_POTENTIAL_REWARD: bool = True  # 是否启用势函数奖励
        self.POTENTIAL_REWARD_WEIGHT: float = 1.0  # 势函数奖励权重
        
        # --- 面积匹配奖励配置 ---
        self.AREA_MATCH_REWARD_WEIGHT: float = 0.2  # 面积匹配在势函数中的权重
        self.AREA_MATCH_BONUS_BASE: float = 10.0  # 基础面积匹配奖励值

        # --- 断点续训配置 ---
        self.RESUME_TRAINING: bool = False  # 是否启用断点续训
        self.PRETRAINED_MODEL_PATH: str = "data/model"  # 预训练模型路径（用于断点续训）
        self.CHECKPOINT_FREQUENCY: int = 50000  # checkpoint保存频率（训练步数）
        self.SAVE_TRAINING_STATE: bool = True  # 是否保存完整训练状态（优化器、调度器等）

        # 确保关键路径存在
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_PATH.mkdir(parents=True, exist_ok=True)