"""Configuration module for the network generation project."""

import pathlib
from typing import Dict, Tuple, List, Any, Optional

# 全局颜色映射表 - 医院区域类型的RGB颜色编码
COLOR_MAP: Dict[Tuple[int, int, int], Dict[str, Any]] = {
    (244, 67, 54): {'name': '内诊药房', 'time': 781, 'e_name': 'Pharmacy', 'code': 'T16'},
    (0, 150, 136): {'name': '挂号收费', 'time': 678, 'e_name': 'Registration & fee', 'code': 'R1-R4'},
    (103, 58, 183): {'name': '急诊科', 'time': 9000, 'e_name': 'Emergency', 'code': 'D26'},
    (145, 102, 86): {'name': '中心供应室', 'time': 1, 'e_name': 'CSSD', 'code': 'T3'},
    (33, 150, 243): {'name': '全科', 'time': 1965, 'e_name': 'General practice', 'code': 'D1'},
    (3, 169, 244): {'name': '放射科', 'time': 1484, 'e_name': 'Radiology', 'code': 'T2'},
    (0, 188, 212): {'name': '儿科', 'time': 5632, 'e_name': 'Pediatrics', 'code': 'D2'},
    (207, 216, 220): {'name': '走廊', 'time': 1, 'e_name': "Corridor", 'code': "Z1"},
    (117, 117, 117): {'name': '楼梯', 'time': 1, 'e_name': "Stairs", 'code': "Z2"},
    (189, 189, 189): {'name': '电梯', 'time': 1, 'e_name': "Elevator", 'code': "Z3"},
    (158, 158, 158): {'name': '扶梯', 'time': 1, 'e_name': "Escalator", 'code': "Z4"},
    (76, 175, 80): {'name': '绿化', 'time': 1, 'e_name': "Greening", 'code': "Z5"},
    (255, 235, 59): {'name': '墙', 'time': 1, 'e_name': "Wall", 'code': "Z6"},
    (121, 85, 72): {'name': '门', 'time': 1, 'e_name': "Door", 'code': "Z7"},
    (156, 39, 176): {'name': '室外', 'time': 1, 'e_name': "Outdoor", 'code': "Z8"},
    (139, 195, 74): {'name': '内镜中心', 'time': 5333, 'e_name': 'Endoscopy', 'code': 'T7'},
    (205, 220, 57): {'name': '检验中心', 'time': 180, 'e_name': 'Clinical Laboratory', 'code': 'T4'},
    (255, 193, 7): {'name': '消化内科', 'time': 916, 'e_name': 'Gastroenterology', 'code': 'D10'},
    (255, 152, 0): {'name': '甲状腺外科', 'time': 5637, 'e_name': 'Thyroid surgery', 'code': 'D18'},
    (254, 87, 34): {'name': '肾内科', 'time': 1493, 'e_name': 'Nephrology', 'code': 'D6'},
    (169, 238, 90): {'name': '心血管内科', 'time': 8089, 'e_name': 'Cardiovascular Medicine', 'code': 'D8'},
    (88, 67, 60): {'name': '采血处', 'time': 1104, 'e_name': 'Blood collection', 'code': 'T15'},
    (239, 199, 78): {'name': '眼科', 'time': 3228, 'e_name': 'Ophthalmology', 'code': 'D22'},
    (253, 186, 87): {'name': '中医科', 'time': 3110, 'e_name': 'Chinese medicine', 'code': 'D3'},
    (250, 133, 96): {'name': '耳鼻喉科', 'time': 9550, 'e_name': 'Otolaryngology', 'code': 'D23'},
    (197, 254, 130): {'name': '口腔一区', 'time': 10542, 'e_name': 'Dental clinic 1', 'code': 'D24'},
    (173, 133, 11): {'name': '超声科', 'time': 3023, 'e_name': 'Ultrasound', 'code': 'T1'},
    (119, 90, 10): {'name': '病理科', 'time': 1, 'e_name': 'Pathology', 'code': 'T5'},
    (250, 146, 138): {'name': '骨科', 'time': 1090, 'e_name': 'Orthopedics', 'code': 'D7'},
    (255, 128, 171): {'name': '泌尿外科', 'time': 1019, 'e_name': 'Urology', 'code': 'D12'},
    (33, 250, 230): {'name': '肝胆胰外科', 'time': 2953, 'e_name': 'Hepatobiliary & Pancreatic', 'code': 'D19'},
    (82, 108, 255): {'name': '皮肤科', 'time': 2802, 'e_name': 'Dermatology', 'code': 'D21'},
    (226, 58, 255): {'name': '妇科', 'time': 5000, 'e_name': 'Gynaecology', 'code': 'D15'},
    (100, 139, 55): {'name': '产科', 'time': 2208, 'e_name': 'Obstetrics', 'code': 'D14'},
    (170, 190, 150): {'name': '产房', 'time': 1, 'e_name': 'Delivery room', 'code': 'T9'},
    (113, 134, 91): {'name': '手术室', 'time': 1, 'e_name': 'Theater', 'code': 'T13'},
    (175, 207, 142): {'name': '门诊手术室', 'time': 1, 'e_name': 'Ambulatory Surgery', 'code': 'T12'},
    (179, 116, 190): {'name': '中庭', 'time': 1, 'e_name': "Courtyard", 'code': "Z9"},
    (232, 137, 248): {'name': '口腔科二区', 'time': 4964, 'e_name': 'Dental clinic 2', 'code': 'D25'},
    (63, 100, 23): {'name': '神经内科', 'time': 2396, 'e_name': 'Neurology', 'code': 'D9'},
    (240, 222, 165): {'name': '呼吸内科', 'time': 1457, 'e_name': 'Respiratory', 'code': 'D5'},
    (187, 24, 80): {'name': '综合激光科', 'time': 1848, 'e_name': 'Laser clinic', 'code': 'D20'},
    (150, 133, 179): {'name': '透析中心', 'time': 44267, 'e_name': 'Haemodialysis unit', 'code': 'T8'},
    (112, 40, 236): {'name': '肿瘤科', 'time': 11729, 'e_name': 'Oncology', 'code': 'D11'},
    (241, 190, 186): {'name': '产前诊断门诊', 'time': 1252, 'e_name': 'Prenatal Diagnosis', 'code': 'D13'},
    (186, 146, 160): {'name': '体检科', 'time': 7336, 'e_name': 'Physical Examination', 'code': 'D4'},
    (71, 195, 180): {'name': '生殖医学科', 'time': 1086, 'e_name': 'Reproductive Medicine', 'code': 'D16'},
    (187, 152, 247): {'name': '烧伤整形科', 'time': 854, 'e_name': 'Plastic surgery', 'code': 'D17'},
    (254, 210, 145): {'name': '介入科', 'time': 14688, 'e_name': 'Interventional Therapy', 'code': 'T6'},
    (251, 242, 159): {'name': '栏杆', 'time': 1, 'e_name': "Handrail", 'code': "Z10"},
    (240, 61, 123): {'name': 'NICU', 'time': 1, 'e_name': 'NICU', 'code': 'T10'},
    (250, 162, 193): {'name': 'ICU', 'time': 1, 'e_name': 'ICU', 'code': 'T11'},
    (252, 201, 126): {'name': '静配中心', 'time': 1, 'e_name': 'PIVAS', 'code': 'T14'},
    (255, 255, 255): {'name': '空房间', 'time': 1, 'e_name': "Vacant Room", 'code': "Z11"}
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

        self.NODE_SIZE_DEFAULT: int = 2
        self.NODE_SIZE_PEDESTRIAN: int = 2
        self.NODE_SIZE_CONNECTION: int = 2
        self.NODE_SIZE_VERTICAL: int = 2
        self.NODE_SIZE_ROOM: int = 7
        self.NODE_SIZE_OUTSIDE: int = 2
        self.NODE_OPACITY: float = 0.8
        self.SHOW_PEDESTRIAN_LABELS: bool = False

        self.HORIZONTAL_EDGE_COLOR: str = "#1f77b4"
        self.VERTICAL_EDGE_COLOR: str = "rgba(151,152,155,0.7)"  #"#ff7e0e75"
        self.DOOR_EDGE_COLOR: str = "#2ca02c"
        self.SPECIAL_EDGE_COLOR: str = "#d62728"
        self.EDGE_WIDTH: float = 1.5

        self.X_AXIS_RATIO: float = 1
        self.Y_AXIS_RATIO: float = 1
        self.Z_AXIS_RATIO: float = 2

        #Plotly Configuration
        self.PLOTLY_CONFIG = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'Network',
                'scale': 3
            }
        }

        # SuperNetwork多层网络专用配置
        self.DEFAULT_FLOOR_HEIGHT: float = 10
        self.DEFAULT_VERTICAL_CONNECTION_TOLERANCE: int = 0  # 跨楼层连接垂直节点的默认像素距离容差
        # Estimated max nodes per floor. Used for pre-allocating ID ranges in multi-processing.
        # Should be an overestimate to avoid ID collisions.
        self.ESTIMATED_MAX_NODES_PER_FLOOR: int = 10000
        self.DEFAULT_OUTSIDE_PROCESSING_IN_SUPERNETWORK: bool = False # Default for processing outside nodes per floor in SuperNetwork
        self.GROUND_FLOOR_NUMBER_FOR_OUTSIDE: Optional[int] = None # Or 0, or None to rely on auto-detection

        # 垂直连接相关配置
        self.Z_LEVEL_DIFF_THRESHOLD: float = 1.0  # Z层级差异阈值，判断是否为不同楼层
        self.MIN_VERTICAL_TOLERANCE: int = 10  # 最小垂直连接容差值
        self.VERTICAL_TOLERANCE_FACTOR: float = 0.5  # 垂直容差计算因子（基于平均最小距离）

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
        self.PROCESS_TEMPLATES_JSON: pathlib.Path = self.DATA_PATH / 'process_templates_traditional.json'

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
        self.FIXED_NODE_TYPES: List[str] = [
            '门', '楼梯', '电梯', '扶梯', '走廊', '墙', '栏杆', 
            '室外', '绿化', '中庭', '空房间', '急诊科','挂号收费'
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
        self.LEARNING_RATE_INITIAL: float = 1e-4  # 初始学习率
        self.LEARNING_RATE_FINAL: float = 1e-8   # 最终学习率（线性衰减的目标值）
        self.LEARNING_RATE: float = 1e-4  # 保持向后兼容性

        # --- PPO 训练超参数 ---
        self.NUM_ENVS: int = 8
        self.N_STEPS: int = 512  # 修正为N_STEPS以匹配PPO参数
        self.TOTAL_TIMESTEPS: int = 5_000_000
        self.GAMMA: float = 0.99
        self.GAE_LAMBDA: float = 0.95
        self.CLIP_RANGE: float = 0.2  # 修正为CLIP_RANGE以匹配PPO参数
        self.ENT_COEF: float = 0.05
        self.VF_COEF: float = 0.5  # 添加值函数损失系数
        self.MAX_GRAD_NORM: float = 0.5  # 添加梯度裁剪参数
        self.BATCH_SIZE: int = 64
        self.N_EPOCHS: int = 10  # 修正为N_EPOCHS以匹配PPO参数
        
        # --- 性能优化参数 ---
        self.COST_CACHE_SIZE: int = 1000  # LRU缓存大小
        self.MAX_PATHWAY_COMBINATIONS: int = 10000  # 最大流线组合数
        
        # --- 算法默认参数 ---
        self.DEFAULT_PENALTY: float = max([i['time'] for i in COLOR_MAP.values()])  # 默认惩罚值使用最大通行时间
        self.INVALID_ACTION_PENALTY: float = -100.0  # 无效动作惩罚
        
        # --- 模拟退火默认参数 ---
        self.SA_DEFAULT_INITIAL_TEMP: float = 1000.0 # 初始温度
        self.SA_DEFAULT_FINAL_TEMP: float = 0.1 # 最终温度
        self.SA_DEFAULT_COOLING_RATE: float = 0.95 # 冷却速率
        self.SA_DEFAULT_TEMPERATURE_LENGTH: int = 100 # 温度长度
        self.SA_DEFAULT_MAX_ITERATIONS: int = 10000 # 最大迭代次数
        self.SA_MAX_REPAIR_ATTEMPTS: int = 10 # 约束修复最大尝试次数

        # --- 遗传算法默认参数 ---
        self.GA_DEFAULT_POPULATION_SIZE: int = 300 # 种群大小
        self.GA_DEFAULT_ELITE_SIZE: int = 20 # 精英大小
        self.GA_DEFAULT_MUTATION_RATE: float = 0.15  # 初始变异率
        self.GA_DEFAULT_CROSSOVER_RATE: float = 0.85  # 初始交叉率
        self.GA_DEFAULT_TOURNAMENT_SIZE: int = 5 # 锦标赛大小
        self.GA_DEFAULT_MAX_AGE: int = 100 # 最大代数
        self.GA_DEFAULT_CONVERGENCE_THRESHOLD: int = 300 # 停滞代数阈值
        self.GA_DEFAULT_MAX_ITERATIONS: int = 10000 # 最大迭代次数

        # --- 约束感知遗传算法增强参数 ---
        self.GA_CONSTRAINT_REPAIR_STRATEGY: str = 'greedy_area_matching'  # 默认约束修复策略
        self.GA_ADAPTIVE_PARAMETERS: bool = True  # 启用自适应参数调整
        self.GA_MAX_REPAIR_ATTEMPTS: int = 5  # 约束修复最大尝试次数
        self.GA_DIVERSITY_THRESHOLD_LOW: float = 0.05  # 多样性极低阈值
        self.GA_DIVERSITY_THRESHOLD_MEDIUM: float = 0.15  # 多样性较低阈值
        
        # --- 并发控制 ---
        self.MAX_PARALLEL_ALGORITHMS: int = 3
        
        # --- 评估和检查点配置 ---
        self.EVAL_FREQUENCY: int = 10000  # 添加评估频率

        # --- 软约束奖励权重 ---
        self.REWARD_TIME_WEIGHT: float = 1.0
        self.REWARD_ADJACENCY_WEIGHT: float = 0.1
        self.REWARD_PLACEMENT_BONUS: float = 0.1  # 成功放置一个科室的即时奖励
        self.REWARD_EMPTY_SLOT_PENALTY: float = 0.5  # 每个空槽位的最终惩罚
        self.REWARD_SCALE_FACTOR: float = 40000.0  # 奖励缩放因子, 仅对加权总时间成本有效
        self.REWARD_SKIP_PENALTY: float = -0.2  # 跳过惩罚
        self.REWARD_COMPLETION_BONUS: float = 1.0  # 完成奖励

        # --- 势函数奖励配置 ---
        self.ENABLE_POTENTIAL_REWARD: bool = False  # 是否启用势函数奖励
        self.POTENTIAL_REWARD_WEIGHT: float = 1.0  # 势函数奖励权重
        
        # --- 面积匹配奖励配置 ---
        self.AREA_MATCH_REWARD_WEIGHT: float = 0.2  # 面积匹配在势函数中的权重
        self.AREA_MATCH_BONUS_BASE: float = 10.0  # 基础面积匹配奖励值
        
        # --- 相邻性奖励配置 ---
        # 相邻性奖励总开关
        self.ENABLE_ADJACENCY_REWARD: bool = True
        
        # 相邻性奖励优化开关
        self.ENABLE_ADJACENCY_OPTIMIZATION: bool = True  # 启用优化相邻性计算器
        
        # 相邻性奖励权重(在势函数中的权重)
        self.ADJACENCY_REWARD_WEIGHT: float = 0.15
        
        # 多维度相邻性权重分配
        self.SPATIAL_ADJACENCY_WEIGHT: float = 0.4      # 空间相邻性权重
        self.FUNCTIONAL_ADJACENCY_WEIGHT: float = 0.5   # 功能相邻性权重  
        self.CONNECTIVITY_ADJACENCY_WEIGHT: float = 0.1 # 连通性相邻性权重
        
        # 相邻性判定参数
        self.ADJACENCY_PERCENTILE_THRESHOLD: float = 0.1  # 相邻性分位数阈值
        self.ADJACENCY_K_NEAREST: int | None = None              # 最近邻数量(None=自动)
        self.ADJACENCY_CLUSTER_EPS_PERCENTILE: float = 0.1 # 聚类邻域分位数
        self.ADJACENCY_MIN_CLUSTER_SIZE: int = 2          # 最小聚类大小
        
        # 连通性相邻性参数
        self.CONNECTIVITY_DISTANCE_PERCENTILE: float = 0.3  # 连通距离分位数阈值
        self.CONNECTIVITY_MAX_PATH_LENGTH: int = 3          # 最大路径长度
        self.CONNECTIVITY_WEIGHT_DECAY: float = 0.8         # 路径长度权重衰减因子
        
        # 奖励缩放参数
        self.ADJACENCY_REWARD_BASE: float = 5.0           # 基础奖励值
        self.ADJACENCY_PENALTY_MULTIPLIER: float = 1.5    # 负向惩罚倍数
        
        # 缓存和性能参数
        self.ADJACENCY_CACHE_SIZE: int = 500              # 相邻性缓存大小
        self.ADJACENCY_PRECOMPUTE: bool = True            # 是否预计算相邻性矩阵
        
        # 优化相邻性计算器配置
        self.ADJACENCY_OPTIMIZATION_SPARSE_THRESHOLD: float = 0.1  # 稀疏矩阵阈值
        self.ADJACENCY_OPTIMIZATION_VECTORIZE_BATCH_SIZE: int = 1000  # 向量化批处理大小
        self.ADJACENCY_OPTIMIZATION_MEMORY_EFFICIENT: bool = True  # 启用内存优化
        
        # 医疗功能相邻性数据
        self.MEDICAL_ADJACENCY_PREFERENCES: Dict[str, Dict[str, float]] = {
            # 正向偏好 (值为正数)
            "静配中心": {
                "ICU": 0.8,      # 静配中心与ICU物流相关
            },
            "中心供应室": {
                "内镜中心": 0.8,        # 中心供应室与内镜中心物流相关
            },
            "采血处": {
                "检验中心": 0.8,    # 采血处与检验中心物流相关
            },
            "手术室": {
                "ICU": 0.8,          # 手术室与ICU强相关
                "门诊手术室": 0.6
            },
            "ICU": {
                "NICU": 0.8          # ICU与NICU强相关
            },
            "产房": {
                "NICU": 0.8          # 产房与NICU强相关
            },
            "检验中心": {
                "病理科": 0.8,    # 检验中心与病理科物流相关
            }
        }

        # --- 断点续训配置 ---
        self.RESUME_TRAINING: bool = False  # 是否启用断点续训
        self.PRETRAINED_MODEL_PATH: str = "data/model"  # 预训练模型路径（用于断点续训）
        self.CHECKPOINT_FREQUENCY: int = 50000  # checkpoint保存频率（训练步数）
        self.SAVE_TRAINING_STATE: bool = True  # 是否保存完整训练状态（优化器、调度器等）

        # 确保关键路径存在
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_PATH.mkdir(parents=True, exist_ok=True)
        
        # --- 动态基线奖励归一化配置 ---
        self.ENABLE_DYNAMIC_BASELINE: bool = True  # 启用动态基线奖励归一化
        self.EMA_ALPHA: float = 0.1  # 指数移动平均平滑因子 (0 < alpha <= 1)
        self.BASELINE_WARMUP_EPISODES: int = 10  # 基线预热期episode数量
        self.BASELINE_UPDATE_FREQUENCY: int = 10  # 基线更新频率（每N个episode更新一次）
        
        # 动态基线归一化权重（归一化后各组件的权重）
        self.NORMALIZED_TIME_WEIGHT: float = 1.0  # 时间成本归一化权重
        self.NORMALIZED_ADJACENCY_WEIGHT: float = 0.5  # 相邻性奖励归一化权重
        self.NORMALIZED_AREA_WEIGHT: float = 0.3  # 面积匹配归一化权重
        self.NORMALIZED_SKIP_PENALTY_WEIGHT: float = 0.3  # 跳过惩罚归一化权重
        self.NORMALIZED_PLACEMENT_BONUS_WEIGHT: float = 0.1  # 放置奖励归一化权重
        self.NORMALIZED_COMPLETION_BONUS_WEIGHT: float = 0.1  # 完成奖励归一化权重
        
        # 奖励归一化参数
        self.REWARD_NORMALIZATION_CLIP_RANGE: float = 1.0  # 归一化时的裁剪范围（几个标准差）
        self.REWARD_NORMALIZATION_MIN_STD: float = 1e-8  # 最小标准差，防止除零错误
        self.ENABLE_REWARD_CLIPPING: bool = True  # 启用奖励裁剪
        self.REWARD_CLIP_RANGE: Tuple[float, float] = (-10.0, 10.0)  # 最终奖励裁剪范围
        
        # 基线预测相关配置
        self.ENABLE_RELATIVE_IMPROVEMENT_REWARD: bool = True  # 启用相对改进奖励
        self.RELATIVE_IMPROVEMENT_SCALE: float = 5.0  # 相对改进奖励缩放因子
        self.BASELINE_SMOOTHING_WINDOW: int = 50  # 基线平滑窗口大小
        
        # 验证相邻性奖励配置参数
        if self.ENABLE_ADJACENCY_REWARD:
            self._validate_adjacency_parameters()
        
        # 验证动态基线配置参数
        if self.ENABLE_DYNAMIC_BASELINE:
            self._validate_dynamic_baseline_parameters()
    
    def _validate_dynamic_baseline_parameters(self):
        """
        验证动态基线配置参数的有效性。
        在配置错误时抛出异常或自动修正参数值。
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 验证EMA平滑因子
        if not (0 < self.EMA_ALPHA <= 1):
            raise ValueError(f"EMA平滑因子必须在(0,1]范围内，当前值：{self.EMA_ALPHA}")
        
        # 验证预热期参数
        if not isinstance(self.BASELINE_WARMUP_EPISODES, int) or self.BASELINE_WARMUP_EPISODES < 0:
            logger.warning(f"基线预热期必须为非负整数，当前值：{self.BASELINE_WARMUP_EPISODES}，自动修正为200")
            self.BASELINE_WARMUP_EPISODES = 200
        
        # 验证更新频率
        if not isinstance(self.BASELINE_UPDATE_FREQUENCY, int) or self.BASELINE_UPDATE_FREQUENCY <= 0:
            logger.warning(f"基线更新频率必须为正整数，当前值：{self.BASELINE_UPDATE_FREQUENCY}，自动修正为10")
            self.BASELINE_UPDATE_FREQUENCY = 10
        
        # 验证归一化权重
        weight_params = [
            ('NORMALIZED_TIME_WEIGHT', self.NORMALIZED_TIME_WEIGHT),
            ('NORMALIZED_ADJACENCY_WEIGHT', self.NORMALIZED_ADJACENCY_WEIGHT),
            ('NORMALIZED_AREA_WEIGHT', self.NORMALIZED_AREA_WEIGHT),
            ('NORMALIZED_SKIP_PENALTY_WEIGHT', self.NORMALIZED_SKIP_PENALTY_WEIGHT),
            ('NORMALIZED_COMPLETION_BONUS_WEIGHT', self.NORMALIZED_COMPLETION_BONUS_WEIGHT)
        ]
        
        for param_name, param_value in weight_params:
            if not isinstance(param_value, (int, float)) or param_value < 0:
                raise ValueError(f"归一化权重参数 {param_name} 必须为非负数值，当前值：{param_value}")
        
        # 验证裁剪范围
        if not isinstance(self.REWARD_NORMALIZATION_CLIP_RANGE, (int, float)) or self.REWARD_NORMALIZATION_CLIP_RANGE <= 0:
            raise ValueError(f"奖励归一化裁剪范围必须为正数，当前值：{self.REWARD_NORMALIZATION_CLIP_RANGE}")
        
        # 验证最小标准差
        if not isinstance(self.REWARD_NORMALIZATION_MIN_STD, (int, float)) or self.REWARD_NORMALIZATION_MIN_STD <= 0:
            raise ValueError(f"最小标准差必须为正数，当前值：{self.REWARD_NORMALIZATION_MIN_STD}")
        
        # 验证奖励裁剪范围
        if isinstance(self.REWARD_CLIP_RANGE, (list, tuple)) and len(self.REWARD_CLIP_RANGE) == 2:
            min_reward, max_reward = self.REWARD_CLIP_RANGE
            if min_reward >= max_reward:
                raise ValueError(f"奖励裁剪范围必须满足min < max，当前值：{self.REWARD_CLIP_RANGE}")
        else:
            raise ValueError(f"奖励裁剪范围必须为包含两个元素的tuple，当前值：{self.REWARD_CLIP_RANGE}")
        
        # 验证相对改进奖励配置
        if not isinstance(self.RELATIVE_IMPROVEMENT_SCALE, (int, float)) or self.RELATIVE_IMPROVEMENT_SCALE <= 0:
            logger.warning(f"相对改进奖励缩放因子必须为正数，当前值：{self.RELATIVE_IMPROVEMENT_SCALE}，自动修正为5.0")
            self.RELATIVE_IMPROVEMENT_SCALE = 5.0
        
        # 验证基线平滑窗口
        if not isinstance(self.BASELINE_SMOOTHING_WINDOW, int) or self.BASELINE_SMOOTHING_WINDOW <= 0:
            logger.warning(f"基线平滑窗口必须为正整数，当前值：{self.BASELINE_SMOOTHING_WINDOW}，自动修正为50")
            self.BASELINE_SMOOTHING_WINDOW = 50
        
        logger.info("动态基线配置参数验证通过")
    
    def _validate_adjacency_parameters(self):
        """
        验证相邻性奖励相关配置参数的有效性。
        在配置错误时抛出异常或自动修正参数值。
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 验证权重参数
        weight_params = [
            ('ADJACENCY_REWARD_WEIGHT', self.ADJACENCY_REWARD_WEIGHT),
            ('SPATIAL_ADJACENCY_WEIGHT', self.SPATIAL_ADJACENCY_WEIGHT), 
            ('FUNCTIONAL_ADJACENCY_WEIGHT', self.FUNCTIONAL_ADJACENCY_WEIGHT),
            ('CONNECTIVITY_ADJACENCY_WEIGHT', self.CONNECTIVITY_ADJACENCY_WEIGHT)
        ]
        
        for param_name, param_value in weight_params:
            if not isinstance(param_value, (int, float)) or param_value < 0:
                raise ValueError(f"相邻性权重参数 {param_name} 必须为非负数值，当前值：{param_value}")
                
        # 验证分位数阈值参数
        if not (0 < self.ADJACENCY_PERCENTILE_THRESHOLD < 1):
            raise ValueError(f"相邻性分位数阈值必须在(0,1)范围内，当前值：{self.ADJACENCY_PERCENTILE_THRESHOLD}")
        
        # 验证连通性参数
        if self.CONNECTIVITY_ADJACENCY_WEIGHT > 0:
            if not isinstance(self.CONNECTIVITY_MAX_PATH_LENGTH, int) or not (2 <= self.CONNECTIVITY_MAX_PATH_LENGTH <= 5):
                logger.warning(f"连通性最大路径长度应在2-5之间，当前值：{self.CONNECTIVITY_MAX_PATH_LENGTH}，自动修正为3")
                self.CONNECTIVITY_MAX_PATH_LENGTH = 3
                
            if not (0 < self.CONNECTIVITY_WEIGHT_DECAY < 1):
                logger.warning(f"连通性权重衰减因子应在(0,1)范围内，当前值：{self.CONNECTIVITY_WEIGHT_DECAY}，自动修正为0.8")
                self.CONNECTIVITY_WEIGHT_DECAY = 0.8
                
            if not (0 < self.CONNECTIVITY_DISTANCE_PERCENTILE < 1):
                logger.warning(f"连通性距离分位数应在(0,1)范围内，当前值：{self.CONNECTIVITY_DISTANCE_PERCENTILE}，自动修正为0.3")
                self.CONNECTIVITY_DISTANCE_PERCENTILE = 0.3
        
        # 验证奖励缩放参数
        if not isinstance(self.ADJACENCY_REWARD_BASE, (int, float)) or self.ADJACENCY_REWARD_BASE <= 0:
            raise ValueError(f"相邻性奖励基础值必须为正数，当前值：{self.ADJACENCY_REWARD_BASE}")
            
        if not isinstance(self.ADJACENCY_PENALTY_MULTIPLIER, (int, float)) or self.ADJACENCY_PENALTY_MULTIPLIER <= 0:
            raise ValueError(f"相邻性惩罚倍数必须为正数，当前值：{self.ADJACENCY_PENALTY_MULTIPLIER}")
        
        # 验证优化配置参数
        if not isinstance(self.ENABLE_ADJACENCY_OPTIMIZATION, bool):
            logger.warning(f"相邻性优化开关应为布尔值，当前值：{self.ENABLE_ADJACENCY_OPTIMIZATION}，自动修正为True")
            self.ENABLE_ADJACENCY_OPTIMIZATION = True
            
        if not (0 < self.ADJACENCY_OPTIMIZATION_SPARSE_THRESHOLD <= 1):
            logger.warning(f"稀疏矩阵阈值应在(0,1]范围内，当前值：{self.ADJACENCY_OPTIMIZATION_SPARSE_THRESHOLD}，自动修正为0.1")
            self.ADJACENCY_OPTIMIZATION_SPARSE_THRESHOLD = 0.1
            
        if not isinstance(self.ADJACENCY_OPTIMIZATION_VECTORIZE_BATCH_SIZE, int) or self.ADJACENCY_OPTIMIZATION_VECTORIZE_BATCH_SIZE <= 0:
            logger.warning(f"向量化批处理大小应为正整数，当前值：{self.ADJACENCY_OPTIMIZATION_VECTORIZE_BATCH_SIZE}，自动修正为1000")
            self.ADJACENCY_OPTIMIZATION_VECTORIZE_BATCH_SIZE = 1000
        
        # 验证缓存参数
        if not isinstance(self.ADJACENCY_CACHE_SIZE, int) or self.ADJACENCY_CACHE_SIZE <= 0:
            logger.warning(f"相邻性缓存大小应为正整数，当前值：{self.ADJACENCY_CACHE_SIZE}，自动修正为500")
            self.ADJACENCY_CACHE_SIZE = 500
        
        # 验证医疗功能相邻性偏好数据
        if not isinstance(self.MEDICAL_ADJACENCY_PREFERENCES, dict):
            raise ValueError("医疗相邻性偏好配置必须为字典类型")
            
        # 检查偏好数据的格式正确性
        for dept, preferences in self.MEDICAL_ADJACENCY_PREFERENCES.items():
            if not isinstance(dept, str) or not isinstance(preferences, dict):
                raise ValueError(f"医疗相邻性偏好格式错误：{dept} -> {preferences}")
                
            for target_dept, score in preferences.items():
                if not isinstance(target_dept, str) or not isinstance(score, (int, float)):
                    raise ValueError(f"医疗相邻性偏好分数格式错误：{dept} -> {target_dept}: {score}")
                
                if not (-2.0 <= score <= 2.0):
                    logger.warning(f"医疗相邻性偏好分数建议在[-2.0, 2.0]范围内：{dept} -> {target_dept}: {score}")
        
        # 权重组合合理性检查
        total_adjacency_weight = (self.SPATIAL_ADJACENCY_WEIGHT + 
                                 self.FUNCTIONAL_ADJACENCY_WEIGHT + 
                                 self.CONNECTIVITY_ADJACENCY_WEIGHT)
        
        if total_adjacency_weight <= 0:
            raise ValueError("至少需要启用一种相邻性权重（空间、功能、连通性）")
            
        if total_adjacency_weight > 3.0:
            logger.warning(f"相邻性权重总和过高可能影响训练稳定性：{total_adjacency_weight:.2f}")
        
        logger.info("相邻性奖励配置参数验证通过")