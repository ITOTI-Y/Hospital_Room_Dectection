"""Configuration module for the network generation project."""

import pathlib
from typing import Dict, Tuple, List, Any

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
        all_type_names: List[str] = [v['name'] for v in color_map_data.values()]

        self.CONNECTION_TYPES: List[str] = ['门']
        _ban_type_base: List[str] = ['墙', '栏杆', '室外', '走廊', '电梯', '扶梯', '楼梯', '空房间', '绿化', '中庭']
        self.BAN_TYPES: List[str] = [name for name in _ban_type_base if name in all_type_names]

        self.ROOM_TYPES: List[str] = [
            v['name'] for v in color_map_data.values()
            if v['name'] not in self.BAN_TYPES and v['name'] not in self.CONNECTION_TYPES
        ]
        self.VERTICAL_TYPES: List[str] = [name for name in ['电梯', '扶梯', '楼梯'] if name in all_type_names]
        self.PEDESTRIAN_TYPES: List[str] = [name for name in ['走廊'] if name in all_type_names]
        self.OUTSIDE_TYPES: List[str] = [name for name in ['室外'] if name in all_type_names]

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

        # Morphology Kernel
        self.MORPHOLOGY_KERNEL_SIZE: Tuple[int, int] = (5, 5)
        self.CONNECTION_DILATION_KERNEL_SIZE: Tuple[int, int] = (3,3)

        # KDTree query parameters
        self.MESH_NODE_CONNECTIVITY_K: int = 9 # k-nearest neighbors for mesh node connection

        # Ensure paths exist
        self.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        self.DEBUG_PATH.mkdir(parents=True, exist_ok=True)