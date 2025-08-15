# 完善_calculate_adjacency_reward方法 - 代码架构

## 架构总览

本文档描述了动态相邻性奖励机制的代码架构设计，包括模块组织、数据流设计、编码规范和扩展性架构，为开发团队提供清晰的代码结构指导。

## 整体架构设计

### 分层架构模式

```
展示层 (Presentation Layer)
├── LayoutEnv._calculate_adjacency_reward()    # 对外接口
└── 配置参数 (RLConfig.ADJACENCY_*)

业务层 (Business Layer)
├── AdjacencyAnalyzer                          # 业务协调器
├── RewardIntegrator                           # 奖励集成业务逻辑
└── 医疗偏好规则 (Medical Preference Rules)

计算层 (Computation Layer)
├── SpatialAdjacencyCalculator                 # 空间相邻计算
├── FunctionalAdjacencyCalculator              # 功能相邻计算
├── TravelTimeAdjacencyCalculator              # 时间相邻计算
└── 算法工具类 (Algorithm Utilities)

数据层 (Data Layer)
├── AdjacencyCache                             # 缓存管理
├── 通行时间数据 (Travel Times Matrix)
├── 医疗流程数据 (Medical Pathways)
└── 配置数据 (Configuration Data)
```

## 模块组织结构

### 目录结构设计

```
src/algorithms/adjacency/
├── __init__.py                          # 包导出接口
├── core/                                # 核心模块
│   ├── __init__.py
│   ├── adjacency_analyzer.py           # 主分析器
│   ├── reward_integrator.py            # 奖励集成器
│   └── base_calculator.py              # 计算器基类
├── calculators/                         # 计算器实现
│   ├── __init__.py
│   ├── spatial_calculator.py           # 空间相邻计算器
│   ├── functional_calculator.py        # 功能相邻计算器
│   └── travel_time_calculator.py       # 通行时间计算器
├── cache/                               # 缓存模块
│   ├── __init__.py
│   ├── adjacency_cache.py              # 相邻性缓存
│   └── cache_strategies.py             # 缓存策略
├── utils/                               # 工具模块
│   ├── __init__.py
│   ├── matrix_ops.py                   # 矩阵操作工具
│   ├── graph_utils.py                  # 图算法工具
│   └── validation.py                   # 数据验证工具
├── config/                              # 配置模块
│   ├── __init__.py
│   ├── adjacency_config.py             # 相邻性专用配置
│   └── medical_preferences.py          # 医疗偏好配置
└── tests/                               # 测试模块
    ├── __init__.py
    ├── test_adjacency_analyzer.py
    ├── test_calculators.py
    └── test_integration.py
```

### 包导出接口设计

```python
# src/algorithms/adjacency/__init__.py

"""
医院布局相邻性奖励计算模块

主要功能：
1. 动态相邻性判定算法
2. 多维度相邻性评分
3. 医疗功能导向的奖励计算
4. 高性能缓存机制

使用示例：
    from src.algorithms.adjacency import AdjacencyAnalyzer
    
    analyzer = AdjacencyAnalyzer(config, travel_times, slots, depts, pathways)
    reward = analyzer.calculate_adjacency_reward(layout)
"""

from .core.adjacency_analyzer import AdjacencyAnalyzer
from .core.reward_integrator import RewardIntegrator
from .calculators.spatial_calculator import SpatialAdjacencyCalculator
from .calculators.functional_calculator import FunctionalAdjacencyCalculator
from .calculators.travel_time_calculator import TravelTimeAdjacencyCalculator
from .cache.adjacency_cache import AdjacencyCache
from .config.adjacency_config import AdjacencyConfig

# 版本信息
__version__ = "1.0.0"
__author__ = "Hospital Layout Optimization Team"

# 导出的主要类
__all__ = [
    'AdjacencyAnalyzer',
    'RewardIntegrator',
    'SpatialAdjacencyCalculator',
    'FunctionalAdjacencyCalculator',
    'TravelTimeAdjacencyCalculator',
    'AdjacencyCache',
    'AdjacencyConfig'
]

# 模块级配置
DEFAULT_CONFIG = {
    'ENABLE_ADJACENCY_REWARD': True,
    'ADJACENCY_REWARD_WEIGHT': 0.15,
    'ADJACENCY_CACHE_SIZE': 500,
    'ADJACENCY_PRECOMPUTE': True
}
```

## 数据流和控制流设计

### 数据流架构

```mermaid
graph TD
    A[LayoutEnv.step()] --> B[_calculate_potential()]
    B --> C[AdjacencyAnalyzer.calculate_adjacency_reward()]
    C --> D{缓存检查}
    D -->|命中| E[返回缓存结果]
    D -->|未命中| F[计算相邻性得分]
    F --> G[SpatialCalculator]
    F --> H[FunctionalCalculator]
    F --> I[TravelTimeCalculator]
    G --> J[RewardIntegrator]
    H --> J
    I --> J
    J --> K[缓存结果]
    K --> L[返回奖励值]
    E --> M[更新势函数]
    L --> M
```

### 控制流设计

```python
# 主控制流伪代码
def calculate_adjacency_reward(layout):
    """主控制流程"""
    
    # 1. 输入验证
    if not validate_layout(layout):
        return 0.0
    
    # 2. 缓存检查
    cached_result = cache.get_reward(layout)
    if cached_result is not None:
        return cached_result
    
    # 3. 并行计算各维度得分
    spatial_score = spatial_calculator.calculate_score(layout)
    functional_score = functional_calculator.calculate_score(layout)
    connectivity_score = travel_time_calculator.calculate_score(layout)
    
    # 4. 集成奖励
    integrated_reward = reward_integrator.integrate_scores(
        spatial_score, functional_score, connectivity_score)
    
    # 5. 缓存结果
    cache.put_reward(layout, integrated_reward)
    
    # 6. 记录统计
    log_adjacency_statistics(layout, integrated_reward)
    
    return integrated_reward
```

## 接口定义和抽象基类

### 计算器抽象基类

```python
# src/algorithms/adjacency/core/base_calculator.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class BaseAdjacencyCalculator(ABC):
    """
    相邻性计算器抽象基类
    
    定义所有相邻性计算器必须实现的接口
    """
    
    def __init__(self, config, *args, **kwargs):
        """
        初始化计算器
        
        Args:
            config: 配置对象
            *args, **kwargs: 计算器特定参数
        """
        self.config = config
        self._initialize(*args, **kwargs)
    
    @abstractmethod
    def _initialize(self, *args, **kwargs):
        """
        计算器特定的初始化逻辑
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def calculate_adjacency_matrix(self) -> np.ndarray:
        """
        计算相邻性关系矩阵
        
        Returns:
            相邻性矩阵，形状根据计算器类型确定
        """
        pass
    
    @abstractmethod
    def get_adjacent_items(self, item_index: int) -> List[int]:
        """
        获取指定项目的相邻项目列表
        
        Args:
            item_index: 项目索引（槽位或科室）
            
        Returns:
            相邻项目索引列表
        """
        pass
    
    @abstractmethod
    def calculate_adjacency_score(self, layout: List[str]) -> float:
        """
        计算布局的相邻性得分
        
        Args:
            layout: 当前布局
            
        Returns:
            相邻性得分
        """
        pass
    
    def calculate_score_from_matrix(self, layout: List[str], 
                                  adjacency_matrix: np.ndarray) -> float:
        """
        基于预计算矩阵计算得分
        
        默认实现，子类可重写以提高效率
        """
        return self.calculate_adjacency_score(layout)
    
    def validate_layout(self, layout: List[str]) -> bool:
        """
        验证布局有效性
        
        Args:
            layout: 待验证的布局
            
        Returns:
            布局是否有效
        """
        if not isinstance(layout, list):
            return False
        
        if len(layout) == 0:
            return False
        
        # 检查是否有有效的科室放置
        placed_count = sum(1 for dept in layout if dept is not None)
        return placed_count > 0
    
    def get_calculation_metadata(self) -> Dict[str, Any]:
        """
        获取计算器元数据信息
        
        Returns:
            包含计算器状态和配置的元数据字典
        """
        return {
            'calculator_type': self.__class__.__name__,
            'config': {
                key: getattr(self.config, key) for key in dir(self.config)
                if key.startswith('ADJACENCY_') and not key.startswith('_')
            },
            'is_initialized': hasattr(self, '_initialized') and self._initialized
        }

class MatrixBasedCalculator(BaseAdjacencyCalculator):
    """
    基于矩阵的计算器基类
    
    为需要预计算矩阵的计算器提供通用功能
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._adjacency_matrix = None
        self._matrix_computed = False
    
    def get_or_compute_matrix(self) -> np.ndarray:
        """
        获取或计算相邻性矩阵
        
        Returns:
            相邻性矩阵
        """
        if not self._matrix_computed:
            self._adjacency_matrix = self.calculate_adjacency_matrix()
            self._matrix_computed = True
        
        return self._adjacency_matrix
    
    def invalidate_matrix(self):
        """使矩阵缓存失效"""
        self._matrix_computed = False
        self._adjacency_matrix = None

class GraphBasedCalculator(BaseAdjacencyCalculator):
    """
    基于图结构的计算器基类
    
    为需要图结构分析的计算器提供通用功能
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._graph = None
        self._graph_computed = False
    
    @abstractmethod
    def build_graph(self) -> Any:
        """
        构建图结构
        子类必须实现此方法
        
        Returns:
            图对象（NetworkX Graph或其他图结构）
        """
        pass
    
    def get_or_build_graph(self) -> Any:
        """
        获取或构建图结构
        
        Returns:
            图对象
        """
        if not self._graph_computed:
            self._graph = self.build_graph()
            self._graph_computed = True
        
        return self._graph
    
    def invalidate_graph(self):
        """使图缓存失效"""
        self._graph_computed = False
        self._graph = None
```

### 奖励集成器接口

```python
# src/algorithms/adjacency/core/reward_integrator.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class RewardIntegratorInterface(ABC):
    """奖励集成器接口"""
    
    @abstractmethod
    def integrate_scores(self, *scores) -> float:
        """
        集成多个得分为最终奖励
        
        Args:
            *scores: 各维度得分
            
        Returns:
            集成后的奖励值
        """
        pass
    
    @abstractmethod
    def get_integration_weights(self) -> Dict[str, float]:
        """
        获取集成权重配置
        
        Returns:
            权重配置字典
        """
        pass

class WeightedRewardIntegrator(RewardIntegratorInterface):
    """
    加权奖励集成器实现
    
    使用配置的权重进行线性加权求和
    """
    
    def __init__(self, config):
        self.config = config
        self._normalize_weights()
    
    def _normalize_weights(self):
        """归一化权重"""
        total_weight = (
            self.config.SPATIAL_ADJACENCY_WEIGHT +
            self.config.FUNCTIONAL_ADJACENCY_WEIGHT +
            self.config.CONNECTIVITY_ADJACENCY_WEIGHT
        )
        
        if total_weight > 0:
            self.spatial_weight = self.config.SPATIAL_ADJACENCY_WEIGHT / total_weight
            self.functional_weight = self.config.FUNCTIONAL_ADJACENCY_WEIGHT / total_weight
            self.connectivity_weight = self.config.CONNECTIVITY_ADJACENCY_WEIGHT / total_weight
        else:
            # 均匀权重
            self.spatial_weight = self.functional_weight = self.connectivity_weight = 1/3
    
    def integrate_scores(self, spatial_score: float, functional_score: float, 
                        connectivity_score: float) -> float:
        """集成三个维度的得分"""
        integrated_score = (
            self.spatial_weight * spatial_score +
            self.functional_weight * functional_score +
            self.connectivity_weight * connectivity_score
        )
        
        return integrated_score * self.config.ADJACENCY_REWARD_WEIGHT
    
    def get_integration_weights(self) -> Dict[str, float]:
        """获取当前权重配置"""
        return {
            'spatial_weight': self.spatial_weight,
            'functional_weight': self.functional_weight,
            'connectivity_weight': self.connectivity_weight,
            'overall_weight': self.config.ADJACENCY_REWARD_WEIGHT
        }
```

## 数据结构定义

### 相邻性数据结构

```python
# src/algorithms/adjacency/utils/data_structures.py

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class AdjacencyRelation:
    """相邻性关系数据结构"""
    from_item: str              # 源项目（槽位或科室）
    to_item: str                # 目标项目
    strength: float             # 相邻性强度 (0-1)
    relation_type: str          # 关系类型：'spatial', 'functional', 'connectivity'
    metadata: Dict[str, Any]    # 附加元数据

@dataclass
class LayoutScore:
    """布局评分数据结构"""
    layout: List[str]                    # 布局配置
    spatial_score: float                 # 空间相邻性得分
    functional_score: float              # 功能相邻性得分
    connectivity_score: float            # 连通性得分
    total_score: float                   # 总得分
    adjacency_relations: List[AdjacencyRelation]  # 相邻关系列表
    computation_time: float              # 计算耗时（毫秒）
    metadata: Dict[str, Any]             # 计算元数据

@dataclass
class AdjacencyMatrix:
    """相邻性矩阵数据结构"""
    matrix: np.ndarray                   # 相邻性矩阵
    item_names: List[str]                # 项目名称列表
    matrix_type: str                     # 矩阵类型
    computation_params: Dict[str, Any]   # 计算参数
    creation_time: float                 # 创建时间戳

class AdjacencyGraph:
    """相邻性图数据结构"""
    
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str, float]]):
        self.nodes = nodes
        self.edges = edges
        self._adjacency_list = self._build_adjacency_list()
    
    def _build_adjacency_list(self) -> Dict[str, List[Tuple[str, float]]]:
        """构建邻接表"""
        adjacency_list = {node: [] for node in self.nodes}
        for from_node, to_node, weight in self.edges:
            adjacency_list[from_node].append((to_node, weight))
        return adjacency_list
    
    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """获取节点的邻居"""
        return self._adjacency_list.get(node, [])
    
    def get_edge_weight(self, from_node: str, to_node: str) -> Optional[float]:
        """获取边的权重"""
        neighbors = self.get_neighbors(from_node)
        for neighbor, weight in neighbors:
            if neighbor == to_node:
                return weight
        return None

@dataclass
class CacheEntry:
    """缓存条目数据结构"""
    key: Tuple[str, ...]        # 缓存键
    value: Any                  # 缓存值
    access_count: int           # 访问次数
    last_access_time: float     # 最后访问时间
    creation_time: float        # 创建时间
    size_bytes: int             # 条目大小（字节）
```

### 配置数据结构

```python
# src/algorithms/adjacency/config/adjacency_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class AdjacencyConfig:
    """相邻性计算配置"""
    
    # 基础配置
    enable_adjacency_reward: bool = True
    adjacency_reward_weight: float = 0.15
    
    # 算法权重配置
    spatial_adjacency_weight: float = 0.4
    functional_adjacency_weight: float = 0.5
    connectivity_adjacency_weight: float = 0.1
    
    # 相邻性判定参数
    adjacency_percentile_threshold: float = 0.2
    adjacency_k_nearest: Optional[int] = None
    adjacency_cluster_eps_percentile: float = 0.1
    adjacency_min_cluster_size: int = 2
    
    # 奖励计算参数
    adjacency_reward_base: float = 5.0
    adjacency_penalty_multiplier: float = 1.5
    
    # 性能优化配置
    adjacency_cache_size: int = 500
    adjacency_precompute: bool = True
    enable_parallel_computation: bool = False
    max_computation_threads: int = 2
    
    # 医疗偏好配置
    medical_adjacency_preferences: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # 正向偏好
            "急诊科": {
                "放射科": 0.8,
                "检验中心": 0.7,
                "手术室": 0.6,
            },
            "妇科": {
                "产科": 0.9,
                "超声科": 0.6,
            },
            "儿科": {
                "挂号收费": 0.5,
                "检验中心": 0.6,
            },
            # 负向约束
            "手术室": {
                "挂号收费": -0.8,
                "急诊科": -0.3,
            },
            "透析中心": {
                "急诊科": -0.5,
                "挂号收费": -0.6,
            }
        }
    )
    
    # 调试和监控配置
    enable_detailed_logging: bool = False
    log_adjacency_statistics: bool = True
    enable_computation_profiling: bool = False
    
    def validate(self) -> List[str]:
        """
        验证配置有效性
        
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        # 权重验证
        if not 0 <= self.adjacency_reward_weight <= 1:
            errors.append("adjacency_reward_weight must be between 0 and 1")
        
        total_dimension_weight = (
            self.spatial_adjacency_weight +
            self.functional_adjacency_weight +
            self.connectivity_adjacency_weight
        )
        if total_dimension_weight <= 0:
            errors.append("Sum of dimension weights must be positive")
        
        # 参数范围验证
        if not 0 < self.adjacency_percentile_threshold <= 1:
            errors.append("adjacency_percentile_threshold must be between 0 and 1")
        
        if self.adjacency_min_cluster_size < 1:
            errors.append("adjacency_min_cluster_size must be at least 1")
        
        if self.adjacency_cache_size < 0:
            errors.append("adjacency_cache_size must be non-negative")
        
        # 医疗偏好验证
        for dept, preferences in self.medical_adjacency_preferences.items():
            for target_dept, preference in preferences.items():
                if not -1 <= preference <= 1:
                    errors.append(f"Preference {dept}->{target_dept} must be between -1 and 1")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdjacencyConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
```

## 编码规范和最佳实践

### 代码风格规范

```python
# 命名规范
class AdjacencyAnalyzer:           # 类名：PascalCase
    def calculate_score(self):      # 方法名：snake_case
        adjacency_matrix = None     # 变量名：snake_case
        MAX_ITERATIONS = 100        # 常量：UPPER_CASE

# 类型注解规范
def calculate_adjacency_reward(
    self,
    layout: List[str],              # 必须的类型注解
    config: Optional[RLConfig] = None,
    validate: bool = True
) -> float:                         # 返回类型注解
    """
    计算相邻性奖励
    
    Args:
        layout: 当前布局配置
        config: 可选的配置对象
        validate: 是否验证输入
        
    Returns:
        相邻性奖励值
        
    Raises:
        ValueError: 当layout无效时
        RuntimeError: 当计算失败时
    """
    pass

# 错误处理规范
try:
    result = self._compute_adjacency_matrix()
except ValueError as e:
    logger.error(f"Invalid input for adjacency computation: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error in adjacency computation: {e}")
    return self._get_default_reward()
finally:
    self._cleanup_temporary_data()
```

### 性能优化最佳实践

```python
# 1. 缓存策略
class AdjacencyCache:
    def __init__(self):
        self._lru_cache = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0
    
    def get_cache_efficiency(self) -> float:
        """计算缓存命中率"""
        total_requests = self._hit_count + self._miss_count
        return self._hit_count / total_requests if total_requests > 0 else 0.0

# 2. 预计算策略
def precompute_adjacency_matrices(self):
    """预计算相邻性矩阵"""
    logger.info("Starting adjacency matrix precomputation...")
    
    # 使用numpy进行向量化计算
    distances = cdist(self.slot_coordinates, self.slot_coordinates, metric='euclidean')
    
    # 并行计算（如果启用）
    if self.config.enable_parallel_computation:
        with ThreadPoolExecutor(max_workers=self.config.max_computation_threads) as executor:
            futures = [
                executor.submit(self._compute_slot_adjacency, i)
                for i in range(self.n_slots)
            ]
            results = [future.result() for future in futures]
    
    logger.info("Adjacency matrix precomputation completed")

# 3. 内存管理
def _cleanup_temporary_data(self):
    """清理临时数据"""
    if hasattr(self, '_temp_matrices'):
        del self._temp_matrices
    
    # 强制垃圾回收
    import gc
    gc.collect()
```

### 日志记录规范

```python
import logging
from src.rl_optimizer.utils.setup import setup_logger

# 模块级logger
logger = setup_logger(__name__)

class AdjacencyAnalyzer:
    def calculate_adjacency_reward(self, layout: List[str]) -> float:
        # DEBUG级别：详细调试信息
        logger.debug(f"Computing adjacency reward for layout with {len(layout)} slots")
        
        # INFO级别：重要操作信息
        logger.info(f"Adjacency reward calculation completed: reward={reward:.4f}")
        
        # WARNING级别：潜在问题
        logger.warning(f"Layout has {empty_slots} empty slots, may affect reward quality")
        
        # ERROR级别：错误情况
        logger.error(f"Failed to compute adjacency matrix: {error}")
        
        # 统计信息日志
        if self.config.log_adjacency_statistics:
            self._log_computation_statistics(layout, reward)
```

## 扩展性和维护性指导

### 插件式扩展架构

```python
# 计算器插件接口
class AdjacencyCalculatorPlugin(BaseAdjacencyCalculator):
    """相邻性计算器插件基类"""
    
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """插件名称"""
        pass
    
    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """插件版本"""
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """获取支持的功能列表"""
        pass

# 插件管理器
class AdjacencyPluginManager:
    """相邻性计算器插件管理器"""
    
    def __init__(self):
        self._plugins = {}
        self._load_built_in_plugins()
    
    def register_plugin(self, plugin: AdjacencyCalculatorPlugin):
        """注册插件"""
        self._plugins[plugin.plugin_name] = plugin
    
    def get_plugin(self, name: str) -> Optional[AdjacencyCalculatorPlugin]:
        """获取插件"""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """列出所有可用插件"""
        return list(self._plugins.keys())

# 使用示例
plugin_manager = AdjacencyPluginManager()
custom_calculator = plugin_manager.get_plugin("custom_spatial_calculator")
```

### 版本兼容性设计

```python
# 版本兼容性管理
class CompatibilityManager:
    """版本兼容性管理器"""
    
    SUPPORTED_VERSIONS = ['1.0', '1.1', '1.2']
    
    @classmethod
    def migrate_config(cls, config_dict: Dict[str, Any], 
                      from_version: str, to_version: str) -> Dict[str, Any]:
        """配置迁移"""
        if from_version == '1.0' and to_version == '1.1':
            # 添加新配置项的默认值
            config_dict.setdefault('enable_parallel_computation', False)
        
        return config_dict
    
    @classmethod
    def check_compatibility(cls, version: str) -> bool:
        """检查版本兼容性"""
        return version in cls.SUPPORTED_VERSIONS
```

### 单元测试架构

```python
# 测试基类
class AdjacencyTestBase(unittest.TestCase):
    """相邻性模块测试基类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = self._create_test_config()
        self.test_layout = ['急诊科', '放射科', None, '检验中心']
        self.mock_travel_times = self._create_mock_travel_times()
    
    def _create_test_config(self):
        """创建测试配置"""
        from src.algorithms.adjacency.config import AdjacencyConfig
        return AdjacencyConfig(
            enable_adjacency_reward=True,
            adjacency_reward_weight=0.15,
            adjacency_cache_size=10  # 小缓存用于测试
        )
    
    def _create_mock_travel_times(self):
        """创建模拟通行时间数据"""
        return np.array([
            [0, 10, 20, 15],
            [10, 0, 25, 12],
            [20, 25, 0, 30],
            [15, 12, 30, 0]
        ])

# 具体测试类
class TestSpatialAdjacencyCalculator(AdjacencyTestBase):
    def test_adjacency_matrix_computation(self):
        """测试相邻性矩阵计算"""
        calculator = SpatialAdjacencyCalculator(self.config, self.mock_travel_times)
        matrix = calculator.calculate_adjacency_matrix()
        
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(np.all(np.diag(matrix) == 0))  # 对角线应为0
```

## 质量保证检查清单

### 代码质量检查

- [ ] 所有公共方法都有类型注解
- [ ] 所有类和方法都有完整的文档字符串
- [ ] 错误处理覆盖了所有可能的异常情况
- [ ] 日志记录适当且信息丰富
- [ ] 没有硬编码的魔术数字
- [ ] 使用了适当的设计模式

### 性能质量检查

- [ ] 相邻性计算时间复杂度不超过O(n²)
- [ ] 内存使用量在合理范围内（< 100MB）
- [ ] 缓存机制有效降低重复计算
- [ ] 支持大规模布局（> 50个槽位）的计算
- [ ] 并行计算（如启用）能正确工作

### 功能质量检查

- [ ] 所有算法产生合理的相邻性判断结果
- [ ] 奖励值在预期范围内
- [ ] 配置参数变化能正确影响计算结果
- [ ] 边界情况处理正确
- [ ] 与现有奖励机制协调工作

---

**文档版本：** v1.0  
**架构设计：** 产品经理架构师  
**更新时间：** 2025-08-14  
**代码规范：** PEP 8, Google Style Guide  
**质量标准：** 90%+ 测试覆盖率，< 10%计算时间开销