"""
相邻性计算器抽象基类和工具函数

提供所有相邻性计算器的统一接口定义和通用功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class AdjacencyCalculator(ABC):
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
    def get_adjacent_slots(self, slot_index: int) -> List[int]:
        """
        获取指定槽位的相邻槽位列表
        
        Args:
            slot_index: 槽位索引
            
        Returns:
            相邻槽位索引列表
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


class MatrixBasedCalculator(AdjacencyCalculator):
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


def safe_adjacency_calculation(func):
    """相邻性计算的异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, IndexError) as e:
            logger.error(f"相邻性计算错误：{e}")
            return 0.0
        except Exception as e:
            logger.error(f"未预期的相邻性计算错误：{e}")
            return 0.0
    return wrapper


def calculate_distance_percentile(distances: np.ndarray, percentile: float) -> float:
    """
    计算距离数组的指定分位数
    
    Args:
        distances: 距离数组
        percentile: 分位数（0-1之间）
        
    Returns:
        分位数值
    """
    if len(distances) == 0:
        return 0.0
    
    # 过滤掉零值（自身距离）
    valid_distances = distances[distances > 0]
    if len(valid_distances) == 0:
        return 0.0
    
    return np.percentile(valid_distances, percentile * 100)


def normalize_adjacency_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    归一化相邻性矩阵
    
    Args:
        matrix: 原始相邻性矩阵
        
    Returns:
        归一化后的矩阵
    """
    if matrix.size == 0:
        return matrix
    
    # 按行归一化
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    # 避免除零
    row_sums[row_sums == 0] = 1
    
    return matrix / row_sums


def validate_adjacency_preferences(preferences: Dict[str, Dict[str, float]]) -> bool:
    """
    验证医疗相邻性偏好配置的有效性
    
    Args:
        preferences: 相邻性偏好配置字典
        
    Returns:
        配置是否有效
    """
    try:
        for dept, dept_prefs in preferences.items():
            if not isinstance(dept, str) or not isinstance(dept_prefs, dict):
                return False
            
            for target_dept, preference in dept_prefs.items():
                if not isinstance(target_dept, str):
                    return False
                    
                if not isinstance(preference, (int, float)):
                    return False
                
                # 偏好值应在合理范围内
                if not -1 <= preference <= 1:
                    return False
        
        return True
    except Exception:
        return False