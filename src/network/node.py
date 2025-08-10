"""简化的节点类，用于网络图构建"""

from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Node:
    """
    表示网络图中的一个节点
    
    存储节点的基本信息，包括位置、类型、名称等属性
    """
    
    def __init__(self, 
                 node_id: int,
                 x: Optional[float] = None, 
                 y: Optional[float] = None, 
                 z: Optional[float] = None,
                 node_type: str = "",
                 name: Optional[str] = None,
                 pos: Optional[Tuple[float, float, float]] = None,
                 time: float = 1.0,
                 default_time: Optional[float] = None,
                 color: Optional[Tuple[int, int, int]] = None,
                 **kwargs):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识符
            x, y, z: 节点的三维坐标（如果提供pos则忽略这些参数）
            node_type: 节点类型（如房间、走廊、门等）
            name: 节点名称
            pos: 位置元组 (x, y, z)，如果提供则优先使用
            time: 节点通行时间
            default_time: 默认通行时间（如果提供则使用此值作为time）
            color: 节点颜色（RGB元组）
            **kwargs: 其他属性
        """
        self.id = node_id
        
        # 处理位置参数
        if pos is not None:
            # 确保位置坐标是数值类型
            pos_values = [float(v) if v is not None else 0.0 for v in pos]
            self.x, self.y, self.z = pos_values
        else:
            self.x = float(x) if x is not None else 0.0
            self.y = float(y) if y is not None else 0.0
            self.z = float(z) if z is not None else 0.0
        
        self.node_type = node_type
        self.name = name if name is not None else f"{node_type}_{node_id}"
        self.time = default_time if default_time is not None else time
        self.color = color
        
        # 存储其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """获取节点位置坐标"""
        return (self.x, self.y, self.z)
    
    def __str__(self) -> str:
        """节点的字符串表示"""
        # 确保坐标是数值类型
        x_val = float(self.x) if self.x is not None else 0.0
        y_val = float(self.y) if self.y is not None else 0.0
        z_val = float(self.z) if self.z is not None else 0.0
        return f"Node(id={self.id}, type={self.node_type}, name={self.name}, pos=({x_val:.1f}, {y_val:.1f}, {z_val:.1f}))"
    
    def __repr__(self) -> str:
        """节点的详细字符串表示"""
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典格式"""
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'type': self.node_type,
            'name': self.name,
            'time': self.time,
            'color': self.color
        }