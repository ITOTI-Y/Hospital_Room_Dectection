"""简化的节点类，用于网络图构建"""

from typing import Tuple, Optional, Dict, Any
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class Node:
    """
    表示网络图中的一个节点
    
    存储节点的基本信息，包括位置、类型、名称等属性
    """
    
    def __init__(self, 
                 node_id: int,
                 x: float = 0.0, 
                 y: float = 0.0, 
                 z: float = 0.0,
                 node_type: str = "",
                 name: Optional[str] = None,
                 time: float = 1.0,
                 color: Optional[Tuple[int, int, int]] = None,
                 e_name: Optional[str] = None,
                 code: Optional[str] = None,
                 **kwargs):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识符
            x, y, z: 节点的三维坐标
            node_type: 节点类型（如房间、走廊、门等）
            name: 节点名称
            time: 节点通行时间
            color: 节点颜色（RGB元组）
            **kwargs: 其他属性
        """
        self.id = node_id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.node_type = node_type
        self.name = name if name is not None else f"{node_type}_{node_id}"
        self.time = float(time)
        self.color = color
        self.e_name = e_name
        self.code = code

        # 存储其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """获取节点位置坐标"""
        return (self.x, self.y, self.z)
    
    def __str__(self) -> str:
        """节点的字符串表示"""
        return f"Node(id={self.id}, type={self.node_type}, name={self.name}, pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}))"
    
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
            'color': self.color,
            'e_name': self.e_name,
            'code': self.code
        }