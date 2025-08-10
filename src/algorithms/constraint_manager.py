"""
约束管理器 - 统一处理布局优化中的各种约束条件
"""

import logging
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class DepartmentInfo:
    """科室信息"""
    name: str
    area_requirement: float
    is_fixed: bool = False
    fixed_position: Optional[int] = None
    adjacency_preferences: List[str] = None
    
    def __post_init__(self):
        if self.adjacency_preferences is None:
            self.adjacency_preferences = []


@dataclass 
class SlotInfo:
    """槽位信息"""
    index: int
    name: str
    area: float
    is_available: bool = True


class ConstraintManager:
    """
    约束管理器
    
    统一管理布局优化中的各种约束条件，包括：
    - 面积约束：科室面积需求与槽位面积的匹配
    - 固定位置约束：某些科室必须位于特定位置
    - 相邻性约束：某些科室之间的邻接偏好
    - 容量约束：每个槽位只能放置一个科室
    """
    
    def __init__(self,
                 config: RLConfig,
                 cache_manager: CacheManager):
        """
        初始化约束管理器
        
        Args:
            config: RL配置对象，包含面积容差等参数
            cache_manager: 缓存管理器，提供节点和面积数据
        """
        self.config = config
        self.cache_manager = cache_manager
        self.placeable_slots = cache_manager.placeable_slots
        self.placeable_departments = cache_manager.placeable_departments
        self.area_tolerance_ratio = config.AREA_SCALING_FACTOR
        
        # 初始化槽位和科室信息
        self.slots_info = self._initialize_slots_info()
        self.departments_info = self._initialize_departments_info()
        
        # 构建索引映射（优化查找性能）
        self.dept_name_to_index = {
            dept.name: i for i, dept in enumerate(self.departments_info)
        }
        self.slot_name_to_index = {
            slot.name: i for i, slot in enumerate(self.slots_info)
        }
        
        # 构建约束关系
        self.area_compatibility_matrix = self._build_area_compatibility_matrix()
        self.fixed_assignments = self._build_fixed_assignments()
        
        logger.info(f"约束管理器初始化完成:")
        logger.info(f"  可用槽位数: {len(self.slots_info)}")
        logger.info(f"  可放置科室数: {len(self.departments_info)}")
        logger.info(f"  面积容差: {self.area_tolerance_ratio}")
        logger.info(f"  面积兼容对数: {self.area_compatibility_matrix.sum()}")
        logger.info(f"  固定分配数: {len(self.fixed_assignments)}")
    
    def _initialize_slots_info(self) -> List[SlotInfo]:
        """初始化槽位信息"""
        slots_info = []
        for i, slot_name in enumerate(self.placeable_slots):
            # 从行程时间矩阵中获取槽位面积信息
            # 这里假设槽位面积存储在travel_times的索引属性中
            # 如果没有面积信息，使用默认值
            area = self._get_slot_area(slot_name)
            slots_info.append(SlotInfo(
                index=i,
                name=slot_name,
                area=area,
                is_available=True
            ))
        return slots_info
    
    def _initialize_departments_info(self) -> List[DepartmentInfo]:
        """初始化科室信息"""
        departments_info = []
        for dept_name in self.placeable_departments:
            # 获取科室面积需求
            area_req = self._get_department_area_requirement(dept_name)
            
            # 检查是否为固定科室
            is_fixed, fixed_pos = self._check_fixed_department(dept_name)
            
            departments_info.append(DepartmentInfo(
                name=dept_name,
                area_requirement=area_req,
                is_fixed=is_fixed,
                fixed_position=fixed_pos,
                adjacency_preferences=self._get_adjacency_preferences(dept_name)
            ))
        return departments_info
    
    def _get_slot_area(self, slot_name: str) -> float:
        """
        获取槽位面积
        
        Args:
            slot_name: 槽位名称
            
        Returns:
            float: 槽位面积
        """
        # 从CacheManager获取真实面积数据
        placeable_df = self.cache_manager.placeable_nodes_df
        area_data = placeable_df[placeable_df['node_id'] == slot_name]['area']
        if not area_data.empty:
            return float(area_data.iloc[0])
        else:
            logger.warning(f"未找到槽位 {slot_name} 的面积信息，使用默认值")
            return 100.0  # 默认面积
    
    def _get_department_area_requirement(self, dept_name: str) -> float:
        """
        获取科室面积需求
        
        Args:
            dept_name: 科室名称
            
        Returns:
            float: 面积需求
        """
        # 从CacheManager获取真实面积数据
        placeable_df = self.cache_manager.placeable_nodes_df
        area_data = placeable_df[placeable_df['node_id'] == dept_name]['area']
        if not area_data.empty:
            return float(area_data.iloc[0])
        else:
            logger.warning(f"未找到科室 {dept_name} 的面积信息，使用默认值")
            return 100.0  # 默认面积
    
    def _check_fixed_department(self, dept_name: str) -> Tuple[bool, Optional[int]]:
        """
        检查科室是否需要固定位置
        
        Args:
            dept_name: 科室名称
            
        Returns:
            Tuple[bool, Optional[int]]: (是否固定, 固定位置索引)
        """
        # 这里定义需要固定位置的科室
        # 例如：入口、急诊科等可能需要固定在特定位置
        fixed_departments = {
            '急诊科': 0,  # 固定在第一个槽位
            '入口': None,  # 固定但位置待定
        }
        
        if dept_name in fixed_departments:
            return True, fixed_departments[dept_name]
        return False, None
    
    def _get_adjacency_preferences(self, dept_name: str) -> List[str]:
        """
        获取科室相邻偏好
        
        Args:
            dept_name: 科室名称
            
        Returns:
            List[str]: 偏好相邻的科室列表
        """
        # 定义科室间的相邻偏好关系
        adjacency_preferences = {
            '妇科': ['超声科', '采血处'],
            '心血管内科': ['采血处', '超声科', '放射科'],
            '呼吸内科': ['采血处', '放射科'],
            '采血处': ['检验中心'],
            '超声科': ['放射科'],
        }
        
        return adjacency_preferences.get(dept_name, [])
    
    def _build_area_compatibility_matrix(self) -> any:
        """
        构建面积兼容性矩阵
        
        Returns:
            numpy.ndarray: 兼容性矩阵 [slots x departments]
        """
        import numpy as np
        
        num_slots = len(self.slots_info)
        num_depts = len(self.departments_info)
        compatibility = np.zeros((num_slots, num_depts), dtype=bool)
        
        for i, slot in enumerate(self.slots_info):
            for j, dept in enumerate(self.departments_info):
                # 检查面积兼容性
                area_diff = abs(slot.area - dept.area_requirement)
                max_allowed_diff = dept.area_requirement * self.area_tolerance_ratio
                compatibility[i, j] = area_diff <= max_allowed_diff
        
        return compatibility
    
    def _build_fixed_assignments(self) -> Dict[str, int]:
        """
        构建固定分配映射
        
        Returns:
            Dict[str, int]: 科室名 -> 槽位索引的映射
        """
        fixed_assignments = {}
        for dept in self.departments_info:
            if dept.is_fixed and dept.fixed_position is not None:
                fixed_assignments[dept.name] = dept.fixed_position
        return fixed_assignments
    
    def is_valid_layout(self, layout: List[str]) -> bool:
        """
        检查布局是否满足所有约束（优化检查顺序：先简单后复杂）
        
        Args:
            layout: 布局列表，索引为槽位，值为科室名
            
        Returns:
            bool: 是否有效
        """
        # 1. 最快的检查：长度匹配
        if len(layout) != len(self.slots_info):
            return False
        
        # 2. 快速检查：唯一性约束（O(n)）
        if not self._check_uniqueness_constraints(layout):
            return False
        
        # 3. 中等复杂度：固定位置约束（O(固定数量)）
        if not self._check_fixed_constraints(layout):
            return False
        
        # 4. 最复杂的检查：面积约束（需要矩阵查找）
        if not self._check_area_constraints(layout):
            return False
        
        return True
    
    def _check_uniqueness_constraints(self, layout: List[str]) -> bool:
        """检查唯一性约束（每个科室只能出现一次，不允许null）- 优化版"""
        # 首先检查是否有None值
        if None in layout:
            logger.debug(f"布局中包含None值")
            return False
        
        seen = set()
        for dept_name in layout:
            # 已经检查过None，这里dept_name不会是None
            if dept_name in seen:
                return False  # 发现重复，立即返回
            seen.add(dept_name)
        
        # 检查是否所有科室都被放置
        placed_depts = seen
        all_depts = set(dept.name for dept in self.departments_info)
        missing_depts = all_depts - placed_depts
        
        if missing_depts:
            logger.debug(f"缺少科室: {missing_depts}")
            return False
        
        return True
    
    def _check_fixed_constraints(self, layout: List[str]) -> bool:
        """检查固定位置约束"""
        for dept_name, fixed_slot_idx in self.fixed_assignments.items():
            if layout[fixed_slot_idx] != dept_name:
                return False
        return True
    
    def _check_area_constraints(self, layout: List[str]) -> bool:
        """检查面积约束"""
        for slot_idx, dept_name in enumerate(layout):
            if dept_name is None:
                continue
            
            dept_idx = self._get_department_index(dept_name)
            if dept_idx is None:
                return False
            
            if not self.area_compatibility_matrix[slot_idx, dept_idx]:
                return False
        
        return True
    
    def _check_uniqueness_constraints(self, layout: List[str]) -> bool:
        """检查唯一性约束"""
        non_none_depts = [dept for dept in layout if dept is not None]
        return len(non_none_depts) == len(set(non_none_depts))
    
    def _get_department_index(self, dept_name: str) -> Optional[int]:
        """获取科室在departments_info中的索引（使用哈希表优化）"""
        return self.dept_name_to_index.get(dept_name)
    
    def generate_original_layout(self) -> List[str]:
        """
        生成原始布局（未经优化的基准布局）
        将科室按顺序直接映射到槽位，不进行任何随机化或优化
        
        Returns:
            List[str]: 原始布局
        """
        # 在这个系统中，placeable_departments 和 placeable_slots 是相同的
        # 原始布局就是简单地将科室按原始顺序排列
        # 确保返回的是副本，避免外部修改影响原始列表
        layout = self.placeable_departments.copy()
        
        # 验证布局完整性
        if len(layout) != len(self.slots_info):
            logger.warning(f"科室数量({len(layout)})与槽位数量({len(self.slots_info)})不匹配")
            # 如果数量不匹配，调整布局长度
            while len(layout) < len(self.slots_info):
                # 这种情况不应该发生，但为了健壮性添加处理
                logger.error(f"科室数量少于槽位数量，这不应该发生！")
                layout.append(layout[0])  # 临时填充，避免None
            layout = layout[:len(self.slots_info)]
        
        return layout
    
    def generate_valid_layout(self) -> List[str]:
        """
        生成一个满足约束的有效布局
        
        Returns:
            List[str]: 有效布局
        """
        # 首先创建一个包含所有科室的列表
        available_depts = list(self.placeable_departments)
        layout = []
        
        # 如果有固定位置约束，先处理
        fixed_positions = {}
        for dept_name, slot_idx in self.fixed_assignments.items():
            if dept_name in available_depts:
                fixed_positions[slot_idx] = dept_name
                available_depts.remove(dept_name)
        
        # 随机打乱剩余科室
        random.shuffle(available_depts)
        
        # 填充布局
        dept_idx = 0
        for slot_idx in range(len(self.slots_info)):
            if slot_idx in fixed_positions:
                # 使用固定位置的科室
                layout.append(fixed_positions[slot_idx])
            else:
                # 填充剩余科室
                if dept_idx < len(available_depts):
                    layout.append(available_depts[dept_idx])
                    dept_idx += 1
                else:
                    # 不应该发生，但为了健壮性
                    logger.error(f"生成布局时科室不足，这不应该发生！")
                    # 使用第一个可用科室填充
                    layout.append(self.placeable_departments[0])
        
        # 验证生成的布局
        if not self.is_valid_layout(layout):
            logger.warning("生成的布局不满足约束，返回原始布局")
            return self.generate_original_layout()
        
        return layout
    
    def get_compatible_departments(self, slot_idx: int) -> List[str]:
        """
        获取与指定槽位兼容的科室列表
        
        Args:
            slot_idx: 槽位索引
            
        Returns:
            List[str]: 兼容的科室名称列表
        """
        compatible_depts = []
        for dept_idx, dept in enumerate(self.departments_info):
            if self.area_compatibility_matrix[slot_idx, dept_idx]:
                compatible_depts.append(dept.name)
        return compatible_depts
    
    def get_swap_candidates(self, layout: List[str], slot1: int, slot2: int) -> bool:
        """
        检查两个槽位是否可以交换科室
        
        Args:
            layout: 当前布局
            slot1: 槽位1索引
            slot2: 槽位2索引
            
        Returns:
            bool: 是否可以交换
        """
        dept1, dept2 = layout[slot1], layout[slot2]
        
        # 创建交换后的布局进行验证
        new_layout = layout.copy()
        new_layout[slot1], new_layout[slot2] = dept2, dept1
        
        return self.is_valid_layout(new_layout)
    
    def calculate_constraint_violation_penalty(self, layout: List[str]) -> float:
        """
        计算布局的约束违反惩罚
        
        Args:
            layout: 布局
            
        Returns:
            float: 惩罚值（0表示无违反）
        """
        penalty = 0.0
        
        # 固定位置违反惩罚
        for dept_name, fixed_slot_idx in self.fixed_assignments.items():
            if layout[fixed_slot_idx] != dept_name:
                penalty += 1000.0  # 高惩罚
        
        # 面积约束违反惩罚
        for slot_idx, dept_name in enumerate(layout):
            if dept_name is None:
                continue
            
            dept_idx = self._get_department_index(dept_name)
            if dept_idx is not None and not self.area_compatibility_matrix[slot_idx, dept_idx]:
                penalty += 500.0  # 中等惩罚
        
        return penalty