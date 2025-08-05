"""
约束管理器 - 统一处理布局优化中的各种约束条件
"""

import logging
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

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
                 placeable_slots: List[str],
                 placeable_departments: List[str],
                 travel_times: pd.DataFrame,
                 area_tolerance_ratio: float = 0.3):
        """
        初始化约束管理器
        
        Args:
            placeable_slots: 可用槽位列表（原始节点名）
            placeable_departments: 可放置科室列表
            travel_times: 行程时间矩阵（用于获取槽位信息）
            area_tolerance_ratio: 面积容忍比例
        """
        self.placeable_slots = placeable_slots
        self.placeable_departments = placeable_departments
        self.travel_times = travel_times
        self.area_tolerance_ratio = area_tolerance_ratio
        
        # 初始化槽位和科室信息
        self.slots_info = self._initialize_slots_info()
        self.departments_info = self._initialize_departments_info()
        
        # 构建约束关系
        self.area_compatibility_matrix = self._build_area_compatibility_matrix()
        self.fixed_assignments = self._build_fixed_assignments()
        
        logger.info(f"约束管理器初始化完成:")
        logger.info(f"  可用槽位数: {len(self.slots_info)}")
        logger.info(f"  可放置科室数: {len(self.departments_info)}")
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
        # 这里需要根据实际的数据结构来获取面积
        # 临时使用随机值，实际应该从网络节点属性或配置中获取
        return random.uniform(50, 500)  # 临时实现
    
    def _get_department_area_requirement(self, dept_name: str) -> float:
        """
        获取科室面积需求
        
        Args:
            dept_name: 科室名称
            
        Returns:
            float: 面积需求
        """
        # 这里应该从配置或数据中获取科室面积需求
        # 临时使用随机值
        return random.uniform(80, 400)  # 临时实现
    
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
        检查布局是否满足所有约束
        
        Args:
            layout: 布局列表，索引为槽位，值为科室名
            
        Returns:
            bool: 是否有效
        """
        if len(layout) != len(self.slots_info):
            return False
        
        # 检查固定位置约束
        if not self._check_fixed_constraints(layout):
            return False
        
        # 检查面积约束
        if not self._check_area_constraints(layout):
            return False
        
        # 检查唯一性约束（每个科室只能出现一次）
        if not self._check_uniqueness_constraints(layout):
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
        """获取科室在departments_info中的索引"""
        for i, dept in enumerate(self.departments_info):
            if dept.name == dept_name:
                return i
        return None
    
    def generate_valid_layout(self) -> List[str]:
        """
        生成一个满足约束的有效布局
        
        Returns:
            List[str]: 有效布局
        """
        layout = [None] * len(self.slots_info)
        
        # 首先放置固定科室
        available_depts = set(self.placeable_departments)
        for dept_name, slot_idx in self.fixed_assignments.items():
            layout[slot_idx] = dept_name
            available_depts.remove(dept_name)
        
        # 随机放置剩余科室
        available_slots = [i for i, dept in enumerate(layout) if dept is None]
        available_depts_list = list(available_depts)
        
        for slot_idx in available_slots:
            if not available_depts_list:
                break
                
            # 找到与该槽位面积兼容的科室
            compatible_depts = []
            for dept_name in available_depts_list:
                dept_idx = self._get_department_index(dept_name)
                if dept_idx is not None and self.area_compatibility_matrix[slot_idx, dept_idx]:
                    compatible_depts.append(dept_name)
            
            if compatible_depts:
                chosen_dept = random.choice(compatible_depts)
                layout[slot_idx] = chosen_dept
                available_depts_list.remove(chosen_dept)
        
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