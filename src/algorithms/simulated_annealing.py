"""模拟退火优化器 - 基于模拟退火算法的布局优化."""

import math
import random
import copy
from typing import List, Optional, Dict, Any

from src.rl_optimizer.utils.setup import setup_logger
from src.algorithms.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.env.cost_calculator import CostCalculator

logger = setup_logger(__name__)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    模拟退火优化器
    
    使用模拟退火算法进行布局优化。模拟退火是一种启发式全局优化算法，
    通过模拟固体退火过程来寻找全局最优解。
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: ConstraintManager,
                 initial_temperature: float = 1000.0,
                 final_temperature: float = 0.1,
                 cooling_rate: float = 0.95,
                 temperature_length: int = 100):
        """
        初始化模拟退火优化器
        
        Args:
            cost_calculator: 成本计算器
            constraint_manager: 约束管理器
            initial_temperature: 初始温度
            final_temperature: 终止温度
            cooling_rate: 冷却速率
            temperature_length: 每个温度下的迭代次数
        """
        super().__init__(cost_calculator, constraint_manager, "SimulatedAnnealing")
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.temperature_length = temperature_length
        
        # 算法状态
        self.current_temperature = initial_temperature
        self.acceptance_count = 0
        self.rejection_count = 0
        self.improvement_count = 0
        
        logger.info(f"模拟退火优化器初始化完成:")
        logger.info(f"  初始温度: {initial_temperature}")
        logger.info(f"  终止温度: {final_temperature}") 
        logger.info(f"  冷却速率: {cooling_rate}")
        logger.info(f"  温度长度: {temperature_length}")
    
    def optimize(self, 
                 initial_layout: Optional[List[str]] = None,
                 max_iterations: int = 10000,
                 original_layout: Optional[List[str]] = None,
                 original_cost: Optional[float] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行模拟退火优化
        
        Args:
            initial_layout: 初始布局
            max_iterations: 最大迭代次数
            original_layout: 原始布局（未经优化的基准）
            original_cost: 原始布局的成本
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        self.start_optimization()
        
        # 保存原始布局信息
        self.original_layout = original_layout
        self.original_cost = original_cost
        
        # 初始化当前解
        if initial_layout is None:
            current_layout = self.generate_initial_layout()
        else:
            current_layout = initial_layout.copy()
            
        current_cost = self.evaluate_layout(current_layout)
        self.update_best_solution(current_layout, current_cost)
        
        logger.info(f"模拟退火优化开始，初始成本: {current_cost:.2f}")
        
        # 重置算法状态
        self.current_temperature = self.initial_temperature
        self.acceptance_count = 0
        self.rejection_count = 0
        self.improvement_count = 0
        
        try:
            while (self.current_temperature > self.final_temperature and 
                   self.current_iteration < max_iterations):
                
                # 在当前温度下进行多次迭代
                for _ in range(self.temperature_length):
                    if self.current_iteration >= max_iterations:
                        break
                    
                    # 生成邻域解
                    neighbor_layout = self._generate_neighbor(current_layout)
                    neighbor_cost = self.evaluate_layout(neighbor_layout)
                    
                    # 计算成本差
                    delta_cost = neighbor_cost - current_cost
                    
                    # 决定是否接受新解
                    if self._accept_solution(delta_cost, self.current_temperature):
                        current_layout = neighbor_layout
                        current_cost = neighbor_cost
                        self.acceptance_count += 1
                        
                        # 检查是否为新的最优解
                        if neighbor_cost < self.best_cost:
                            self.update_best_solution(neighbor_layout, neighbor_cost)
                            self.improvement_count += 1
                    else:
                        self.rejection_count += 1
                    
                    self.current_iteration += 1
                    self.log_iteration(current_cost, f"T={self.current_temperature:.2f}")
                
                # 降温
                self.current_temperature *= self.cooling_rate
                
                if self.current_iteration % 1000 == 0:
                    logger.info(f"迭代 {self.current_iteration}: "
                              f"当前成本={current_cost:.2f}, "
                              f"最优成本={self.best_cost:.2f}, "
                              f"温度={self.current_temperature:.2f}")
                              
        except KeyboardInterrupt:
            logger.warning("模拟退火优化被用户中断")
        
        logger.info(f"模拟退火优化完成:")
        logger.info(f"  最终温度: {self.current_temperature:.2f}")
        logger.info(f"  接受次数: {self.acceptance_count}")
        logger.info(f"  拒绝次数: {self.rejection_count}")
        logger.info(f"  改进次数: {self.improvement_count}")
        logger.info(f"  接受率: {self.acceptance_count/(self.acceptance_count + self.rejection_count)*100:.1f}%")
        
        return self.finish_optimization()
    
    def _generate_neighbor(self, current_layout: List[str]) -> List[str]:
        """
        生成邻域解
        
        Args:
            current_layout: 当前布局
            
        Returns:
            List[str]: 邻域布局
        """
        neighbor = current_layout.copy()
        
        # 选择邻域生成策略
        strategies = ['swap', 'relocate', 'multiple_swap']
        strategy = random.choice(strategies)
        
        if strategy == 'swap':
            # 交换两个科室的位置
            self._swap_departments(neighbor)
        elif strategy == 'relocate':
            # 重新定位一个科室
            self._relocate_department(neighbor)
        elif strategy == 'multiple_swap':
            # 多次小幅调整
            num_swaps = random.randint(2, min(5, len(neighbor)//2))
            for _ in range(num_swaps):
                if random.random() < 0.7:
                    self._swap_departments(neighbor)
                else:
                    self._relocate_department(neighbor)
        
        # 确保生成的邻域解满足约束
        if not self.constraint_manager.is_valid_layout(neighbor):
            # 如果不满足约束，尝试修复或返回原布局
            neighbor = self._repair_layout(neighbor) or current_layout.copy()
        
        return neighbor
    
    def _swap_departments(self, layout: List[str]):
        """随机交换两个科室的位置"""
        if len(layout) < 2:
            return
        
        # 过滤掉None值，只对有效的位置进行交换
        valid_positions = [i for i, dept in enumerate(layout) if dept is not None]
        if len(valid_positions) < 2:
            return
            
        # 随机选择两个不同的有效位置
        pos1, pos2 = random.sample(valid_positions, 2)
        
        # 检查是否可以交换（考虑固定位置约束）
        if self.constraint_manager.get_swap_candidates(layout, pos1, pos2):
            # 确保两个位置都不是None（额外的安全检查）
            if layout[pos1] is not None and layout[pos2] is not None:
                layout[pos1], layout[pos2] = layout[pos2], layout[pos1]
    
    def _relocate_department(self, layout: List[str]):
        """
        改进的重定位操作：安全地将一个科室移动到新位置
        使用更温和的方式保持布局结构完整性
        """
        if len(layout) < 2:
            return
        
        # 过滤掉None值，只对有效的位置进行操作
        valid_positions = [i for i, dept in enumerate(layout) if dept is not None]
        if len(valid_positions) < 2:
            return
            
        # 随机选择两个有效位置
        old_pos = random.choice(valid_positions)
        new_pos = random.choice(valid_positions)
        
        if old_pos != new_pos:
            # 保存要移动的科室
            dept = layout[old_pos]
            
            # 确保dept不是None
            if dept is None:
                return
            
            # 使用安全的移动策略：通过一系列移位来实现重定位
            # 这种方式保证不会丢失任何科室
            if old_pos < new_pos:
                # 向右移动：将中间的元素向左移一位
                temp = layout[old_pos]
                for i in range(old_pos, new_pos):
                    layout[i] = layout[i + 1]
                layout[new_pos] = temp
            else:
                # 向左移动：将中间的元素向右移一位
                temp = layout[old_pos]
                for i in range(old_pos, new_pos, -1):
                    layout[i] = layout[i - 1]
                layout[new_pos] = temp
    
    def _repair_layout(self, layout: List[str]) -> Optional[List[str]]:
        """
        尝试修复不满足约束的布局
        
        Args:
            layout: 待修复的布局
            
        Returns:
            Optional[List[str]]: 修复后的布局，如果无法修复则返回None
        """
        repaired = layout.copy()
        
        # 简单修复策略：随机重新分配违反约束的科室
        max_attempts = getattr(self.constraint_manager.config, 'SA_MAX_REPAIR_ATTEMPTS', 10)
        for attempt in range(max_attempts):  # 最多尝试次数从配置读取
            if self.constraint_manager.is_valid_layout(repaired):
                return repaired
                
            # 找到违反约束的位置并尝试修复
            for i, dept in enumerate(repaired):
                if dept is not None:
                    compatible_depts = self.constraint_manager.get_compatible_departments(i)
                    if dept not in compatible_depts and compatible_depts:
                        # 尝试找到一个兼容的科室进行交换
                        for j, other_dept in enumerate(repaired):
                            if other_dept in compatible_depts:
                                repaired[i], repaired[j] = repaired[j], repaired[i]
                                break
        
        return None  # 无法修复
    
    def _accept_solution(self, delta_cost: float, temperature: float) -> bool:
        """
        根据模拟退火接受准则决定是否接受新解
        
        Args:
            delta_cost: 成本差值
            temperature: 当前温度
            
        Returns:
            bool: 是否接受新解
        """
        if delta_cost <= 0:
            # 更优解总是接受
            return True
        
        if temperature <= 0:
            # 温度为0时只接受更优解
            return False
        
        # 按概率接受较差解
        probability = math.exp(-delta_cost / temperature)
        return random.random() < probability
    
    def get_additional_metrics(self) -> Dict[str, Any]:
        """获取模拟退火特定的额外指标"""
        total_attempts = self.acceptance_count + self.rejection_count
        acceptance_rate = self.acceptance_count / total_attempts if total_attempts > 0 else 0
        
        return {
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "current_temperature": self.current_temperature,
            "cooling_rate": self.cooling_rate,
            "temperature_length": self.temperature_length,
            "acceptance_count": self.acceptance_count,
            "rejection_count": self.rejection_count,
            "improvement_count": self.improvement_count,
            "acceptance_rate": acceptance_rate,
            "total_temperature_cycles": int((math.log(self.final_temperature/self.initial_temperature) / 
                                           math.log(self.cooling_rate)) if self.cooling_rate < 1 else 0)
        }