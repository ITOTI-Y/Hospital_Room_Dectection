"""
基础优化器抽象类 - 定义所有优化算法的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from dataclasses import dataclass

from src.rl_optimizer.env.cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    algorithm_name: str
    best_layout: List[str]
    best_cost: float
    execution_time: float
    iterations: int
    convergence_history: List[float]
    additional_metrics: Dict[str, Any]


class BaseOptimizer(ABC):
    """
    优化算法基类
    
    所有优化算法（PPO、模拟退火、遗传算法）都应继承此类，
    确保使用统一的接口和相同的目标函数（CostCalculator）。
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: 'ConstraintManager',
                 name: str = "BaseOptimizer"):
        """
        初始化基础优化器
        
        Args:
            cost_calculator: 统一的成本计算器
            constraint_manager: 约束管理器
            name: 算法名称
        """
        self.cost_calculator = cost_calculator
        self.constraint_manager = constraint_manager
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 优化过程跟踪
        self.current_iteration = 0
        self.best_cost = float('inf')
        self.best_layout = None
        self.convergence_history = []
        self.start_time = None
        
    @abstractmethod
    def optimize(self, 
                 initial_layout: Optional[List[str]] = None,
                 max_iterations: int = 1000,
                 **kwargs) -> OptimizationResult:
        """
        执行优化算法
        
        Args:
            initial_layout: 初始布局，如果为None则生成随机布局
            max_iterations: 最大迭代次数
            **kwargs: 算法特定参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        pass
    
    def evaluate_layout(self, layout: List[str]) -> float:
        """
        评估布局的成本
        
        Args:
            layout: 待评估的布局
            
        Returns:
            float: 布局成本
        """
        if not self.constraint_manager.is_valid_layout(layout):
            return float('inf')
        
        return self.cost_calculator.calculate_total_cost(layout)
    
    def generate_initial_layout(self) -> List[str]:
        """
        生成符合约束的初始布局
        
        Returns:
            List[str]: 初始布局
        """
        return self.constraint_manager.generate_valid_layout()
    
    def update_best_solution(self, layout: List[str], cost: float):
        """
        更新最优解
        
        Args:
            layout: 候选布局
            cost: 布局成本
        """
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_layout = layout.copy()
            self.logger.info(f"发现更优解: 成本 = {cost:.2f}, 迭代 = {self.current_iteration}")
    
    def start_optimization(self):
        """开始优化过程"""
        self.start_time = time.time()
        self.current_iteration = 0
        self.best_cost = float('inf')
        self.best_layout = None
        self.convergence_history = []
        self.logger.info(f"开始 {self.name} 优化")
    
    def finish_optimization(self) -> OptimizationResult:
        """
        结束优化过程并返回结果
        
        Returns:
            OptimizationResult: 优化结果
        """
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        result = OptimizationResult(
            algorithm_name=self.name,
            best_layout=self.best_layout,
            best_cost=self.best_cost,
            execution_time=execution_time,
            iterations=self.current_iteration,
            convergence_history=self.convergence_history.copy(),
            additional_metrics=self.get_additional_metrics()
        )
        
        self.logger.info(f"{self.name} 优化完成:")
        self.logger.info(f"  最优成本: {self.best_cost:.2f}")
        self.logger.info(f"  执行时间: {execution_time:.2f}s")
        self.logger.info(f"  迭代次数: {self.current_iteration}")
        
        return result
    
    def get_additional_metrics(self) -> Dict[str, Any]:
        """
        获取算法特定的额外指标
        
        Returns:
            Dict[str, Any]: 额外指标字典
        """
        return {}
    
    def log_iteration(self, cost: float, additional_info: str = ""):
        """
        记录迭代信息
        
        Args:
            cost: 当前成本
            additional_info: 额外信息
        """
        self.convergence_history.append(cost)
        
        if self.current_iteration % 100 == 0:  # 每100次迭代记录一次
            self.logger.debug(f"迭代 {self.current_iteration}: 成本 = {cost:.2f} {additional_info}")