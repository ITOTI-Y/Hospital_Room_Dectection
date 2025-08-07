"""
遗传算法优化器 - 基于遗传算法的布局优化
"""

import random
import copy
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.algorithms.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.env.cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """遗传算法中的个体"""
    layout: List[str]
    fitness: float = float('inf')
    age: int = 0
    
    def __post_init__(self):
        if self.fitness == float('inf'):
            # 如果没有设置适应度，则需要外部计算
            pass


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    遗传算法优化器
    
    使用遗传算法进行布局优化。遗传算法模拟自然选择和进化过程，
    通过选择、交叉、变异等操作来寻找最优解。
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: ConstraintManager,
                 population_size: int = 100,
                 elite_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 5,
                 max_age: int = 50):
        """
        初始化遗传算法优化器
        
        Args:
            cost_calculator: 成本计算器
            constraint_manager: 约束管理器
            population_size: 种群大小
            elite_size: 精英个体数量
            mutation_rate: 变异率
            crossover_rate: 交叉率
            tournament_size: 锦标赛选择的参与者数量
            max_age: 个体最大年龄
        """
        super().__init__(cost_calculator, constraint_manager, "GeneticAlgorithm")
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_age = max_age
        
        # 算法状态
        self.population = []
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness_history = []
        
        logger.info(f"遗传算法优化器初始化完成:")
        logger.info(f"  种群大小: {population_size}")
        logger.info(f"  精英数量: {elite_size}")
        logger.info(f"  变异率: {mutation_rate}")
        logger.info(f"  交叉率: {crossover_rate}")
        logger.info(f"  锦标赛大小: {tournament_size}")
    
    def optimize(self, 
                 initial_layout: Optional[List[str]] = None,
                 max_iterations: int = 1000,
                 convergence_threshold: int = 50,
                 **kwargs) -> OptimizationResult:
        """
        执行遗传算法优化
        
        Args:
            initial_layout: 初始布局（用于种群初始化的种子）
            max_iterations: 最大代数
            convergence_threshold: 收敛阈值（连续多少代无改进则停止）
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        self.start_optimization()
        
        logger.info(f"遗传算法优化开始，最大代数: {max_iterations}")
        
        # 初始化种群
        self._initialize_population(initial_layout)
        
        # 评估初始种群
        self._evaluate_population()
        
        # 更新最优解
        best_individual = min(self.population, key=lambda x: x.fitness)
        self.update_best_solution(best_individual.layout, best_individual.fitness)
        self.best_fitness_history.append(best_individual.fitness)
        
        logger.info(f"初始种群最优适应度: {best_individual.fitness:.2f}")
        
        # 重置算法状态
        self.generation = 0
        self.stagnation_count = 0
        
        try:
            while (self.generation < max_iterations and 
                   self.stagnation_count < convergence_threshold):
                
                # 执行一代进化
                self._evolve_generation()
                
                # 评估种群
                self._evaluate_population()
                
                # 更新最优解
                current_best = min(self.population, key=lambda x: x.fitness)
                previous_best = self.best_cost
                
                if current_best.fitness < self.best_cost:
                    self.update_best_solution(current_best.layout, current_best.fitness)
                    self.stagnation_count = 0
                    logger.info(f"第{self.generation}代发现更优解: {current_best.fitness:.2f}")
                else:
                    self.stagnation_count += 1
                
                self.best_fitness_history.append(current_best.fitness)
                self.generation += 1
                self.current_iteration = self.generation
                
                # 记录进化信息
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                self.log_iteration(current_best.fitness, 
                                 f"avg={avg_fitness:.2f}, stagnation={self.stagnation_count}")
                
                if self.generation % 50 == 0:
                    logger.info(f"第{self.generation}代: "
                              f"最优={current_best.fitness:.2f}, "
                              f"平均={avg_fitness:.2f}, "
                              f"停滞={self.stagnation_count}")
                
                # 种群多样性维护
                if self.generation % 100 == 0:
                    self._maintain_diversity()
                    
        except KeyboardInterrupt:
            logger.warning("遗传算法优化被用户中断")
        
        logger.info(f"遗传算法优化完成:")
        logger.info(f"  总代数: {self.generation}")
        logger.info(f"  停滞代数: {self.stagnation_count}")
        logger.info(f"  种群多样性: {self._calculate_diversity():.3f}")
        
        return self.finish_optimization()
    
    def _initialize_population(self, seed_layout: Optional[List[str]] = None):
        """
        初始化种群
        
        Args:
            seed_layout: 种子布局，用于生成部分个体
        """
        self.population = []
        
        # 如果有种子布局，用它生成一些个体
        if seed_layout is not None:
            self.population.append(Individual(layout=seed_layout.copy()))
            
            # 基于种子布局生成变异个体
            for _ in range(min(10, self.population_size // 4)):
                variant = seed_layout.copy()
                self._mutate_individual_layout(variant)
                if self.constraint_manager.is_valid_layout(variant):
                    self.population.append(Individual(layout=variant))
        
        # 生成剩余的随机个体
        while len(self.population) < self.population_size:
            random_layout = self.generate_initial_layout()
            self.population.append(Individual(layout=random_layout))
        
        logger.info(f"初始化种群完成，大小: {len(self.population)}")
    
    def _evaluate_population(self):
        """评估种群中所有个体的适应度"""
        for individual in self.population:
            if individual.fitness == float('inf'):  # 只计算未评估的个体
                individual.fitness = self.evaluate_layout(individual.layout)
    
    def _evolve_generation(self):
        """执行一代进化"""
        new_population = []
        
        # 1. 精英保留
        elite_individuals = self._select_elite()
        new_population.extend(elite_individuals)
        
        # 2. 生成新个体
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # 交叉生成后代
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                if random.random() < self.mutation_rate:
                    self._mutate_individual(child1)
                if random.random() < self.mutation_rate:
                    self._mutate_individual(child2)
                
                new_population.extend([child1, child2])
            else:
                # 复制并变异
                parent = self._tournament_selection()
                child = Individual(layout=parent.layout.copy())
                if random.random() < self.mutation_rate:
                    self._mutate_individual(child)
                new_population.append(child)
        
        # 3. 更新年龄并移除过老个体
        new_population = self._age_and_filter_population(new_population)
        
        # 4. 裁剪到目标大小
        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]
        
        self.population = new_population
    
    def _select_elite(self) -> List[Individual]:
        """选择精英个体"""
        # 按适应度排序并选择最优的
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        elite = []
        
        for individual in sorted_pop[:self.elite_size]:
            elite_copy = Individual(
                layout=individual.layout.copy(),
                fitness=individual.fitness,
                age=individual.age + 1
            )
            elite.append(elite_copy)
        
        return elite
    
    def _tournament_selection(self) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(self.population, 
                                 min(self.tournament_size, len(self.population)))
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        交叉操作 - 使用基于位置的交叉 (Position-based Crossover)
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            Tuple[Individual, Individual]: 两个子代
        """
        layout1, layout2 = parent1.layout.copy(), parent2.layout.copy()
        
        # 选择交叉点
        if len(layout1) > 2:
            # 使用顺序交叉 (Order Crossover, OX)
            child1_layout, child2_layout = self._order_crossover(layout1, layout2)
        else:
            child1_layout, child2_layout = layout1, layout2
        
        # 修复约束违反
        child1_layout = self._repair_layout_constraints(child1_layout)
        child2_layout = self._repair_layout_constraints(child2_layout)
        
        child1 = Individual(layout=child1_layout)
        child2 = Individual(layout=child2_layout)
        
        return child1, child2
    
    def _order_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        改进的顺序交叉操作（OX）
        确保不产生重复科室，保持布局的有效性
        """
        length = len(parent1)
        
        # 随机选择交叉区间
        start = random.randint(0, length - 1)
        end = random.randint(start, length - 1)
        
        # 初始化子代
        child1 = [None] * length
        child2 = [None] * length
        
        # 复制交叉区间
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # 填充剩余位置（改进的填充策略）
        self._fill_remaining_positions_improved(child1, parent2, start, end)
        self._fill_remaining_positions_improved(child2, parent1, start, end)
        
        # 验证并修复可能的问题
        child1 = self._validate_and_repair_child(child1, parent1)
        child2 = self._validate_and_repair_child(child2, parent2)
        
        return child1, child2
    
    def _fill_remaining_positions_improved(self, child: List[str], parent: List[str], start: int, end: int):
        """
        改进的填充策略，确保不产生重复
        使用循环填充法保持父代顺序
        """
        length = len(child)
        # 获取已经在子代中的科室集合
        child_set = set(item for item in child if item is not None)
        
        # 从parent中按顺序提取未在child中的科室
        fill_items = []
        for item in parent:
            if item is not None and item not in child_set:
                fill_items.append(item)
                child_set.add(item)  # 立即添加到集合中，防止重复
        
        # 循环填充：从end+1位置开始，循环到start-1
        fill_index = 0
        for offset in range(1, length):
            pos = (end + offset) % length
            if child[pos] is None and fill_index < len(fill_items):
                child[pos] = fill_items[fill_index]
                fill_index += 1
    
    def _validate_and_repair_child(self, child: List[str], parent: List[str]) -> List[str]:
        """
        验证子代有效性，修复可能的问题
        """
        # 检查是否有None值
        if None in child:
            # 用父代对应位置的值填充None
            for i, val in enumerate(child):
                if val is None:
                    child[i] = parent[i]
        
        # 检查是否有重复
        seen = set()
        duplicates = []
        for i, dept in enumerate(child):
            if dept in seen:
                duplicates.append(i)
            else:
                seen.add(dept)
        
        # 如果有重复，用缺失的科室替换
        if duplicates:
            all_depts = set(self.constraint_manager.placeable_departments)
            missing_depts = list(all_depts - seen)
            for i, dup_idx in enumerate(duplicates):
                if i < len(missing_depts):
                    child[dup_idx] = missing_depts[i]
                    seen.add(missing_depts[i])
        
        return child
    
    def _fill_remaining_positions(self, child: List[str], parent: List[str], start: int, end: int):
        """保留原方法签名以兼容，但使用改进的实现"""
        self._fill_remaining_positions_improved(child, parent, start, end)
    
    def _repair_layout_constraints(self, layout: List[str]) -> List[str]:
        """修复布局约束违反"""
        if self.constraint_manager.is_valid_layout(layout):
            return layout
        
        # 尝试修复
        repaired = layout.copy()
        
        # 处理重复的科室
        seen = set()
        duplicates = []
        for i, dept in enumerate(repaired):
            if dept is not None:
                if dept in seen:
                    duplicates.append(i)
                else:
                    seen.add(dept)
        
        # 用未使用的科室替换重复的科室
        available_depts = set(self.constraint_manager.placeable_departments) - seen
        available_depts_list = list(available_depts)
        
        for i, dup_idx in enumerate(duplicates):
            if i < len(available_depts_list):
                repaired[dup_idx] = available_depts_list[i]
        
        # 如果仍不满足约束，返回一个新的有效布局
        if not self.constraint_manager.is_valid_layout(repaired):
            return self.generate_initial_layout()
        
        return repaired
    
    def _mutate_individual(self, individual: Individual):
        """个体变异"""
        self._mutate_individual_layout(individual.layout)
        individual.fitness = float('inf')  # 重置适应度，需要重新计算
    
    def _mutate_individual_layout(self, layout: List[str]):
        """布局变异操作"""
        mutation_type = random.choice(['swap', 'insert', 'scramble'])
        
        if mutation_type == 'swap' and len(layout) >= 2:
            # 交换变异
            i, j = random.sample(range(len(layout)), 2)
            if self.constraint_manager.get_swap_candidates(layout, i, j):
                layout[i], layout[j] = layout[j], layout[i]
        
        elif mutation_type == 'insert' and len(layout) >= 2:
            # 插入变异
            i = random.randint(0, len(layout) - 1)
            j = random.randint(0, len(layout) - 1)
            if i != j:
                item = layout.pop(i)
                layout.insert(j, item)
        
        elif mutation_type == 'scramble' and len(layout) >= 3:
            # 乱序变异
            start = random.randint(0, len(layout) - 3)
            end = random.randint(start + 2, len(layout))
            segment = layout[start:end]
            random.shuffle(segment)
            layout[start:end] = segment
    
    def _age_and_filter_population(self, population: List[Individual]) -> List[Individual]:
        """更新年龄并过滤过老的个体"""
        filtered = []
        for individual in population:
            individual.age += 1
            if individual.age <= self.max_age:
                filtered.append(individual)
            else:
                # 替换过老的个体
                new_individual = Individual(layout=self.generate_initial_layout())
                filtered.append(new_individual)
        
        return filtered
    
    def _maintain_diversity(self):
        """维护种群多样性"""
        # 计算多样性
        diversity = self._calculate_diversity()
        
        if diversity < 0.1:  # 多样性过低
            logger.info(f"种群多样性过低({diversity:.3f})，注入新个体")
            
            # 替换一部分个体为随机个体
            num_to_replace = self.population_size // 4
            worst_individuals = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            
            for i in range(min(num_to_replace, len(worst_individuals))):
                new_layout = self.generate_initial_layout()
                worst_individuals[i].layout = new_layout
                worst_individuals[i].fitness = float('inf')
                worst_individuals[i].age = 0
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 1.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._layout_distance(self.population[i].layout, self.population[j].layout)
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        avg_distance = total_distance / comparisons
        max_distance = len(self.population[0].layout)  # 最大可能距离
        
        return avg_distance / max_distance if max_distance > 0 else 0
    
    def _layout_distance(self, layout1: List[str], layout2: List[str]) -> float:
        """计算两个布局之间的距离"""
        if len(layout1) != len(layout2):
            return float('inf')
        
        # 使用汉明距离
        distance = sum(1 for a, b in zip(layout1, layout2) if a != b)
        return distance / len(layout1)
    
    def get_additional_metrics(self) -> Dict[str, Any]:
        """获取遗传算法特定的额外指标"""
        diversity = self._calculate_diversity() if self.population else 0
        
        return {
            "population_size": self.population_size,
            "elite_size": self.elite_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "max_age": self.max_age,
            "final_generation": self.generation,
            "stagnation_count": self.stagnation_count,
            "population_diversity": diversity,
            "best_fitness_history": self.best_fitness_history.copy(),
            "convergence_rate": (self.best_fitness_history[0] - self.best_cost) / self.best_fitness_history[0] 
                               if self.best_fitness_history and self.best_fitness_history[0] > 0 else 0
        }