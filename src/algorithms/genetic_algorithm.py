"""
遗传算法优化器 - 基于遗传算法的布局优化
"""

import random
import copy
import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass

from src.algorithms.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.constraint_manager import ConstraintManager, SmartConstraintRepairer
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
    约束感知遗传算法优化器
    
    使用约束感知的遗传算法进行布局优化。集成了智能约束修复器、
    FPX交叉算子、约束导向变异等先进技术，大幅提升约束满足度和优化效果。
    
    主要特性：
    1. 智能初始种群生成（贪心面积匹配 + 随机多样性）
    2. FPX（Fixed Position Crossover）约束感知交叉算子
    3. 约束导向的智能变异策略
    4. 多策略约束修复机制
    5. 动态参数调整和种群多样性维护
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: ConstraintManager,
                 population_size: int = 100,
                 elite_size: int = 20,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.85,
                 tournament_size: int = 5,
                 max_age: int = 50,
                 constraint_repair_strategy: str = 'greedy_area_matching',
                 adaptive_parameters: bool = True):
        """
        初始化约束感知遗传算法优化器
        
        Args:
            cost_calculator: 成本计算器
            constraint_manager: 约束管理器
            population_size: 种群大小
            elite_size: 精英个体数量
            mutation_rate: 初始变异率
            crossover_rate: 初始交叉率
            tournament_size: 锦标赛选择的参与者数量
            max_age: 个体最大年龄
            constraint_repair_strategy: 约束修复策略
            adaptive_parameters: 是否启用自适应参数调整
        """
        super().__init__(cost_calculator, constraint_manager, "ConstraintAwareGeneticAlgorithm")
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_age = max_age
        self.constraint_repair_strategy = constraint_repair_strategy
        self.adaptive_parameters = adaptive_parameters
        
        # 初始化智能约束修复器
        self.constraint_repairer = SmartConstraintRepairer(constraint_manager)
        
        # 算法状态
        self.population = []
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness_history = []
        self.diversity_history = []
        
        # 自适应参数历史
        self.mutation_rate_history = []
        self.crossover_rate_history = []
        
        # 约束违反统计
        self.constraint_violation_count = 0
        self.successful_repair_count = 0
        
        # 性能统计
        self.fpx_crossover_count = 0
        self.smart_mutation_count = 0
        self.greedy_initialization_count = 0
        
        logger.info(f"约束感知遗传算法优化器初始化完成:")
        logger.info(f"  种群大小: {population_size}")
        logger.info(f"  精英数量: {elite_size}")
        logger.info(f"  初始变异率: {mutation_rate}")
        logger.info(f"  初始交叉率: {crossover_rate}")
        logger.info(f"  约束修复策略: {constraint_repair_strategy}")
        logger.info(f"  自适应参数: {adaptive_parameters}")
    
    def optimize(self, 
                 initial_layout: Optional[List[str]] = None,
                 max_iterations: int = 1000,
                 convergence_threshold: int = 50,
                 original_layout: Optional[List[str]] = None,
                 original_cost: Optional[float] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行遗传算法优化
        
        Args:
            initial_layout: 初始布局（用于种群初始化的种子）
            max_iterations: 最大代数
            convergence_threshold: 收敛阈值（连续多少代无改进则停止）
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
                
                # 自适应参数调整
                if self.generation > 0:
                    self._adapt_parameters()
                
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
                                 f"avg={avg_fitness:.2f}, stagnation={self.stagnation_count}, "
                                 f"diversity={self._calculate_diversity():.3f}")
                
                if self.generation % 50 == 0:
                    logger.info(f"第{self.generation}代: "
                              f"最优={current_best.fitness:.2f}, "
                              f"平均={avg_fitness:.2f}, "
                              f"停滞={self.stagnation_count}, "
                              f"多样性={self._calculate_diversity():.3f}")
                
                # 种群多样性维护
                if self.generation % 100 == 0:
                    self._maintain_diversity()
                
                # 更新约束统计信息
                if self.generation % 20 == 0:
                    self._update_constraint_statistics()
                    
        except KeyboardInterrupt:
            logger.warning("约束感知遗传算法优化被用户中断")
        
        logger.info(f"遗传算法优化完成:")
        logger.info(f"  总代数: {self.generation}")
        logger.info(f"  停滞代数: {self.stagnation_count}")
        logger.info(f"  种群多样性: {self._calculate_diversity():.3f}")
        
        return self.finish_optimization()
    
    def _initialize_population(self, seed_layout: Optional[List[str]] = None):
        """
        智能初始化种群
        
        结合贪心面积匹配和随机多样性策略生成高质量的初始种群。
        
        Args:
            seed_layout: 种子布局，用于生成部分个体
        """
        self.population = []
        
        # 1. 如果有种子布局，将其作为第一个个体
        if seed_layout is not None:
            if self.constraint_manager.is_valid_layout(seed_layout):
                self.population.append(Individual(layout=seed_layout.copy()))
                logger.debug("添加有效种子布局到种群")
            else:
                # 修复种子布局
                repaired_seed = self.constraint_repairer.repair_layout(seed_layout, self.constraint_repair_strategy)
                self.population.append(Individual(layout=repaired_seed))
                logger.debug("修复并添加种子布局到种群")
        
        # 2. 生成贪心面积匹配个体（种群的30%）
        greedy_count = min(int(self.population_size * 0.3), self.population_size - len(self.population))
        for _ in range(greedy_count):
            greedy_layout = self._generate_greedy_area_matched_layout()
            self.population.append(Individual(layout=greedy_layout))
            self.greedy_initialization_count += 1
        
        # 3. 基于种子布局生成变异个体（如果有种子布局，占种群的20%）
        if seed_layout is not None and len(self.population) < self.population_size:
            variant_count = min(int(self.population_size * 0.2), self.population_size - len(self.population))
            for _ in range(variant_count):
                variant_layout = seed_layout.copy()
                self._smart_mutate_layout(variant_layout)
                variant_layout = self.constraint_repairer.repair_layout(variant_layout, self.constraint_repair_strategy)
                self.population.append(Individual(layout=variant_layout))
        
        # 4. 生成约束感知的随机个体（填充剩余位置）
        while len(self.population) < self.population_size:
            random_layout = self._generate_constraint_aware_random_layout()
            self.population.append(Individual(layout=random_layout))
        
        logger.info(f"智能初始化种群完成:")
        logger.info(f"  总大小: {len(self.population)}")
        logger.info(f"  贪心个体: {self.greedy_initialization_count}")
        logger.info(f"  约束修复次数: {self.constraint_repairer.repair_attempts}")
    
    def _generate_greedy_area_matched_layout(self) -> List[str]:
        """
        生成基于贪心面积匹配的布局
        
        使用贪心算法优先分配面积匹配度最高的科室-槽位对。
        
        Returns:
            List[str]: 贪心面积匹配布局
        """
        import numpy as np
        
        # 获取所有科室和槽位
        departments = list(self.constraint_manager.placeable_departments)
        slots = list(range(len(self.constraint_manager.slots_info)))
        
        # 计算面积匹配度矩阵
        match_scores = np.zeros((len(slots), len(departments)))
        
        for slot_idx in slots:
            slot_area = self.constraint_manager.slots_info[slot_idx].area
            for dept_idx, dept_name in enumerate(departments):
                dept_area = self.constraint_manager.departments_info[dept_idx].area_requirement
                
                # 计算面积匹配度（值越大表示匹配度越高）
                if max(slot_area, dept_area) > 0:
                    area_ratio = min(slot_area, dept_area) / max(slot_area, dept_area)
                    # 结合约束兼容性
                    if self.constraint_manager.area_compatibility_matrix[slot_idx, dept_idx]:
                        match_scores[slot_idx, dept_idx] = area_ratio
                    else:
                        match_scores[slot_idx, dept_idx] = 0.0
                else:
                    match_scores[slot_idx, dept_idx] = 0.0
        
        # 贪心分配
        layout = [''] * len(slots)
        used_departments = set()
        used_slots = set()
        
        # 处理固定位置约束
        for dept_name, fixed_slot_idx in self.constraint_manager.fixed_assignments.items():
            if dept_name in departments and fixed_slot_idx in slots:
                layout[fixed_slot_idx] = dept_name
                used_departments.add(dept_name)
                used_slots.add(fixed_slot_idx)
        
        # 创建候选列表并按匹配度排序
        candidates = []
        for slot_idx in slots:
            if slot_idx not in used_slots:
                for dept_idx, dept_name in enumerate(departments):
                    if dept_name not in used_departments:
                        score = match_scores[slot_idx, dept_idx]
                        candidates.append((score, slot_idx, dept_name))
        
        # 按匹配度降序排序
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 贪心分配
        for score, slot_idx, dept_name in candidates:
            if slot_idx not in used_slots and dept_name not in used_departments:
                layout[slot_idx] = dept_name
                used_departments.add(dept_name)
                used_slots.add(slot_idx)
        
        return layout
    
    def _generate_constraint_aware_random_layout(self) -> List[str]:
        """
        生成约束感知的随机布局
        
        在随机分配的基础上，考虑约束兼容性，提高初始布局的有效性。
        
        Returns:
            List[str]: 约束感知的随机布局
        """
        # 首先生成一个基本的随机布局
        departments = list(self.constraint_manager.placeable_departments)
        random.shuffle(departments)
        
        layout = departments.copy()
        
        # 尝试改善约束违反
        max_improvement_attempts = 10
        for _ in range(max_improvement_attempts):
            violations = []
            
            # 检查面积约束违反
            for slot_idx, dept_name in enumerate(layout):
                dept_idx = self.constraint_manager.dept_name_to_index.get(dept_name)
                if (dept_idx is not None and 
                    not self.constraint_manager.area_compatibility_matrix[slot_idx, dept_idx]):
                    violations.append(slot_idx)
            
            if not violations:
                break
            
            # 随机选择一个违反位置进行改善
            violation_slot = random.choice(violations)
            compatible_depts = self.constraint_manager.get_compatible_departments(violation_slot)
            
            if compatible_depts:
                # 寻找可以交换的兼容科室
                for target_dept in compatible_depts:
                    if target_dept in layout:
                        target_slot = layout.index(target_dept)
                        # 检查交换是否改善整体约束满足度
                        if self.constraint_manager.get_swap_candidates(layout, violation_slot, target_slot):
                            layout[violation_slot], layout[target_slot] = layout[target_slot], layout[violation_slot]
                            break
        
        # 如果仍有约束违反，使用约束修复器
        if not self.constraint_manager.is_valid_layout(layout):
            layout = self.constraint_repairer.repair_layout(layout, 'random_repair', max_attempts=5)
        
        return layout
    
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
        FPX (Fixed Position Crossover) 约束感知交叉操作
        
        这是一种专门为约束布局问题设计的交叉算子，能够：
        1. 保持固定位置约束不变
        2. 在非固定位置进行智能信息交换
        3. 自动修复可能的约束违反
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            Tuple[Individual, Individual]: 两个子代
        """
        self.fpx_crossover_count += 1
        
        layout1, layout2 = parent1.layout.copy(), parent2.layout.copy()
        
        # 使用FPX算法进行约束感知交叉
        child1_layout, child2_layout = self._fpx_crossover(layout1, layout2)
        
        # 修复可能的约束违反
        child1_layout = self.constraint_repairer.repair_layout(child1_layout, self.constraint_repair_strategy, max_attempts=3)
        child2_layout = self.constraint_repairer.repair_layout(child2_layout, self.constraint_repair_strategy, max_attempts=3)
        
        child1 = Individual(layout=child1_layout)
        child2 = Individual(layout=child2_layout)
        
        return child1, child2
    
    def _fpx_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        Fixed Position Crossover (FPX) 算法实现
        
        FPX是专门为带固定位置约束的布局问题设计的交叉算子：
        1. 固定位置的科室保持不变
        2. 在非固定位置间进行有序交叉
        3. 使用启发式修复策略处理冲突
        
        Args:
            parent1: 父代1布局
            parent2: 父代2布局
            
        Returns:
            Tuple[List[str], List[str]]: 交叉后的两个子代布局
        """
        length = len(parent1)
        
        # 初始化子代
        child1 = [''] * length
        child2 = [''] * length
        
        # 1. 保持固定位置约束
        fixed_positions = set()
        for dept_name, slot_idx in self.constraint_manager.fixed_assignments.items():
            if slot_idx < length:
                child1[slot_idx] = dept_name
                child2[slot_idx] = dept_name
                fixed_positions.add(slot_idx)
        
        # 2. 获取非固定位置
        non_fixed_positions = [i for i in range(length) if i not in fixed_positions]
        
        if len(non_fixed_positions) <= 1:
            # 如果非固定位置太少，直接复制父代
            return parent1.copy(), parent2.copy()
        
        # 3. 在非固定位置执行改进的顺序交叉
        # 随机选择交叉区间
        start_idx = random.randint(0, len(non_fixed_positions) - 1)
        end_idx = random.randint(start_idx, len(non_fixed_positions) - 1)
        
        # 交叉区间的实际位置
        crossover_positions = non_fixed_positions[start_idx:end_idx + 1]
        
        # 复制交叉区间
        for pos in crossover_positions:
            child1[pos] = parent1[pos]
            child2[pos] = parent2[pos]
        
        # 4. 填充剩余位置（使用改进的填充策略）
        self._fill_remaining_positions_fpx(child1, parent2, crossover_positions, non_fixed_positions, fixed_positions)
        self._fill_remaining_positions_fpx(child2, parent1, crossover_positions, non_fixed_positions, fixed_positions)
        
        return child1, child2
    
    def _fill_remaining_positions_fpx(self, 
                                     child: List[str], 
                                     parent: List[str], 
                                     crossover_positions: List[int],
                                     non_fixed_positions: List[int],
                                     fixed_positions: Set[int]):
        """
        FPX交叉算子的智能填充策略
        
        使用约束感知的填充方法，优先考虑面积兼容性。
        
        Args:
            child: 待填充的子代
            parent: 参考的父代
            crossover_positions: 已交叉的位置
            non_fixed_positions: 所有非固定位置
            fixed_positions: 所有固定位置
        """
        # 获取已在子代中的科室
        child_departments = set(dept for dept in child if dept and dept != '')
        
        # 获取需要填充的位置
        positions_to_fill = [pos for pos in non_fixed_positions if pos not in crossover_positions]
        
        # 从父代中按顺序提取未在子代中的科室
        available_departments = []
        for dept in parent:
            if dept and dept != '' and dept not in child_departments:
                available_departments.append(dept)
                child_departments.add(dept)  # 立即添加以避免重复
        
        # 如果可用科室不足，从所有科室中补充
        all_departments = set(self.constraint_manager.placeable_departments)
        missing_departments = all_departments - child_departments
        available_departments.extend(list(missing_departments))
        
        # 智能分配：优先考虑面积兼容性
        for i, pos in enumerate(positions_to_fill):
            if i < len(available_departments):
                dept_to_assign = available_departments[i]
                
                # 检查是否有更好的面积匹配选择
                if i + 1 < len(available_departments):
                    current_dept = available_departments[i]
                    next_dept = available_departments[i + 1]
                    
                    # 比较面积匹配度
                    current_match = self._calculate_area_match_score(pos, current_dept)
                    next_match = self._calculate_area_match_score(pos, next_dept)
                    
                    # 如果下一个科室的匹配度明显更好，进行交换
                    if next_match > current_match + 0.1:  # 0.1是匹配度改善阈值
                        available_departments[i], available_departments[i + 1] = \
                            available_departments[i + 1], available_departments[i]
                        dept_to_assign = available_departments[i]
                
                child[pos] = dept_to_assign
    
    def _calculate_area_match_score(self, slot_idx: int, dept_name: str) -> float:
        """
        计算科室与槽位的面积匹配度
        
        Args:
            slot_idx: 槽位索引
            dept_name: 科室名称
            
        Returns:
            float: 匹配度得分 (0-1之间，越高越好)
        """
        dept_idx = self.constraint_manager.dept_name_to_index.get(dept_name)
        if dept_idx is None:
            return 0.0
        
        slot_area = self.constraint_manager.slots_info[slot_idx].area
        dept_area = self.constraint_manager.departments_info[dept_idx].area_requirement
        
        if max(slot_area, dept_area) == 0:
            return 0.0
        
        # 面积匹配度：越接近1表示匹配度越高
        area_ratio = min(slot_area, dept_area) / max(slot_area, dept_area)
        
        # 结合约束兼容性
        if self.constraint_manager.area_compatibility_matrix[slot_idx, dept_idx]:
            return area_ratio
        else:
            return 0.0  # 不兼容则匹配度为0
    
    def _smart_mutate_layout(self, layout: List[str]):
        """
        约束导向的智能变异操作
        
        这个变异算子考虑约束条件，优先进行有益的变异：
        1. 识别约束违反位置
        2. 优先修复违反的约束
        3. 在不违反约束的前提下进行多样性变异
        
        Args:
            layout: 待变异的布局
        """
        self.smart_mutation_count += 1
        
        # 1. 检查是否有约束违反
        violations = self.constraint_repairer._find_constraint_violations(layout)
        
        if violations:
            # 优先修复约束违反
            self._constraint_repair_mutation(layout, violations)
        else:
            # 进行多样性导向的变异
            self._diversity_oriented_mutation(layout)
    
    def _constraint_repair_mutation(self, layout: List[str], violations: List[int]):
        """
        约束修复导向的变异
        
        专门针对约束违反位置进行修复性变异。
        
        Args:
            layout: 布局
            violations: 违反约束的位置列表
        """
        # 随机选择一个违反位置进行修复
        violation_pos = random.choice(violations)
        
        # 获取该位置兼容的科室
        compatible_depts = self.constraint_manager.get_compatible_departments(violation_pos)
        
        if compatible_depts:
            # 寻找可以交换的科室
            current_dept = layout[violation_pos]
            
            for candidate_dept in compatible_depts:
                if candidate_dept in layout and candidate_dept != current_dept:
                    candidate_pos = layout.index(candidate_dept)
                    
                    # 检查交换是否有益
                    if self.constraint_manager.get_swap_candidates(layout, violation_pos, candidate_pos):
                        layout[violation_pos], layout[candidate_pos] = layout[candidate_pos], layout[violation_pos]
                        break
        
        # 如果交换修复失败，尝试其他变异策略
        if violation_pos in self.constraint_repairer._find_constraint_violations(layout):
            self._diversity_oriented_mutation(layout)
    
    def _diversity_oriented_mutation(self, layout: List[str]):
        """
        多样性导向的变异
        
        在不违反约束的前提下，进行多种变异操作以增加种群多样性。
        
        Args:
            layout: 布局
        """
        mutation_type = random.choice(['constraint_aware_swap', 'smart_insertion', 'local_scramble'])
        
        if mutation_type == 'constraint_aware_swap':
            self._constraint_aware_swap_mutation(layout)
        elif mutation_type == 'smart_insertion':
            self._smart_insertion_mutation(layout)
        elif mutation_type == 'local_scramble':
            self._local_scramble_mutation(layout)
    
    def _constraint_aware_swap_mutation(self, layout: List[str]):
        """约束感知的交换变异"""
        # 排除固定位置
        non_fixed_positions = [i for i in range(len(layout)) 
                              if i not in self.constraint_manager.fixed_assignments.values()]
        
        if len(non_fixed_positions) >= 2:
            pos1, pos2 = random.sample(non_fixed_positions, 2)
            
            # 只有当交换不违反约束时才执行
            if self.constraint_manager.get_swap_candidates(layout, pos1, pos2):
                layout[pos1], layout[pos2] = layout[pos2], layout[pos1]
    
    def _smart_insertion_mutation(self, layout: List[str]):
        """智能插入变异"""
        non_fixed_positions = [i for i in range(len(layout)) 
                              if i not in self.constraint_manager.fixed_assignments.values()]
        
        if len(non_fixed_positions) >= 2:
            # 选择源位置和目标位置
            source_pos = random.choice(non_fixed_positions)
            target_pos = random.choice(non_fixed_positions)
            
            if source_pos != target_pos:
                # 执行插入操作
                dept = layout.pop(source_pos)
                layout.insert(target_pos, dept)
                
                # 如果插入后违反约束，回滚操作
                if not self.constraint_manager.is_valid_layout(layout):
                    layout.pop(target_pos)
                    layout.insert(source_pos, dept)
    
    def _local_scramble_mutation(self, layout: List[str]):
        """局部乱序变异"""
        non_fixed_positions = [i for i in range(len(layout)) 
                              if i not in self.constraint_manager.fixed_assignments.values()]
        
        if len(non_fixed_positions) >= 3:
            # 随机选择一个局部区域
            start_idx = random.randint(0, len(non_fixed_positions) - 3)
            end_idx = min(start_idx + random.randint(2, 5), len(non_fixed_positions))
            
            # 获取区域内的实际位置
            region_positions = non_fixed_positions[start_idx:end_idx]
            
            # 提取该区域的科室
            region_depts = [layout[pos] for pos in region_positions]
            
            # 乱序并重新分配
            random.shuffle(region_depts)
            
            # 临时保存原始布局以备回滚
            original_layout = layout.copy()
            
            for pos, dept in zip(region_positions, region_depts):
                layout[pos] = dept
            
            # 如果乱序后违反约束，回滚操作
            if not self.constraint_manager.is_valid_layout(layout):
                for i, pos in enumerate(region_positions):
                    layout[pos] = original_layout[pos]
    
    def _mutate_individual(self, individual: Individual):
        """个体变异"""
        self._smart_mutate_layout(individual.layout)
        individual.fitness = float('inf')  # 重置适应度，需要重新计算
    
    def _adapt_parameters(self):
        """
        自适应参数调整
        
        根据算法的进化状态动态调整变异率和交叉率：
        1. 如果停滞时间过长，增加变异率以增强探索
        2. 如果种群多样性过低，调整参数促进多样性
        3. 记录参数变化历史用于分析
        """
        if not self.adaptive_parameters:
            return
        
        # 计算当前种群多样性
        current_diversity = self._calculate_diversity()
        self.diversity_history.append(current_diversity)
        
        # 根据停滞情况调整变异率
        if self.stagnation_count > 20:
            # 长期停滞，增加变异率
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        elif self.stagnation_count < 5:
            # 快速改进，适当降低变异率
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        
        # 根据多样性调整交叉率
        if current_diversity < 0.1:
            # 多样性过低，降低交叉率，增加随机性
            self.crossover_rate = max(0.6, self.crossover_rate * 0.9)
            self.mutation_rate = min(0.4, self.mutation_rate * 1.2)
        elif current_diversity > 0.8:
            # 多样性过高，增加交叉率，促进信息交换
            self.crossover_rate = min(0.95, self.crossover_rate * 1.05)
        
        # 记录参数变化
        self.mutation_rate_history.append(self.mutation_rate)
        self.crossover_rate_history.append(self.crossover_rate)
        
        if self.generation % 50 == 0:
            logger.debug(f"参数调整: 变异率={self.mutation_rate:.3f}, "
                        f"交叉率={self.crossover_rate:.3f}, "
                        f"多样性={current_diversity:.3f}")
    
    def _maintain_diversity(self):
        """维护种群多样性（增强版）"""
        # 计算多样性
        diversity = self._calculate_diversity()
        
        if diversity < 0.05:  # 多样性极低
            logger.info(f"种群多样性极低({diversity:.3f})，执行大规模重新初始化")
            
            # 保留最佳的20%个体
            num_to_keep = max(1, self.population_size // 5)
            best_individuals = sorted(self.population, key=lambda x: x.fitness)[:num_to_keep]
            
            # 重新生成剩余个体
            self.population = best_individuals.copy()
            while len(self.population) < self.population_size:
                new_layout = self._generate_constraint_aware_random_layout()
                self.population.append(Individual(layout=new_layout))
            
        elif diversity < 0.15:  # 多样性较低
            logger.info(f"种群多样性较低({diversity:.3f})，注入新个体")
            
            # 替换最差的25%个体
            num_to_replace = self.population_size // 4
            worst_individuals = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            
            for i in range(min(num_to_replace, len(worst_individuals))):
                # 使用不同的生成策略
                if i % 2 == 0:
                    new_layout = self._generate_greedy_area_matched_layout()
                else:
                    new_layout = self._generate_constraint_aware_random_layout()
                
                worst_individuals[i].layout = new_layout
                worst_individuals[i].fitness = float('inf')
                worst_individuals[i].age = 0
    
    def get_additional_metrics(self) -> Dict[str, Any]:
        """获取约束感知遗传算法的增强指标"""
        base_metrics = super().get_additional_metrics() if hasattr(super(), 'get_additional_metrics') else {}
        
        diversity = self._calculate_diversity() if self.population else 0
        
        # 计算约束违反率
        violation_rate = (self.constraint_violation_count / max(1, self.constraint_repairer.repair_attempts))
        repair_success_rate = (self.successful_repair_count / max(1, self.constraint_violation_count))
        
        # 算法特定指标
        constraint_aware_metrics = {
            # 基础参数
            "population_size": self.population_size,
            "elite_size": self.elite_size,
            "initial_mutation_rate": self.mutation_rate_history[0] if self.mutation_rate_history else self.mutation_rate,
            "initial_crossover_rate": self.crossover_rate_history[0] if self.crossover_rate_history else self.crossover_rate,
            "final_mutation_rate": self.mutation_rate,
            "final_crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "max_age": self.max_age,
            "constraint_repair_strategy": self.constraint_repair_strategy,
            "adaptive_parameters": self.adaptive_parameters,
            
            # 算法状态
            "final_generation": self.generation,
            "stagnation_count": self.stagnation_count,
            "population_diversity": diversity,
            
            # 约束处理统计
            "constraint_violation_count": self.constraint_violation_count,
            "successful_repair_count": self.successful_repair_count,
            "constraint_violation_rate": violation_rate,
            "constraint_repair_success_rate": repair_success_rate,
            
            # 算法性能统计
            "fpx_crossover_count": self.fpx_crossover_count,
            "smart_mutation_count": self.smart_mutation_count,
            "greedy_initialization_count": self.greedy_initialization_count,
            
            # 历史数据
            "best_fitness_history": self.best_fitness_history.copy(),
            "diversity_history": self.diversity_history.copy(),
            "mutation_rate_history": self.mutation_rate_history.copy(),
            "crossover_rate_history": self.crossover_rate_history.copy(),
            
            # 收敛性分析
            "convergence_rate": (self.best_fitness_history[0] - self.best_cost) / self.best_fitness_history[0] 
                               if self.best_fitness_history and self.best_fitness_history[0] > 0 else 0,
            
            # 约束修复器统计
            "constraint_repairer_stats": self.constraint_repairer.get_repair_statistics()
        }
        
        # 合并基础指标和约束感知指标
        base_metrics.update(constraint_aware_metrics)
        
        return base_metrics
    
    def reset_algorithm_state(self):
        """重置算法状态（用于多次运行）"""
        self.population = []
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.mutation_rate_history = []
        self.crossover_rate_history = []
        
        # 重置统计计数器
        self.constraint_violation_count = 0
        self.successful_repair_count = 0
        self.fpx_crossover_count = 0
        self.smart_mutation_count = 0
        self.greedy_initialization_count = 0
        
        # 重置约束修复器统计
        self.constraint_repairer.reset_statistics()
        
        # 重置父类状态
        if hasattr(super(), 'reset_algorithm_state'):
            super().reset_algorithm_state()
    
    def _update_constraint_statistics(self):
        """更新约束相关统计信息"""
        # 检查当前种群中的约束违反情况
        for individual in self.population:
            if not self.constraint_manager.is_valid_layout(individual.layout):
                self.constraint_violation_count += 1
        
        # 更新修复成功计数
        repair_stats = self.constraint_repairer.get_repair_statistics()
        self.successful_repair_count = repair_stats['successful_repairs']
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性（使用汉明距离）"""
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
        max_distance = 1.0  # 归一化的最大距离
        
        return avg_distance / max_distance if max_distance > 0 else 0
    
    def _layout_distance(self, layout1: List[str], layout2: List[str]) -> float:
        """计算两个布局之间的归一化汉明距离"""
        if len(layout1) != len(layout2):
            return 1.0  # 最大距离
        
        # 使用汉明距离
        distance = sum(1 for a, b in zip(layout1, layout2) if a != b)
        return distance / len(layout1) if len(layout1) > 0 else 0
    
    def _age_and_filter_population(self, population: List[Individual]) -> List[Individual]:
        """更新年龄并过滤过老的个体"""
        filtered = []
        for individual in population:
            individual.age += 1
            if individual.age <= self.max_age:
                filtered.append(individual)
            else:
                # 替换过老的个体
                new_layout = self._generate_constraint_aware_random_layout()
                new_individual = Individual(layout=new_layout)
                filtered.append(new_individual)
        
        return filtered