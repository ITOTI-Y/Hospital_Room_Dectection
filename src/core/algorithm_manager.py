"""
算法管理器 - 统一管理和运行所有优化算法
"""

from src.rl_optimizer.utils.setup import setup_logger
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from src.algorithms.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.constraint_manager import ConstraintManager
from src.algorithms.ppo_optimizer import PPOOptimizer
from src.algorithms.simulated_annealing import SimulatedAnnealingOptimizer
from src.algorithms.genetic_algorithm import GeneticAlgorithmOptimizer
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.data.cache_manager import CacheManager
from src.config import RLConfig, NetworkConfig

logger = setup_logger(__name__)


class AlgorithmManager:
    """
    算法管理器
    
    统一管理所有优化算法的运行，提供单一接口来执行不同的优化算法，
    支持算法对比、并行执行和结果管理。
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: ConstraintManager,
                 config: RLConfig,
                 cache_manager: CacheManager):
        """
        初始化算法管理器
        
        Args:
            cost_calculator: 成本计算器
            constraint_manager: 约束管理器  
            config: RL配置
            cache_manager: 缓存管理器
        """
        self.cost_calculator = cost_calculator
        self.constraint_manager = constraint_manager
        self.config = config
        self.cache_manager = cache_manager
        
        # 算法注册表
        self.algorithm_registry = {
            'ppo': PPOOptimizer,
            'simulated_annealing': SimulatedAnnealingOptimizer,
            'genetic_algorithm': GeneticAlgorithmOptimizer
        }
        
        # 算法参数配置
        self.algorithm_configs = {
            'ppo': {
                'total_timesteps': config.TOTAL_TIMESTEPS,
            },
            'simulated_annealing': {
                'initial_temperature': config.SA_DEFAULT_INITIAL_TEMP,
                'final_temperature': config.SA_DEFAULT_FINAL_TEMP,
                'cooling_rate': config.SA_DEFAULT_COOLING_RATE,
                'temperature_length': config.SA_DEFAULT_TEMPERATURE_LENGTH,
                'max_iterations': config.SA_DEFAULT_MAX_ITERATIONS
            },
            'genetic_algorithm': {
                'population_size': config.GA_DEFAULT_POPULATION_SIZE,
                'elite_size': config.GA_DEFAULT_ELITE_SIZE,
                'mutation_rate': config.GA_DEFAULT_MUTATION_RATE,
                'crossover_rate': config.GA_DEFAULT_CROSSOVER_RATE,
                'tournament_size': config.GA_DEFAULT_TOURNAMENT_SIZE,
                'max_age': config.GA_DEFAULT_MAX_AGE,
                'max_iterations': config.GA_DEFAULT_MAX_ITERATIONS,
                'convergence_threshold': config.GA_DEFAULT_CONVERGENCE_THRESHOLD,
                'constraint_repair_strategy': config.GA_CONSTRAINT_REPAIR_STRATEGY,
                'adaptive_parameters': config.GA_ADAPTIVE_PARAMETERS
            }
        }
        
        # 结果存储
        self.results = {}
        
        logger.info("算法管理器初始化完成")
        logger.info(f"已注册算法: {list(self.algorithm_registry.keys())}")
    
    def run_single_algorithm(self, 
                           algorithm_name: str,
                           initial_layout: Optional[List[str]] = None,
                           custom_params: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        运行单个算法
        
        Args:
            algorithm_name: 算法名称
            initial_layout: 初始布局
            custom_params: 自定义参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        if algorithm_name not in self.algorithm_registry:
            raise ValueError(f"未知算法: {algorithm_name}. 可用算法: {list(self.algorithm_registry.keys())}")
        
        logger.info(f"开始运行算法: {algorithm_name}")
        
        # 生成原始布局（未经优化的基准）
        original_layout = self.constraint_manager.generate_original_layout()
        original_cost = self.cost_calculator.calculate_total_cost(original_layout)
        logger.info(f"原始布局成本: {original_cost:.2f}")
        
        # 合并参数
        params = self.algorithm_configs[algorithm_name].copy()
        if custom_params:
            params.update(custom_params)
        
        # 创建算法实例（传递自定义参数用于构造函数）
        optimizer = self._create_optimizer(algorithm_name, params)
        
        # 准备运行时参数（移除构造函数参数）
        runtime_params = params.copy()
        if algorithm_name == 'simulated_annealing':
            # 移除SA的构造函数参数，只保留运行时参数
            for key in ['initial_temperature', 'final_temperature', 'cooling_rate', 'temperature_length']:
                runtime_params.pop(key, None)
        elif algorithm_name == 'genetic_algorithm':
            # 移除GA的构造函数参数，只保留运行时参数
            for key in ['population_size', 'elite_size', 'mutation_rate', 'crossover_rate', 'tournament_size', 'max_age']:
                runtime_params.pop(key, None)
        
        # 运行优化
        start_time = time.time()
        try:
            result = optimizer.optimize(
                initial_layout=initial_layout,
                original_layout=original_layout,
                original_cost=original_cost,
                **runtime_params
            )
            result.execution_time = time.time() - start_time
            
            # 存储结果
            self.results[algorithm_name] = result
            
            logger.info(f"算法 {algorithm_name} 完成:")
            logger.info(f"  最优成本: {result.best_cost:.2f}")
            logger.info(f"  执行时间: {result.execution_time:.2f}s")
            logger.info(f"  迭代次数: {result.iterations}")
            
            return result
            
        except Exception as e:
            logger.error(f"算法 {algorithm_name} 执行失败: {e}", exc_info=True)
            raise
    
    def run_multiple_algorithms(self, 
                              algorithm_names: List[str],
                              initial_layout: Optional[List[str]] = None,
                              custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
                              parallel: bool = False) -> Dict[str, OptimizationResult]:
        """
        运行多个算法
        
        Args:
            algorithm_names: 算法名称列表
            initial_layout: 初始布局
            custom_params: 自定义参数字典，键为算法名
            parallel: 是否并行执行
            
        Returns:
            Dict[str, OptimizationResult]: 算法名到结果的映射
        """
        logger.info(f"开始运行多个算法: {algorithm_names}")
        logger.info(f"并行执行: {parallel}")
        
        # 生成共同的原始布局（所有算法使用相同的基准）
        original_layout = self.constraint_manager.generate_original_layout()
        original_cost = self.cost_calculator.calculate_total_cost(original_layout)
        logger.info(f"原始布局成本（所有算法共用）: {original_cost:.2f}")
        
        results = {}
        
        if parallel:
            # 并行执行（注意：PPO可能需要GPU资源，谨慎并行）
            results = self._run_algorithms_parallel(algorithm_names, initial_layout, custom_params, 
                                                   original_layout, original_cost)
        else:
            # 串行执行
            for algorithm_name in algorithm_names:
                params = self.algorithm_configs[algorithm_name].copy()
                if custom_params:
                    params.update(custom_params)
                try:
                    # 创建算法实例
                    optimizer = self._create_optimizer(algorithm_name, params)
                    
                    # 准备运行时参数
                    runtime_params = params.copy()

                    # 移除构造函数参数
                    if algorithm_name == 'simulated_annealing':
                        for key in ['initial_temperature', 'final_temperature', 'cooling_rate', 'temperature_length']:
                            runtime_params.pop(key, None)
                    elif algorithm_name == 'genetic_algorithm':
                        for key in ['population_size', 'elite_size', 'mutation_rate', 'crossover_rate', 'tournament_size', 'max_age']:
                            runtime_params.pop(key, None)
                    
                    # 运行优化
                    result = optimizer.optimize(
                        initial_layout=initial_layout,
                        original_layout=original_layout,
                        original_cost=original_cost,
                        **runtime_params
                    )
                    results[algorithm_name] = result
                except Exception as e:
                    logger.error(f"跳过算法 {algorithm_name}，原因: {e}")
        
        self.results.update(results)
        return results
    
    def _run_algorithms_parallel(self, 
                               algorithm_names: List[str],
                               initial_layout: Optional[List[str]],
                               custom_params: Optional[Dict[str, Dict[str, Any]]],
                               original_layout: List[str],
                               original_cost: float) -> Dict[str, OptimizationResult]:
        """
        并行运行算法（改进版：每个算法使用独立的资源实例）
        """
        results = {}
        
        def run_algorithm_independent(alg_name: str, init_layout: Optional[List[str]], 
                                     params: Dict[str, Any], orig_layout: List[str], orig_cost: float):
            """在独立的上下文中运行算法"""
            # 创建独立的优化器实例
            optimizer = self._create_optimizer(alg_name, params, create_independent=True)
            
            # 准备运行时参数
            runtime_params = params.copy()
            if alg_name == 'simulated_annealing':
                for key in ['initial_temperature', 'final_temperature', 'cooling_rate', 'temperature_length']:
                    runtime_params.pop(key, None)
            elif alg_name == 'genetic_algorithm':
                for key in ['population_size', 'elite_size', 'mutation_rate', 'crossover_rate', 'tournament_size', 'max_age']:
                    runtime_params.pop(key, None)
            
            # 运行优化
            start_time = time.time()
            result = optimizer.optimize(
                initial_layout=init_layout,
                original_layout=orig_layout,
                original_cost=orig_cost,
                **runtime_params
            )
            result.execution_time = time.time() - start_time
            
            return result
        
        with ThreadPoolExecutor(max_workers=min(len(algorithm_names), 3)) as executor:
            # 提交任务
            future_to_algorithm = {}
            for algorithm_name in algorithm_names:
                params = custom_params.get(algorithm_name, {}) if custom_params else {}
                future = executor.submit(run_algorithm_independent, algorithm_name, 
                                        initial_layout, params, original_layout, original_cost)
                future_to_algorithm[future] = algorithm_name
            
            # 收集结果
            for future in as_completed(future_to_algorithm):
                algorithm_name = future_to_algorithm[future]
                try:
                    result = future.result()
                    results[algorithm_name] = result
                except Exception as e:
                    logger.error(f"并行执行算法 {algorithm_name} 失败: {e}")
        
        return results
    
    def _create_optimizer(self, algorithm_name: str, custom_params: Optional[Dict[str, Any]] = None, 
                         create_independent: bool = False) -> BaseOptimizer:
        """
        创建优化器实例
        
        Args:
            algorithm_name: 算法名称
            custom_params: 自定义参数
            create_independent: 是否创建独立的calculator和manager实例（用于并发执行）
        """
        optimizer_class = self.algorithm_registry[algorithm_name]
        
        # 如果需要独立实例（并发执行），创建新的calculator和manager
        if create_independent:
            # 深拷贝以避免共享状态
            cost_calculator = copy.deepcopy(self.cost_calculator)
            constraint_manager = copy.deepcopy(self.constraint_manager)
        else:
            cost_calculator = self.cost_calculator
            constraint_manager = self.constraint_manager
        
        if algorithm_name == 'ppo':
            # 获取预训练模型路径（如果有）
            pretrained_model_path = None
            if custom_params and 'pretrained_model_path' in custom_params:
                pretrained_model_path = custom_params['pretrained_model_path']
            
            return optimizer_class(
                cost_calculator=cost_calculator,
                constraint_manager=constraint_manager,
                config=self.config,
                cache_manager=self.cache_manager,
                pretrained_model_path=pretrained_model_path
            )
        elif algorithm_name == 'simulated_annealing':
            # 获取SA特定的构造参数
            sa_params = {}
            if custom_params:
                if 'initial_temperature' in custom_params:
                    sa_params['initial_temperature'] = custom_params['initial_temperature']
                if 'final_temperature' in custom_params:
                    sa_params['final_temperature'] = custom_params['final_temperature']
                if 'cooling_rate' in custom_params:
                    sa_params['cooling_rate'] = custom_params['cooling_rate']
                if 'temperature_length' in custom_params:
                    sa_params['temperature_length'] = custom_params['temperature_length']
            
            return optimizer_class(
                cost_calculator=cost_calculator,
                constraint_manager=constraint_manager,
                **sa_params
            )
        elif algorithm_name == 'genetic_algorithm':
            # 获取GA特定的构造参数
            ga_params = {}
            if custom_params:
                if 'population_size' in custom_params:
                    ga_params['population_size'] = custom_params['population_size']
                if 'elite_size' in custom_params:
                    ga_params['elite_size'] = custom_params['elite_size']
                if 'mutation_rate' in custom_params:
                    ga_params['mutation_rate'] = custom_params['mutation_rate']
                if 'crossover_rate' in custom_params:
                    ga_params['crossover_rate'] = custom_params['crossover_rate']
                if 'tournament_size' in custom_params:
                    ga_params['tournament_size'] = custom_params['tournament_size']
                if 'max_age' in custom_params:
                    ga_params['max_age'] = custom_params['max_age']
            
            return optimizer_class(
                cost_calculator=cost_calculator,
                constraint_manager=constraint_manager,
                **ga_params
            )
        else:
            # 默认情况
            return optimizer_class(
                cost_calculator=self.cost_calculator,
                constraint_manager=self.constraint_manager
            )
    
    def get_algorithm_comparison(self) -> pd.DataFrame:
        """
        获取算法对比结果
        
        Returns:
            pd.DataFrame: 对比结果表格
        """
        if not self.results:
            logger.warning("没有算法执行结果可供对比")
            return pd.DataFrame()
        
        comparison_data = []
        
        for algorithm_name, result in self.results.items():
            row = {
                '算法名称': result.algorithm_name,
                '最优成本': result.best_cost,
                '执行时间(秒)': result.execution_time,
                '迭代次数': result.iterations,
                '收敛性': self._calculate_convergence_metric(result),
                '最终改进率(%)': self._calculate_improvement_rate(result)
            }
            
            # 添加算法特定指标
            if algorithm_name == 'simulated_annealing':
                metrics = result.additional_metrics
                row['接受率(%)'] = metrics.get('acceptance_rate', 0) * 100
                row['改进次数'] = metrics.get('improvement_count', 0)
            elif algorithm_name == 'genetic_algorithm':
                metrics = result.additional_metrics
                row['最终代数'] = metrics.get('final_generation', 0)
                row['种群多样性'] = metrics.get('population_diversity', 0)
                row['收敛率(%)'] = metrics.get('convergence_rate', 0) * 100
            elif algorithm_name == 'ppo':
                metrics = result.additional_metrics
                row['训练步数'] = metrics.get('total_timesteps', 0)
                row['环境数量'] = metrics.get('num_envs', 0)
        
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 按最优成本排序
        df = df.sort_values('最优成本')
        df = df.reset_index(drop=True)
        
        return df
    
    def _calculate_convergence_metric(self, result: OptimizationResult) -> float:
        """计算收敛性指标"""
        if not result.convergence_history or len(result.convergence_history) < 2:
            return 0.0
        
        # 计算后期收敛稳定性
        history = result.convergence_history
        if len(history) >= 100:
            # 取最后100次迭代的标准差作为收敛性指标
            recent_history = history[-100:]
            std_dev = pd.Series(recent_history).std()
            return 1.0 / (1.0 + std_dev)  # 标准差越小，收敛性越好
        else:
            # 对于较短的历史，计算总体收敛趋势
            initial_cost = history[0]
            final_cost = history[-1]
            if initial_cost > 0:
                return (initial_cost - final_cost) / initial_cost
            return 0.0
    
    def _calculate_improvement_rate(self, result: OptimizationResult) -> float:
        """计算改进率"""
        if not result.convergence_history or len(result.convergence_history) < 2:
            return 0.0
        
        initial_cost = result.original_cost
        final_cost = result.best_cost
        
        if initial_cost > 0:
            return ((initial_cost - final_cost) / initial_cost) * 100
        return 0.0
    
    def save_results(self, output_dir: str = "./results/comparison"):
        """
        保存算法对比结果
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存对比表格
        comparison_df = self.get_algorithm_comparison()
        if not comparison_df.empty:
            csv_path = output_path / f"algorithm_comparison_{timestamp}.csv"
            comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"算法对比结果已保存到: {csv_path}")
        
        # 保存详细结果
        for algorithm_name, result in self.results.items():
            result_dict = {
                'algorithm_name': result.algorithm_name,
                'best_cost': result.best_cost,
                'execution_time': result.execution_time,
                'iterations': result.iterations,
                'best_layout': result.best_layout,
                'original_layout': result.original_layout,  # 添加原始布局
                'original_cost': result.original_cost,      # 添加原始成本
                'convergence_history': result.convergence_history,
                'additional_metrics': result.additional_metrics
            }
            
            # 计算改进率
            if result.original_cost is not None and result.original_cost > 0:
                result_dict['improvement'] = ((result.original_cost - result.best_cost) / result.original_cost) * 100
            
            import json
            json_path = output_path / f"{algorithm_name}_result_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"算法 {algorithm_name} 详细结果已保存到: {json_path}")
    
    def get_best_result(self) -> Optional[OptimizationResult]:
        """
        获取最佳结果
        
        Returns:
            Optional[OptimizationResult]: 最佳优化结果
        """
        if not self.results:
            return None
        
        best_result = min(self.results.values(), key=lambda x: x.best_cost)
        return best_result
    
    def clear_results(self):
        """清除所有结果"""
        self.results.clear()
        logger.info("已清除所有算法结果")
    
    def get_algorithm_names(self) -> List[str]:
        """获取所有可用的算法名称"""
        return list(self.algorithm_registry.keys())
    
    def register_algorithm(self, name: str, optimizer_class: Type[BaseOptimizer], config: Dict[str, Any]):
        """
        注册新算法
        
        Args:
            name: 算法名称
            optimizer_class: 优化器类
            config: 算法配置
        """
        self.algorithm_registry[name] = optimizer_class
        self.algorithm_configs[name] = config
        logger.info(f"已注册新算法: {name}")
    
    def update_algorithm_config(self, algorithm_name: str, config: Dict[str, Any]):
        """
        更新算法配置
        
        Args:
            algorithm_name: 算法名称
            config: 新配置
        """
        if algorithm_name in self.algorithm_configs:
            self.algorithm_configs[algorithm_name].update(config)
            logger.info(f"已更新算法 {algorithm_name} 的配置")
        else:
            logger.warning(f"算法 {algorithm_name} 不存在，无法更新配置")