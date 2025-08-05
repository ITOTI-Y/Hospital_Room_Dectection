"""
医院布局优化系统主入口

统一的命令行接口，支持网络生成、算法优化和结果对比分析。
整合了所有功能模块，提供完整的医院布局优化解决方案。
"""

import argparse
import logging
import sys
import pathlib
from typing import List, Optional, Dict, Any
import pandas as pd

# 导入核心模块
from src.config import NetworkConfig, RLConfig, COLOR_MAP
from src.core.network_generator import NetworkGenerator
from src.core.algorithm_manager import AlgorithmManager
from src.comparison.results_comparator import ResultsComparator

# 导入优化组件
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.algorithms.constraint_manager import ConstraintManager

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, log_file: Optional[pathlib.Path] = None):
    """配置日志系统"""
    root_logger = logging.getLogger()
    
    if root_logger.hasHandlers():
        return
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    root_logger.setLevel(level)


class HospitalLayoutOptimizer:
    """医院布局优化系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.network_config = NetworkConfig(color_map_data=COLOR_MAP)
        self.rl_config = RLConfig()
        self.network_generator = None
        self.algorithm_manager = None
        
        logger.info("医院布局优化系统初始化完成")
    
    def run_network_generation(self, 
                             image_dir: str = "./data/label/",
                             visualization_filename: str = "hospital_network_3d.html",
                             travel_times_filename: str = "hospital_travel_times.csv") -> bool:
        """
        运行网络生成
        
        Args:
            image_dir: 楼层标注图像目录
            visualization_filename: 可视化输出文件名
            travel_times_filename: 行程时间输出文件名
            
        Returns:
            bool: 是否成功
        """
        logger.info("=== 开始网络生成阶段 ===")
        
        self.network_generator = NetworkGenerator(self.network_config)
        
        success = self.network_generator.run_complete_generation(
            image_dir=image_dir,
            visualization_filename=visualization_filename,
            travel_times_filename=travel_times_filename
        )
        
        if success:
            network_info = self.network_generator.get_network_info()
            logger.info("网络生成完成，统计信息:")
            for key, value in network_info.items():
                logger.info(f"  {key}: {value}")
        
        return success
    
    def run_single_algorithm(self, 
                           algorithm_name: str,
                           travel_times_file: str = None,
                           **kwargs) -> bool:
        """
        运行单个优化算法
        
        Args:
            algorithm_name: 算法名称
            travel_times_file: 行程时间文件路径
            **kwargs: 算法特定参数
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"=== 开始运行算法: {algorithm_name} ===")
        
        if travel_times_file is None:
            travel_times_file = self.rl_config.TRAVEL_TIMES_CSV
        
        # 初始化算法管理器
        if not self._initialize_algorithm_manager(travel_times_file):
            return False
        
        try:
            result = self.algorithm_manager.run_single_algorithm(
                algorithm_name=algorithm_name,
                custom_params=kwargs
            )
            
            logger.info(f"算法 {algorithm_name} 执行成功:")
            logger.info(f"  最优成本: {result.best_cost:.2f}")
            logger.info(f"  执行时间: {result.execution_time:.2f}秒")
            logger.info(f"  迭代次数: {result.iterations}")
            
            # 保存结果
            self.algorithm_manager.save_results()
            
            return True
            
        except Exception as e:
            logger.error(f"算法执行失败: {e}", exc_info=True)
            return False
    
    def run_algorithm_comparison(self, 
                               algorithm_names: List[str],
                               travel_times_file: str = None,
                               parallel: bool = False,
                               generate_plots: bool = True,
                               generate_report: bool = True) -> bool:
        """
        运行算法对比分析
        
        Args:
            algorithm_names: 算法名称列表
            travel_times_file: 行程时间文件路径
            parallel: 是否并行执行
            generate_plots: 是否生成图表
            generate_report: 是否生成报告
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"=== 开始算法对比分析: {algorithm_names} ===")
        
        if travel_times_file is None:
            travel_times_file = self.rl_config.TRAVEL_TIMES_CSV
        
        # 初始化算法管理器
        if not self._initialize_algorithm_manager(travel_times_file):
            return False
        
        try:
            # 运行多个算法
            results = self.algorithm_manager.run_multiple_algorithms(
                algorithm_names=algorithm_names,
                parallel=parallel
            )
            
            if not results:
                logger.error("没有算法成功执行")
                return False
            
            logger.info(f"成功执行 {len(results)} 个算法")
            
            # 生成对比表格
            comparison_df = self.algorithm_manager.get_algorithm_comparison()
            logger.info("算法对比结果:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            # 创建结果对比分析器
            comparator = ResultsComparator(results)
            
            # 生成详细对比表格
            detailed_df = comparator.generate_comparison_table()
            
            # 生成图表
            if generate_plots:
                comparator.create_comparison_plots()
                logger.info("对比图表已生成")
            
            # 生成报告
            if generate_report:
                report_path = comparator.generate_detailed_report()
                logger.info(f"详细报告已生成: {report_path}")
            
            # 导出布局对比
            layouts_path = comparator.export_layouts_comparison()
            logger.info(f"最优布局对比已导出: {layouts_path}")
            
            # 保存结果
            self.algorithm_manager.save_results()
            
            # 输出最佳结果
            best_result = self.algorithm_manager.get_best_result()
            if best_result:
                logger.info(f"整体最佳结果来自: {best_result.algorithm_name}")
                logger.info(f"最优成本: {best_result.best_cost:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"算法对比分析失败: {e}", exc_info=True)
            return False
    
    def _initialize_algorithm_manager(self, travel_times_file: str) -> bool:
        """初始化算法管理器"""
        try:
            # 检查行程时间文件
            travel_times_path = pathlib.Path(travel_times_file)
            if not travel_times_path.exists():
                logger.error(f"行程时间文件不存在: {travel_times_file}")
                logger.error("请先运行网络生成阶段：python main.py --mode network")
                return False
            
            logger.info("正在初始化优化组件...")
            
            # 初始化缓存管理器
            cache_manager = CacheManager(self.rl_config)
            logger.info("缓存管理器初始化完成")
            
            # 初始化成本计算器
            cost_calculator = CostCalculator(
                config=self.rl_config,
                resolved_pathways=cache_manager.resolved_pathways,
                travel_times=cache_manager.travel_times_matrix,
                placeable_slots=cache_manager.placeable_slots,
                placeable_departments=cache_manager.placeable_departments
            )
            logger.info("成本计算器初始化完成")
            
            # 初始化约束管理器
            constraint_manager = ConstraintManager(
                placeable_slots=cache_manager.placeable_slots,
                placeable_departments=cache_manager.placeable_departments,
                travel_times=cache_manager.travel_times_matrix
            )
            logger.info("约束管理器初始化完成")
            
            # 初始化算法管理器
            self.algorithm_manager = AlgorithmManager(
                cost_calculator=cost_calculator,
                constraint_manager=constraint_manager,
                config=self.rl_config,
                cache_manager=cache_manager
            )
            logger.info("算法管理器初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"算法管理器初始化失败: {e}", exc_info=True)
            return False
    
    def visualize_results(self, results_file: str):
        """可视化算法结果"""
        logger.info(f"=== 开始结果可视化: {results_file} ===")
        # 这里可以添加结果可视化逻辑
        logger.info("结果可视化功能待实现")


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="医院布局优化系统 - 整合网络生成、算法优化和结果对比分析",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['network', 'optimize', 'compare', 'visualize'],
        required=True,
        help="运行模式:\n"
             "  network    - 生成医院网络和行程时间矩阵\n"
             "  optimize   - 运行单个优化算法\n"
             "  compare    - 运行多个算法进行对比分析\n"
             "  visualize  - 可视化算法结果"
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['ppo', 'simulated_annealing', 'genetic_algorithm'],
        help="优化算法名称 (用于 optimize 模式)"
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        help="算法列表，用逗号分隔 (用于 compare 模式)\n"
             "例如: ppo,simulated_annealing,genetic_algorithm"
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        default="./data/label/",
        help="楼层标注图像目录 (默认: ./data/label/)"
    )
    
    parser.add_argument(
        '--travel-times-file',
        type=str,
        default=None,
        help="行程时间文件路径 (默认: 由config.py中的RLConfig.TRAVEL_TIMES_CSV指定)"
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="并行执行多个算法 (用于 compare 模式)"
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="不生成对比图表"
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help="不生成详细报告"
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        help="结果文件路径 (用于 visualize 模式)"
    )
    
    # 算法特定参数
    parser.add_argument(
        '--max-iterations',
        type=int,
        help="最大迭代次数"
    )
    
    parser.add_argument(
        '--population-size',
        type=int,
        help="遗传算法种群大小"
    )
    
    parser.add_argument(
        '--initial-temperature',
        type=float,
        help="模拟退火初始温度"
    )
    
    parser.add_argument(
        '--total-timesteps',
        type=int,
        help="PPO总训练步数"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="详细输出"
    )
    
    return parser


def main():
    """主函数"""
    # 创建参数解析器
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = pathlib.Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=log_level, log_file=log_dir / "hospital_optimizer.log")
    
    logger.info("=== 医院布局优化系统启动 ===")
    logger.info(f"运行模式: {args.mode}")
    
    # 创建系统实例
    optimizer = HospitalLayoutOptimizer()
    
    success = False
    
    try:
        if args.mode == 'network':
            # 网络生成模式
            success = optimizer.run_network_generation(
                image_dir=args.image_dir
            )
            
        elif args.mode == 'optimize':
            # 单算法优化模式
            if not args.algorithm:
                logger.error("optimize 模式需要指定 --algorithm 参数")
                sys.exit(1)
            
            # 构建算法参数
            algorithm_params = {}
            if args.max_iterations:
                algorithm_params['max_iterations'] = args.max_iterations
            if args.population_size:
                algorithm_params['population_size'] = args.population_size
            if args.initial_temperature:
                algorithm_params['initial_temperature'] = args.initial_temperature
            if args.total_timesteps:
                algorithm_params['total_timesteps'] = args.total_timesteps
            
            success = optimizer.run_single_algorithm(
                algorithm_name=args.algorithm,
                travel_times_file=args.travel_times_file,
                **algorithm_params
            )
            
        elif args.mode == 'compare':
            # 算法对比模式
            if not args.algorithms:
                logger.error("compare 模式需要指定 --algorithms 参数")
                sys.exit(1)
            
            algorithm_names = [name.strip() for name in args.algorithms.split(',')]
            
            success = optimizer.run_algorithm_comparison(
                algorithm_names=algorithm_names,
                travel_times_file=args.travel_times_file,
                parallel=args.parallel,
                generate_plots=not args.no_plots,
                generate_report=not args.no_report
            )
            
        elif args.mode == 'visualize':
            # 结果可视化模式
            if not args.results_file:
                logger.error("visualize 模式需要指定 --results-file 参数")
                sys.exit(1)
            
            optimizer.visualize_results(args.results_file)
            success = True
        
        if success:
            logger.info("=== 系统执行成功完成 ===")
        else:
            logger.error("=== 系统执行失败 ===")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"系统执行异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()