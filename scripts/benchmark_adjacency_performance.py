# scripts/benchmark_adjacency_performance.py

"""
相邻性奖励计算性能基准测试脚本。

此脚本用于比较优化前后的相邻性奖励计算性能，包括：
1. 执行时间对比
2. 内存使用对比
3. 缓存命中率统计
4. 准确性验证
5. 不同数据规模下的性能测试
"""

import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RLConfig, NetworkConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.env.adjacency_reward_calculator import create_adjacency_calculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)

class AdjacencyPerformanceBenchmark:
    """相邻性奖励计算性能基准测试类。"""
    
    def __init__(self, use_existing_data: bool = True):
        """
        初始化性能基准测试。
        
        Args:
            use_existing_data: 是否使用现有数据，False则生成模拟数据
        """
        self.use_existing_data = use_existing_data
        self.results = {}
        
        # 初始化配置和组件
        self._setup_test_environment()
        
    def _setup_test_environment(self):
        """设置测试环境。"""
        logger.info("设置性能基准测试环境...")
        
        # 创建配置
        self.config = RLConfig()
        self.network_config = NetworkConfig()
        
        if self.use_existing_data:
            # 使用现有数据
            self._load_existing_data()
        else:
            # 生成模拟数据
            self._generate_mock_data()
        
        logger.info(f"测试环境设置完成：{len(self.placeable_depts)}个科室")
    
    def _load_existing_data(self):
        """加载现有的项目数据。"""
        try:
            # 初始化缓存管理器
            self.cache_manager = CacheManager(self.config)
            self.cache_manager.initialize()
            
            # 初始化成本计算器
            self.cost_calculator = CostCalculator(self.config, self.cache_manager)
            
            # 初始化约束管理器
            self.constraint_manager = ConstraintManager(self.config, self.cache_manager)
            
            # 获取数据
            self.placeable_depts = self.cache_manager.placeable_departments
            self.travel_times_matrix = self.cache_manager.travel_times_matrix
            
            logger.info("成功加载现有项目数据")
            
        except Exception as e:
            logger.warning(f"加载现有数据失败：{e}，切换到模拟数据模式")
            self._generate_mock_data()
    
    def _generate_mock_data(self):
        """生成模拟测试数据。"""
        logger.info("生成模拟测试数据...")
        
        # 生成模拟科室列表
        dept_types = ['急诊科', '妇科', '儿科', '内科', '外科', '检验中心', '放射科', 'ICU', '手术室', '药房']
        self.placeable_depts = []
        for dept_type in dept_types:
            for i in range(3):  # 每种类型3个科室
                self.placeable_depts.append(f"{dept_type}_{i+1}")
        
        # 生成模拟行程时间矩阵
        n_depts = len(self.placeable_depts)
        np.random.seed(42)  # 确保可重复性
        
        # 生成对称的距离矩阵
        distances = np.random.exponential(scale=100, size=(n_depts, n_depts))
        distances = (distances + distances.T) / 2  # 确保对称性
        np.fill_diagonal(distances, 0)  # 对角线为0
        
        self.travel_times_matrix = pd.DataFrame(
            distances,
            index=self.placeable_depts,
            columns=self.placeable_depts
        )
        
        # 创建模拟的约束管理器
        class MockConstraintManager:
            def __init__(self):
                self.area_compatibility_matrix = np.ones((n_depts, n_depts), dtype=bool)
        
        self.constraint_manager = MockConstraintManager()
        
        logger.info(f"生成模拟数据完成：{n_depts}个科室")
    
    def create_test_layouts(self, num_layouts: int = 100, 
                           layout_sizes: List[int] = None) -> List[List[str]]:
        """
        创建测试用的布局组合。
        
        Args:
            num_layouts: 要生成的布局数量
            layout_sizes: 布局大小列表，None表示使用默认大小
            
        Returns:
            List[List[str]]: 测试布局列表
        """
        if layout_sizes is None:
            layout_sizes = [5, 10, 15, 20, min(25, len(self.placeable_depts))]
        
        test_layouts = []
        np.random.seed(42)  # 确保可重复性
        
        for size in layout_sizes:
            for _ in range(num_layouts // len(layout_sizes)):
                # 随机选择科室组合
                selected_depts = np.random.choice(
                    self.placeable_depts, 
                    size=min(size, len(self.placeable_depts)), 
                    replace=False
                ).tolist()
                test_layouts.append(selected_depts)
        
        logger.info(f"生成{len(test_layouts)}个测试布局")
        return test_layouts
    
    def benchmark_traditional_method(self, test_layouts: List[List[str]]) -> Dict:
        """基准测试传统相邻性计算方法。"""
        logger.info("开始基准测试传统计算方法...")
        
        # 设置使用传统方法
        self.config.ENABLE_ADJACENCY_OPTIMIZATION = False
        
        # 创建使用传统方法的环境
        try:
            # 创建最小可行的CacheManager
            class MinimalCacheManager:
                def __init__(self):
                    self.placeable_departments = test_layouts[0] if test_layouts else ['dept1', 'dept2']
                    self.placeable_slots = self.placeable_departments.copy()
                    self.placeable_nodes_df = pd.DataFrame({
                        'node_id': self.placeable_departments,
                        'area': np.random.uniform(50, 200, len(self.placeable_departments))
                    })
                    self.travel_times_matrix = self.travel_times_matrix if hasattr(self, 'travel_times_matrix') else pd.DataFrame()
            
            minimal_cm = MinimalCacheManager()
            minimal_cc = CostCalculator(self.config, minimal_cm) if hasattr(self, 'cost_calculator') else None
            
            env = LayoutEnv(
                config=self.config,
                cache_manager=minimal_cm,
                cost_calculator=minimal_cc,
                constraint_manager=self.constraint_manager
            )
        except Exception as e:
            logger.error(f"创建传统方法环境失败：{e}")
            return {'error': str(e)}
        
        # 性能测试
        start_time = time.time()
        tracemalloc.start()
        
        traditional_results = []
        for layout in test_layouts:
            try:
                # 模拟传统的相邻性计算
                layout_tuple = tuple(layout)
                reward = env._calculate_legacy_adjacency_reward(layout)
                traditional_results.append(reward)
            except Exception as e:
                logger.warning(f"传统方法计算失败：{e}")
                traditional_results.append(0.0)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        return {
            'execution_time': end_time - start_time,
            'memory_peak': peak / 1024 / 1024,  # MB
            'memory_current': current / 1024 / 1024,  # MB
            'results': traditional_results,
            'avg_time_per_calculation': (end_time - start_time) / len(test_layouts),
            'method': 'traditional'
        }
    
    def benchmark_optimized_method(self, test_layouts: List[List[str]]) -> Dict:
        """基准测试优化的相邻性计算方法。"""
        logger.info("开始基准测试优化计算方法...")
        
        # 设置使用优化方法
        self.config.ENABLE_ADJACENCY_OPTIMIZATION = True
        
        # 创建优化的相邻性计算器
        try:
            adjacency_calculator = create_adjacency_calculator(
                config=self.config,
                placeable_depts=self.placeable_depts,
                travel_times_matrix=self.travel_times_matrix,
                constraint_manager=self.constraint_manager
            )
        except Exception as e:
            logger.error(f"创建优化计算器失败：{e}")
            return {'error': str(e)}
        
        # 性能测试
        start_time = time.time()
        tracemalloc.start()
        
        optimized_results = []
        for layout in test_layouts:
            try:
                rewards_dict = adjacency_calculator.calculate_reward(
                    tuple(layout)
                )
                total_reward = rewards_dict.get('total_reward', 0.0)
                optimized_results.append(total_reward)
            except Exception as e:
                logger.warning(f"优化方法计算失败：{e}")
                optimized_results.append(0.0)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        # 获取性能指标
        performance_metrics = adjacency_calculator.get_performance_metrics()
        
        return {
            'execution_time': end_time - start_time,
            'memory_peak': peak / 1024 / 1024,  # MB
            'memory_current': current / 1024 / 1024,  # MB
            'results': optimized_results,
            'avg_time_per_calculation': (end_time - start_time) / len(test_layouts),
            'performance_metrics': performance_metrics,
            'method': 'optimized'
        }
    
    def run_comparative_benchmark(self, num_layouts: int = 100) -> Dict:
        """运行对比基准测试。"""
        logger.info(f"开始运行对比基准测试：{num_layouts}个测试布局")
        
        # 生成测试布局
        test_layouts = self.create_test_layouts(num_layouts)
        
        if not test_layouts:
            logger.error("无法生成测试布局")
            return {'error': '无法生成测试布局'}
        
        # 测试传统方法
        traditional_results = self.benchmark_traditional_method(test_layouts)
        
        # 测试优化方法
        optimized_results = self.benchmark_optimized_method(test_layouts)
        
        # 计算性能提升
        performance_improvement = self._calculate_performance_improvement(
            traditional_results, optimized_results
        )
        
        return {
            'traditional': traditional_results,
            'optimized': optimized_results,
            'improvement': performance_improvement,
            'test_layouts_count': len(test_layouts)
        }
    
    def _calculate_performance_improvement(self, traditional: Dict, optimized: Dict) -> Dict:
        """计算性能提升指标。"""
        if 'error' in traditional or 'error' in optimized:
            return {'error': '无法计算性能提升，存在计算错误'}
        
        time_improvement = traditional['execution_time'] / optimized['execution_time']
        memory_improvement = traditional['memory_peak'] / optimized['memory_peak']
        
        # 结果准确性验证
        accuracy_check = self._verify_result_accuracy(
            traditional.get('results', []),
            optimized.get('results', [])
        )
        
        return {
            'time_speedup': time_improvement,
            'memory_efficiency': memory_improvement,
            'time_saved_seconds': traditional['execution_time'] - optimized['execution_time'],
            'memory_saved_mb': traditional['memory_peak'] - optimized['memory_peak'],
            'accuracy_correlation': accuracy_check.get('correlation', 0.0),
            'accuracy_mean_diff': accuracy_check.get('mean_diff', 0.0),
            'cache_hit_rate': optimized.get('performance_metrics', {}).get('cache_hit_rate', 0.0)
        }
    
    def _verify_result_accuracy(self, traditional_results: List[float], 
                               optimized_results: List[float]) -> Dict:
        """验证结果准确性。"""
        if not traditional_results or not optimized_results:
            return {'correlation': 0.0, 'mean_diff': float('inf')}
        
        if len(traditional_results) != len(optimized_results):
            return {'correlation': 0.0, 'mean_diff': float('inf')}
        
        try:
            # 计算相关性
            correlation = np.corrcoef(traditional_results, optimized_results)[0, 1]
            
            # 计算平均差异
            mean_diff = np.mean(np.abs(np.array(traditional_results) - np.array(optimized_results)))
            
            return {
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'mean_diff': mean_diff
            }
        except Exception as e:
            logger.warning(f"准确性验证失败：{e}")
            return {'correlation': 0.0, 'mean_diff': float('inf')}
    
    def generate_performance_report(self, results: Dict, output_dir: Path = None):
        """生成性能测试报告。"""
        if output_dir is None:
            output_dir = Path("results/performance_benchmark")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_path = output_dir / "adjacency_performance_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PPO相邻性奖励计算性能优化报告\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试布局数量: {results.get('test_layouts_count', 0)}\n\n")
            
            if 'improvement' in results and 'error' not in results['improvement']:
                improvement = results['improvement']
                f.write("## 性能提升概要\n")
                f.write(f"- 执行时间提升: {improvement['time_speedup']:.2f}倍\n")
                f.write(f"- 内存效率提升: {improvement['memory_efficiency']:.2f}倍\n")
                f.write(f"- 时间节省: {improvement['time_saved_seconds']:.4f}秒\n")
                f.write(f"- 内存节省: {improvement['memory_saved_mb']:.2f}MB\n")
                f.write(f"- 缓存命中率: {improvement['cache_hit_rate']:.2%}\n")
                f.write(f"- 结果准确性相关性: {improvement['accuracy_correlation']:.4f}\n\n")
            
            # 详细结果
            if 'traditional' in results and 'error' not in results['traditional']:
                trad = results['traditional']
                f.write("## 传统方法详细结果\n")
                f.write(f"- 总执行时间: {trad['execution_time']:.4f}秒\n")
                f.write(f"- 平均单次计算时间: {trad['avg_time_per_calculation']:.6f}秒\n")
                f.write(f"- 内存峰值: {trad['memory_peak']:.2f}MB\n\n")
            
            if 'optimized' in results and 'error' not in results['optimized']:
                opt = results['optimized']
                f.write("## 优化方法详细结果\n")
                f.write(f"- 总执行时间: {opt['execution_time']:.4f}秒\n")
                f.write(f"- 平均单次计算时间: {opt['avg_time_per_calculation']:.6f}秒\n")
                f.write(f"- 内存峰值: {opt['memory_peak']:.2f}MB\n")
                
                if 'performance_metrics' in opt:
                    metrics = opt['performance_metrics']
                    f.write(f"- 缓存命中率: {metrics.get('cache_hit_rate', 0):.2%}\n")
                    f.write(f"- 总计算次数: {metrics.get('total_computations', 0)}\n")
                    f.write(f"- 空间相邻性计算时间: {metrics.get('spatial_computation_time', 0):.4f}秒\n")
                    f.write(f"- 功能相邻性计算时间: {metrics.get('functional_computation_time', 0):.4f}秒\n")
        
        # 生成图表
        self._generate_performance_charts(results, output_dir)
        
        logger.info(f"性能测试报告已生成：{report_path}")
    
    def _generate_performance_charts(self, results: Dict, output_dir: Path):
        """生成性能对比图表。"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.warning("matplotlib未安装，跳过图表生成")
            return
        
        if 'improvement' not in results or 'error' in results['improvement']:
            return
        
        improvement = results['improvement']
        
        # 创建性能对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 执行时间对比
        methods = ['传统方法', '优化方法']
        times = [
            results['traditional']['execution_time'],
            results['optimized']['execution_time']
        ]
        ax1.bar(methods, times, color=['#ff7f7f', '#7fbf7f'])
        ax1.set_ylabel('执行时间 (秒)')
        ax1.set_title('执行时间对比')
        for i, v in enumerate(times):
            ax1.text(i, v + max(times) * 0.01, f'{v:.4f}s', ha='center')
        
        # 内存使用对比
        memory = [
            results['traditional']['memory_peak'],
            results['optimized']['memory_peak']
        ]
        ax2.bar(methods, memory, color=['#ff7f7f', '#7fbf7f'])
        ax2.set_ylabel('内存峰值 (MB)')
        ax2.set_title('内存使用对比')
        for i, v in enumerate(memory):
            ax2.text(i, v + max(memory) * 0.01, f'{v:.2f}MB', ha='center')
        
        # 性能提升指标
        metrics = ['时间提升倍数', '内存效率提升']
        values = [improvement['time_speedup'], improvement['memory_efficiency']]
        ax3.bar(metrics, values, color=['#7f7fff', '#bf7fff'])
        ax3.set_ylabel('提升倍数')
        ax3.set_title('性能提升指标')
        for i, v in enumerate(values):
            ax3.text(i, v + max(values) * 0.01, f'{v:.2f}x', ha='center')
        
        # 缓存命中率
        cache_hit_rate = improvement.get('cache_hit_rate', 0)
        ax4.pie([cache_hit_rate, 1 - cache_hit_rate], 
                labels=['缓存命中', '缓存未命中'],
                colors=['#90EE90', '#FFB6C1'],
                autopct='%1.1f%%')
        ax4.set_title('缓存命中率')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("性能对比图表已生成")

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='相邻性奖励计算性能基准测试')
    parser.add_argument('--layouts', type=int, default=100, help='测试布局数量')
    parser.add_argument('--use-existing-data', action='store_true', help='使用现有项目数据')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建基准测试
    benchmark = AdjacencyPerformanceBenchmark(use_existing_data=args.use_existing_data)
    
    # 运行测试
    logger.info("开始相邻性奖励计算性能基准测试")
    results = benchmark.run_comparative_benchmark(num_layouts=args.layouts)
    
    # 生成报告
    output_dir = Path(args.output_dir) if args.output_dir else None
    benchmark.generate_performance_report(results, output_dir)
    
    # 输出摘要
    if 'improvement' in results and 'error' not in results['improvement']:
        improvement = results['improvement']
        logger.info("=" * 60)
        logger.info("性能基准测试完成")
        logger.info(f"执行时间提升: {improvement['time_speedup']:.2f}倍")
        logger.info(f"内存效率提升: {improvement['memory_efficiency']:.2f}倍")
        logger.info(f"缓存命中率: {improvement['cache_hit_rate']:.2%}")
        logger.info(f"结果准确性相关性: {improvement['accuracy_correlation']:.4f}")
        logger.info("=" * 60)
    else:
        logger.error("基准测试失败，请检查错误信息")

if __name__ == "__main__":
    main()