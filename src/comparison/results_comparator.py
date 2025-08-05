"""
结果对比分析器 - 提供详细的算法对比分析和可视化功能
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import seaborn as sns
from datetime import datetime

from src.algorithms.base_optimizer import OptimizationResult

logger = logging.getLogger(__name__)


class ResultsComparator:
    """
    结果对比分析器
    
    提供算法结果的详细对比分析，包括统计分析、可视化图表、
    性能指标计算和报告生成功能。
    """
    
    def __init__(self, results: Dict[str, OptimizationResult]):
        """
        初始化结果对比分析器
        
        Args:
            results: 算法名到结果的映射
        """
        self.results = results
        self.algorithm_names = list(results.keys())
        self.comparison_df = None
        
        logger.info(f"结果对比分析器初始化完成，包含 {len(results)} 个算法结果")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        生成详细的对比表格
        
        Returns:
            pd.DataFrame: 对比表格
        """
        comparison_data = []
        
        for algorithm_name, result in self.results.items():
            # 基础指标
            row = {
                'Algorithm': result.algorithm_name,
                'Best_Cost': result.best_cost,
                'Runtime_Sec': result.execution_time,
                'Iterations': result.iterations,
                'Iter_Per_Sec': result.iterations / result.execution_time if result.execution_time > 0 else 0,
                'Improvement_Rate': self._calculate_improvement_rate(result),
                'Convergence_Stability': self._calculate_convergence_stability(result),
                'Search_Efficiency': self._calculate_search_efficiency(result)
            }
            
            # 算法特定指标
            metrics = result.additional_metrics
            if algorithm_name == 'simulated_annealing':
                row.update({
                    'SA_Acceptance_Rate': metrics.get('acceptance_rate', 0) * 100,
                    'SA_Improvement_Count': metrics.get('improvement_count', 0),
                    'SA_Initial_Temp': metrics.get('initial_temperature', 0),
                    'SA_Final_Temp': metrics.get('current_temperature', 0)
                })
            elif algorithm_name == 'genetic_algorithm':
                row.update({
                    'GA_Final_Generation': metrics.get('final_generation', 0),
                    'GA_Population_Diversity': metrics.get('population_diversity', 0),
                    'GA_Convergence_Rate': metrics.get('convergence_rate', 0) * 100,
                    'GA_Stagnation_Count': metrics.get('stagnation_count', 0)
                })
            elif algorithm_name == 'ppo':
                row.update({
                    'PPO_Training_Steps': metrics.get('total_timesteps', 0),
                    'PPO_Num_Envs': metrics.get('num_envs', 0),
                    'PPO_LR_Schedule': metrics.get('learning_rate_schedule', 'N/A')
                })
            
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # 计算排名
        self.comparison_df['Cost_Rank'] = self.comparison_df['Best_Cost'].rank()
        self.comparison_df['Time_Rank'] = self.comparison_df['Runtime_Sec'].rank()
        self.comparison_df['Efficiency_Rank'] = self.comparison_df['Search_Efficiency'].rank(ascending=False)
        
        # 计算综合得分
        self.comparison_df['Composite_Score'] = self._calculate_composite_score()
        
        # 按最优成本排序
        self.comparison_df = self.comparison_df.sort_values('Best_Cost').reset_index(drop=True)
        
        return self.comparison_df
    
    def _calculate_improvement_rate(self, result: OptimizationResult) -> float:
        """计算改进率"""
        if not result.convergence_history or len(result.convergence_history) < 2:
            return 0.0
        
        initial_cost = result.convergence_history[0]
        final_cost = result.best_cost
        
        if initial_cost > 0:
            return ((initial_cost - final_cost) / initial_cost) * 100
        return 0.0
    
    def _calculate_convergence_stability(self, result: OptimizationResult) -> float:
        """计算收敛稳定性"""
        if not result.convergence_history or len(result.convergence_history) < 10:
            return 0.0
        
        # 取最后20%的迭代历史计算稳定性
        history = result.convergence_history
        tail_length = max(10, len(history) // 5)
        tail_history = history[-tail_length:]
        
        if len(tail_history) < 2:
            return 0.0
        
        # 使用变异系数衡量稳定性
        mean_cost = np.mean(tail_history)
        std_cost = np.std(tail_history)
        
        if mean_cost > 0:
            cv = std_cost / mean_cost
            stability = 1.0 / (1.0 + cv)  # 变异系数越小，稳定性越高
            return stability
        
        return 0.0
    
    def _calculate_search_efficiency(self, result: OptimizationResult) -> float:
        """计算搜索效率"""
        if result.execution_time <= 0 or not result.convergence_history:
            return 0.0
        
        improvement = self._calculate_improvement_rate(result)
        time_penalty = 1.0 / (1.0 + np.log10(result.execution_time + 1))
        
        efficiency = (improvement / 100.0) * time_penalty
        return efficiency
    
    def _calculate_composite_score(self) -> pd.Series:
        """计算综合得分"""
        if self.comparison_df is None:
            return pd.Series()
        
        # 归一化各项指标（越小越好的指标需要取倒数）
        cost_norm = 1.0 / (self.comparison_df['Best_Cost'] / self.comparison_df['Best_Cost'].min())
        time_norm = 1.0 / (self.comparison_df['Runtime_Sec'] / self.comparison_df['Runtime_Sec'].min())
        improvement_norm = self.comparison_df['Improvement_Rate'] / self.comparison_df['Improvement_Rate'].max()
        stability_norm = self.comparison_df['Convergence_Stability'] / self.comparison_df['Convergence_Stability'].max()
        efficiency_norm = self.comparison_df['Search_Efficiency'] / self.comparison_df['Search_Efficiency'].max()
        
        # 加权计算综合得分
        weights = {
            'cost': 0.4,      # 成本权重最高
            'time': 0.2,      # 时间权重
            'improvement': 0.2, # 改进率权重
            'stability': 0.1,  # 稳定性权重
            'efficiency': 0.1  # 效率权重
        }
        
        composite_score = (
            weights['cost'] * cost_norm +
            weights['time'] * time_norm + 
            weights['improvement'] * improvement_norm +
            weights['stability'] * stability_norm +
            weights['efficiency'] * efficiency_norm
        )
        
        return composite_score
    
    def create_comparison_plots(self, output_dir: str = "./results/plots"):
        """
        创建对比图表
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 1. 最优成本对比图
        self._plot_cost_comparison(output_path, timestamp)
        
        # 2. 执行时间对比图
        self._plot_time_comparison(output_path, timestamp)
        
        # 3. 收敛曲线图
        self._plot_convergence_curves(output_path, timestamp)
        
        # 4. 综合性能雷达图
        self._plot_performance_radar(output_path, timestamp)
        
        # 5. 算法特性热力图
        self._plot_algorithm_heatmap(output_path, timestamp)
        
        logger.info(f"所有对比图表已保存到: {output_path}")
    
    def _plot_cost_comparison(self, output_path: Path, timestamp: str):
        """绘制成本对比图"""
        plt.figure(figsize=(10, 6))
        
        costs = [result.best_cost for result in self.results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.algorithm_names)))
        
        bars = plt.bar(self.algorithm_names, costs, color=colors)
        plt.title('Algorithm Cost Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Optimal Cost', fontsize=12)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                    f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"cost_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_comparison(self, output_path: Path, timestamp: str):
        """绘制执行时间对比图"""
        plt.figure(figsize=(10, 6))
        
        times = [result.execution_time for result in self.results.values()]
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.algorithm_names)))
        
        bars = plt.bar(self.algorithm_names, times, color=colors)
        plt.title('Algorithm Runtime Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.xticks(rotation=45)
        plt.yscale('log')  # 使用对数刻度，因为PPO可能时间很长
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"time_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_curves(self, output_path: Path, timestamp: str):
        """绘制收敛曲线图"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.algorithm_names)))
        
        for i, (algorithm_name, result) in enumerate(self.results.items()):
            if result.convergence_history:
                history = result.convergence_history
                # 对于长度不同的历史，进行采样以便比较
                if len(history) > 1000:
                    indices = np.linspace(0, len(history)-1, 1000, dtype=int)
                    sampled_history = [history[idx] for idx in indices]
                    x_vals = np.linspace(0, len(history), 1000)
                else:
                    sampled_history = history
                    x_vals = range(len(history))
                
                plt.plot(x_vals, sampled_history, label=result.algorithm_name, 
                        color=colors[i], linewidth=2)
        
        plt.title('Algorithm Convergence Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"convergence_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, output_path: Path, timestamp: str):
        """绘制性能雷达图"""
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        # 检查DataFrame是否为空
        if self.comparison_df.empty:
            logger.warning("对比数据为空，跳过雷达图生成")
            return
        
        # 选择关键指标
        metrics = ['Best_Cost', 'Runtime_Sec', 'Improvement_Rate', 'Convergence_Stability', 'Search_Efficiency']
        
        # 数据归一化（转换为0-1范围，越大越好）
        normalized_data = {}
        # 使用DataFrame中实际的算法名称而不是self.algorithm_names
        actual_algorithm_names = self.comparison_df['Algorithm'].tolist()
        
        for algorithm_name in actual_algorithm_names:
            try:
                row = self.comparison_df[self.comparison_df['Algorithm'] == algorithm_name].iloc[0]
            except IndexError:
                logger.warning(f"未找到算法 {algorithm_name} 的数据，跳过")
                continue
            
            values = []
            try:
                # 成本：越小越好，取倒数后归一化
                cost_val = 1.0 / row['Best_Cost'] if pd.notna(row['Best_Cost']) and row['Best_Cost'] > 0 else 0
                values.append(cost_val)
                
                # 时间：越小越好，取倒数后归一化
                time_val = 1.0 / row['Runtime_Sec'] if pd.notna(row['Runtime_Sec']) and row['Runtime_Sec'] > 0 else 0
                values.append(time_val)
                
                # 其他指标：越大越好，直接使用
                improvement_rate = row['Improvement_Rate'] / 100.0 if pd.notna(row['Improvement_Rate']) else 0
                convergence = row['Convergence_Stability'] if pd.notna(row['Convergence_Stability']) else 0
                efficiency = row['Search_Efficiency'] if pd.notna(row['Search_Efficiency']) else 0
                
                values.extend([improvement_rate, convergence, efficiency])
                
                normalized_data[algorithm_name] = values
                
            except Exception as e:
                logger.warning(f"处理算法 {algorithm_name} 的雷达图数据时出错: {e}")
                continue
        
        # 检查是否有有效数据
        if not normalized_data:
            logger.warning("没有有效的雷达图数据，跳过雷达图生成")
            return
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(actual_algorithm_names)))
        
        for i, (algorithm_name, values) in enumerate(normalized_data.items()):
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=algorithm_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Cost', 'Time', 'Improvement', 'Stability', 'Efficiency'])
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / f"performance_radar_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_algorithm_heatmap(self, output_path: Path, timestamp: str):
        """绘制算法特性热力图"""
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        # 选择数值列进行热力图展示
        numeric_cols = self.comparison_df.select_dtypes(include=[np.number]).columns
        display_cols = [col for col in numeric_cols if not col.endswith('排名')]
        
        # 准备数据
        heatmap_data = self.comparison_df[['Algorithm'] + display_cols].set_index('Algorithm')
        
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        heatmap_data_scaled = pd.DataFrame(
            scaler.fit_transform(heatmap_data),
            index=heatmap_data.index,
            columns=heatmap_data.columns
        )
        
        # 创建热力图
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data_scaled.T, annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.2f', cbar_kws={'label': '标准化值'})
        plt.title('Algorithm Characteristics Heatmap (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Performance Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / f"algorithm_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, output_dir: str = "./results/reports") -> str:
        """
        生成详细的对比分析报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = output_path / f"algorithm_comparison_report_{timestamp}.md"
        
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 医院布局优化算法对比分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 执行摘要
            f.write("## 执行摘要\n\n")
            best_cost_algo = self.comparison_df.iloc[0]['Algorithm']
            best_cost = self.comparison_df.iloc[0]['Best_Cost']
            f.write(f"- **最优成本算法**: {best_cost_algo} (成本: {best_cost:.2f})\n")
            
            fastest_algo = self.comparison_df.loc[self.comparison_df['Runtime_Sec'].idxmin(), 'Algorithm']
            fastest_time = self.comparison_df['Runtime_Sec'].min()
            f.write(f"- **最快算法**: {fastest_algo} (时间: {fastest_time:.2f}秒)\n")
            
            best_efficiency_algo = self.comparison_df.loc[self.comparison_df['Search_Efficiency'].idxmax(), 'Algorithm']
            best_efficiency = self.comparison_df['Search_Efficiency'].max()
            f.write(f"- **最高效率算法**: {best_efficiency_algo} (效率: {best_efficiency:.4f})\n\n")
            
            # 详细对比表格
            f.write("## 详细对比结果\n\n")
            f.write(self.comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            # 算法分析
            f.write("## 算法详细分析\n\n")
            for algorithm_name, result in self.results.items():
                f.write(f"### {result.algorithm_name}\n\n")
                f.write(f"- **最优成本**: {result.best_cost:.2f}\n")
                f.write(f"- **执行时间**: {result.execution_time:.2f}秒\n")
                f.write(f"- **迭代次数**: {result.iterations}\n")
                
                # 算法特定分析
                metrics = result.additional_metrics
                if algorithm_name == 'simulated_annealing':
                    f.write(f"- **接受率**: {metrics.get('acceptance_rate', 0)*100:.1f}%\n")
                    f.write(f"- **改进次数**: {metrics.get('improvement_count', 0)}\n")
                elif algorithm_name == 'genetic_algorithm':
                    f.write(f"- **最终代数**: {metrics.get('final_generation', 0)}\n")
                    f.write(f"- **种群多样性**: {metrics.get('population_diversity', 0):.3f}\n")
                elif algorithm_name == 'ppo':
                    f.write(f"- **训练步数**: {metrics.get('total_timesteps', 0)}\n")
                    f.write(f"- **环境数量**: {metrics.get('num_envs', 0)}\n")
                
                f.write("\n")
            
            # 结论和建议
            f.write("## 结论和建议\n\n")
            f.write("基于上述分析结果，我们得出以下结论：\n\n")
            
            # 根据结果生成智能建议
            if len(self.results) >= 2:
                cost_range = self.comparison_df['Best_Cost'].max() - self.comparison_df['Best_Cost'].min()
                time_range = self.comparison_df['Runtime_Sec'].max() - self.comparison_df['Runtime_Sec'].min()
                
                if cost_range / self.comparison_df['Best_Cost'].min() > 0.1:
                    f.write("1. **成本差异显著**: 不同算法在优化质量上存在明显差异，建议优先使用成本最低的算法。\n")
                
                if time_range > 60:
                    f.write("2. **时间效率考虑**: 算法执行时间差异较大，在实际应用中需要平衡优化质量和时间成本。\n")
                
                f.write("3. **算法选择建议**:\n")
                f.write(f"   - 追求最优解质量: 推荐使用 **{best_cost_algo}**\n")
                f.write(f"   - 追求快速求解: 推荐使用 **{fastest_algo}**\n")
                f.write(f"   - 平衡质量和效率: 推荐使用 **{best_efficiency_algo}**\n")
        
        logger.info(f"详细对比报告已生成: {report_path}")
        return str(report_path)
    
    def export_layouts_comparison(self, output_dir: str = "./results/layouts"):
        """
        导出所有算法的最优布局进行可视化对比
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        layouts_data = {}
        for algorithm_name, result in self.results.items():
            layouts_data[algorithm_name] = {
                'layout': result.best_layout,
                'cost': result.best_cost,
                'algorithm': result.algorithm_name
            }
        
        # 保存为JSON文件
        layouts_path = output_path / f"best_layouts_{timestamp}.json"
        with open(layouts_path, 'w', encoding='utf-8') as f:
            json.dump(layouts_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"最优布局对比数据已保存到: {layouts_path}")
        
        return str(layouts_path)