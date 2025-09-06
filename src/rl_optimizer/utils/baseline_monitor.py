"""
BaselineMonitor - 动态基线监控器

监控和记录动态基线的变化，提供实时统计和可视化支持。
"""

import time
from src.rl_optimizer.utils.setup import setup_logger
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from src.config import RLConfig
from src.rl_optimizer.utils.shared_state_manager import SharedStateManager

logger = setup_logger(__name__)


@dataclass
class BaselineSnapshot:
    """基线快照数据类"""
    timestamp: float
    episode_count: int
    time_cost_baseline: Optional[float]
    adjacency_baseline: Optional[float]
    area_match_baseline: Optional[float]
    time_cost_std: Optional[float]
    adjacency_std: Optional[float]
    area_match_std: Optional[float]
    warmup_complete: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class BaselineMonitor:
    """
    动态基线监控器
    
    负责监控动态基线的变化，记录历史数据，并提供可视化功能。
    """
    
    def __init__(self, config: RLConfig, shared_state_manager: SharedStateManager, 
                 log_dir: Optional[Path] = None):
        """
        初始化基线监控器
        
        Args:
            config: RL配置对象
            shared_state_manager: 共享状态管理器
            log_dir: 日志目录，默认使用config中的LOG_PATH
        """
        self.config = config
        self.shared_state = shared_state_manager
        self.log_dir = log_dir or config.LOG_PATH
        
        # 创建监控输出目录
        self.monitor_dir = self.log_dir / 'baseline_monitor'
        self.monitor_dir.mkdir(exist_ok=True)
        
        # 历史数据存储
        self.snapshots: List[BaselineSnapshot] = []
        self.last_snapshot_time = 0.0
        self.snapshot_interval = 60.0  # 每60秒记录一次快照
        
        # 统计文件路径
        self.stats_file = self.monitor_dir / 'baseline_history.json'
        self.plots_dir = self.monitor_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"基线监控器初始化完成，监控目录: {self.monitor_dir}")
    
    def take_snapshot(self) -> BaselineSnapshot:
        """
        记录当前状态的快照
        
        Returns:
            BaselineSnapshot: 当前状态快照
        """
        current_time = time.time()
        
        # 从共享状态管理器获取当前状态
        stats = self.shared_state.get_statistics()
        
        snapshot = BaselineSnapshot(
            timestamp=current_time,
            episode_count=stats['global_episode_count'],
            time_cost_baseline=self.shared_state.get_time_cost_baseline(),
            adjacency_baseline=self.shared_state.get_adjacency_baseline(),
            area_match_baseline=self.shared_state.get_area_match_baseline(),
            time_cost_std=self.shared_state.get_ema_std('time_cost_std'),
            adjacency_std=self.shared_state.get_ema_std('adjacency_reward_std'),
            area_match_std=self.shared_state.get_ema_std('area_match_std'),
            warmup_complete=stats['warmup_complete']
        )
        
        return snapshot
    
    def record_snapshot(self, force: bool = False) -> bool:
        """
        记录快照（如果满足时间间隔或强制记录）
        
        Args:
            force: 是否强制记录，忽略时间间隔
            
        Returns:
            bool: 是否成功记录了快照
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_snapshot_time) < self.snapshot_interval:
            return False
        
        try:
            snapshot = self.take_snapshot()
            self.snapshots.append(snapshot)
            self.last_snapshot_time = current_time
            
            # 定期保存到文件
            if len(self.snapshots) % 10 == 0:  # 每10个快照保存一次
                self.save_history()
            
            return True
            
        except Exception as e:
            logger.error(f"记录基线快照时发生错误: {e}")
            return False
    
    def save_history(self) -> None:
        """将快照历史保存到文件"""
        try:
            history_data = {
                'config': {
                    'ema_alpha': self.config.EMA_ALPHA,
                    'warmup_episodes': self.config.BASELINE_WARMUP_EPISODES,
                    'enable_dynamic_baseline': self.config.ENABLE_DYNAMIC_BASELINE,
                    'enable_relative_improvement': self.config.ENABLE_RELATIVE_IMPROVEMENT_REWARD
                },
                'snapshots': [snapshot.to_dict() for snapshot in self.snapshots],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.stats_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.debug(f"基线历史数据已保存: {len(self.snapshots)}个快照")
            
        except Exception as e:
            logger.error(f"保存基线历史数据时发生错误: {e}")
    
    def load_history(self) -> bool:
        """从文件加载快照历史"""
        try:
            if not self.stats_file.exists():
                return False
            
            with open(self.stats_file, 'r') as f:
                history_data = json.load(f)
            
            # 重建快照列表
            self.snapshots = []
            for snapshot_dict in history_data.get('snapshots', []):
                snapshot = BaselineSnapshot(**snapshot_dict)
                self.snapshots.append(snapshot)
            
            logger.info(f"已加载 {len(self.snapshots)} 个历史快照")
            return True
            
        except Exception as e:
            logger.error(f"加载基线历史数据时发生错误: {e}")
            return False
    
    def get_baseline_trends(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        获取基线变化趋势
        
        Returns:
            dict: 包含各基线的时间序列数据
        """
        if not self.snapshots:
            return {}
        
        trends = {
            'time_cost': [],
            'adjacency': [],
            'area_match': [],
            'time_cost_std': [],
            'adjacency_std': [],
            'area_match_std': []
        }
        
        for snapshot in self.snapshots:
            timestamp = snapshot.timestamp
            
            if snapshot.time_cost_baseline is not None:
                trends['time_cost'].append((timestamp, snapshot.time_cost_baseline))
            if snapshot.adjacency_baseline is not None:
                trends['adjacency'].append((timestamp, snapshot.adjacency_baseline))
            if snapshot.area_match_baseline is not None:
                trends['area_match'].append((timestamp, snapshot.area_match_baseline))
            if snapshot.time_cost_std is not None:
                trends['time_cost_std'].append((timestamp, snapshot.time_cost_std))
            if snapshot.adjacency_std is not None:
                trends['adjacency_std'].append((timestamp, snapshot.adjacency_std))
            if snapshot.area_match_std is not None:
                trends['area_match_std'].append((timestamp, snapshot.area_match_std))
        
        return trends
    
    def plot_baseline_evolution(self, save_plots: bool = True) -> Optional[str]:
        """
        绘制基线演变图表
        
        Args:
            save_plots: 是否保存图表到文件
            
        Returns:
            str: 保存的图表文件路径，如果未保存则返回None
        """
        if not self.snapshots:
            logger.warning("没有快照数据，无法生成图表")
            return None
        
        try:
            trends = self.get_baseline_trends()
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('动态基线演变趋势', fontsize=16)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 绘制时间成本基线
            if trends['time_cost']:
                times, values = zip(*trends['time_cost'])
                times_rel = [(t - times[0]) / 3600 for t in times]  # 转换为小时
                axes[0, 0].plot(times_rel, values, 'b-', linewidth=2, label='时间成本基线')
                axes[0, 0].set_title('时间成本基线演变')
                axes[0, 0].set_xlabel('训练时间 (小时)')
                axes[0, 0].set_ylabel('时间成本 (秒)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            # 绘制相邻性基线
            if trends['adjacency']:
                times, values = zip(*trends['adjacency'])
                times_rel = [(t - times[0]) / 3600 for t in times]
                axes[0, 1].plot(times_rel, values, 'g-', linewidth=2, label='相邻性奖励基线')
                axes[0, 1].set_title('相邻性奖励基线演变')
                axes[0, 1].set_xlabel('训练时间 (小时)')
                axes[0, 1].set_ylabel('相邻性奖励')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # 绘制面积匹配基线
            if trends['area_match']:
                times, values = zip(*trends['area_match'])
                times_rel = [(t - times[0]) / 3600 for t in times]
                axes[1, 0].plot(times_rel, values, 'r-', linewidth=2, label='面积匹配基线')
                axes[1, 0].set_title('面积匹配基线演变')
                axes[1, 0].set_xlabel('训练时间 (小时)')
                axes[1, 0].set_ylabel('面积匹配分数')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # 绘制标准差演变
            ax_std = axes[1, 1]
            if trends['time_cost_std']:
                times, values = zip(*trends['time_cost_std'])
                times_rel = [(t - times[0]) / 3600 for t in times]
                ax_std.plot(times_rel, values, 'b--', linewidth=2, label='时间成本标准差')
            if trends['adjacency_std']:
                times, values = zip(*trends['adjacency_std'])
                times_rel = [(t - times[0]) / 3600 for t in times]
                ax_std.plot(times_rel, values, 'g--', linewidth=2, label='相邻性标准差')
            if trends['area_match_std']:
                times, values = zip(*trends['area_match_std'])
                times_rel = [(t - times[0]) / 3600 for t in times]
                ax_std.plot(times_rel, values, 'r--', linewidth=2, label='面积匹配标准差')
            
            ax_std.set_title('标准差演变')
            ax_std.set_xlabel('训练时间 (小时)')
            ax_std.set_ylabel('标准差')
            ax_std.grid(True, alpha=0.3)
            ax_std.legend()
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_file = self.plots_dir / f'baseline_evolution_{timestamp}.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"基线演变图表已保存: {plot_file}")
                return str(plot_file)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"绘制基线演变图表时发生错误: {e}")
            return None
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        生成基线监控摘要报告
        
        Returns:
            dict: 摘要报告数据
        """
        if not self.snapshots:
            return {'error': '没有可用的快照数据'}
        
        latest_snapshot = self.snapshots[-1]
        first_snapshot = self.snapshots[0]
        
        # 计算训练持续时间
        duration_hours = (latest_snapshot.timestamp - first_snapshot.timestamp) / 3600
        
        report = {
            'monitoring_period': {
                'start_time': datetime.fromtimestamp(first_snapshot.timestamp).isoformat(),
                'end_time': datetime.fromtimestamp(latest_snapshot.timestamp).isoformat(),
                'duration_hours': duration_hours,
                'total_snapshots': len(self.snapshots)
            },
            'training_progress': {
                'total_episodes': latest_snapshot.episode_count,
                'warmup_complete': latest_snapshot.warmup_complete,
                'warmup_episodes': self.config.BASELINE_WARMUP_EPISODES
            },
            'latest_baselines': {
                'time_cost': latest_snapshot.time_cost_baseline,
                'adjacency': latest_snapshot.adjacency_baseline,
                'area_match': latest_snapshot.area_match_baseline
            },
            'latest_std_devs': {
                'time_cost': latest_snapshot.time_cost_std,
                'adjacency': latest_snapshot.adjacency_std,
                'area_match': latest_snapshot.area_match_std
            },
            'baseline_stability': self._analyze_baseline_stability(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _analyze_baseline_stability(self) -> Dict[str, Any]:
        """分析基线稳定性"""
        if len(self.snapshots) < 10:
            return {'status': 'insufficient_data', 'message': '数据不足，需要更多快照'}
        
        # 取最近的快照分析稳定性
        recent_snapshots = self.snapshots[-10:]
        
        def coefficient_of_variation(values):
            """计算变异系数"""
            if not values or len(values) < 2:
                return None
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            if mean_val == 0:
                return None
            return np.std(values_array) / abs(mean_val)
        
        # 提取最近的基线值
        time_costs = [s.time_cost_baseline for s in recent_snapshots if s.time_cost_baseline is not None]
        adjacencies = [s.adjacency_baseline for s in recent_snapshots if s.adjacency_baseline is not None]
        area_matches = [s.area_match_baseline for s in recent_snapshots if s.area_match_baseline is not None]
        
        stability = {
            'time_cost_cv': coefficient_of_variation(time_costs),
            'adjacency_cv': coefficient_of_variation(adjacencies),
            'area_match_cv': coefficient_of_variation(area_matches),
            'overall_stability': 'unknown'
        }
        
        # 评估整体稳定性
        cvs = [cv for cv in [stability['time_cost_cv'], stability['adjacency_cv'], stability['area_match_cv']] if cv is not None]
        if cvs:
            avg_cv = np.mean(cvs)
            if avg_cv < 0.1:
                stability['overall_stability'] = 'stable'
            elif avg_cv < 0.3:
                stability['overall_stability'] = 'moderate'
            else:
                stability['overall_stability'] = 'unstable'
        
        return stability
    
    def _generate_recommendations(self) -> List[str]:
        """生成基于监控数据的建议"""
        recommendations = []
        
        if not self.snapshots:
            return ["需要更多训练数据进行分析"]
        
        latest = self.snapshots[-1]
        
        # 检查预热状态
        if not latest.warmup_complete:
            recommendations.append("基线仍在预热期，建议继续训练直到预热完成")
        
        # 检查基线数据完整性
        if latest.time_cost_baseline is None:
            recommendations.append("时间成本基线未初始化，可能需要调整EMA参数")
        
        if latest.adjacency_baseline is None and self.config.ENABLE_ADJACENCY_REWARD:
            recommendations.append("相邻性奖励基线未初始化，检查相邻性奖励配置")
        
        # 分析稳定性
        stability = self._analyze_baseline_stability()
        if stability.get('overall_stability') == 'unstable':
            recommendations.append("基线变化较大，考虑降低EMA平滑因子alpha或增加预热期")
        
        # 检查训练进度
        if len(self.snapshots) > 1:
            episode_rate = (latest.episode_count - self.snapshots[0].episode_count) / len(self.snapshots)
            if episode_rate < 5:
                recommendations.append("训练进度较慢，考虑增加并行环境数量或调整训练参数")
        
        return recommendations if recommendations else ["基线监控正常，继续当前训练配置"]
    
    def close(self) -> None:
        """关闭监控器，保存最终数据"""
        self.record_snapshot(force=True)  # 强制记录最后一个快照
        self.save_history()
        
        # 生成最终报告
        final_report = self.generate_summary_report()
        report_file = self.monitor_dir / 'final_report.json'
        
        try:
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            logger.info(f"基线监控最终报告已保存: {report_file}")
        except Exception as e:
            logger.error(f"保存最终报告时发生错误: {e}")
        
        # 生成最终图表
        self.plot_baseline_evolution(save_plots=True)
        
        logger.info("基线监控器已关闭")