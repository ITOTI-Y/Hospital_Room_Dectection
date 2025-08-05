# src/rl_optimizer/utils/training_metrics_callback.py

import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class TrainingMetricsCallback(BaseCallback):
    """
    训练指标回调类，用于跟踪和记录PPO训练过程中的关键指标。
    
    主要功能：
    - 记录每个episode的实际加权时间
    - 跟踪最佳布局和最低时间成本
    - 提供训练进度的实时反馈
    - 集成到TensorBoard进行可视化
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        save_freq: int = 1000,
        save_path: Optional[str] = None,
        window_size: int = 100,
        verbose: int = 1,
        total_episodes_target: Optional[int] = None
    ):
        """
        初始化训练指标回调。
        
        Args:
            log_freq (int): 每多少个episode记录一次日志
            save_freq (int): 每多少个episode保存一次指标历史
            save_path (Optional[str]): 指标保存路径
            window_size (int): 滑动窗口大小，用于计算平均值
            verbose (int): 日志详细级别
            total_episodes_target (Optional[int]): 总的目标episodes数量，用于显示进度条
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = Path(save_path) if save_path else None
        self.window_size = window_size
        self.total_episodes_target = total_episodes_target
        
        # 检查是否为主进程
        import multiprocessing
        try:
            self.is_main_process = multiprocessing.current_process().name == 'MainProcess'
        except:
            self.is_main_process = True
        
        # 如果不是主进程，降低日志详细程度
        if not self.is_main_process:
            self.verbose = 0
        
        # 指标跟踪
        self.episode_count = 0
        self.time_costs = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        
        # 最佳结果跟踪
        self.best_time_cost = float('inf')
        self.best_layout = None
        self.best_episode = 0
        
        # 历史记录
        self.history = {
            'episodes': [],
            'time_costs': [],
            'rewards': [],
            'episode_lengths': [],
            'best_time_costs': [],
            'timestamps': []
        }
        
        # 时间跟踪
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            
    def _init_callback(self) -> None:
        """初始化回调时的操作。"""
        if self.verbose >= 1 and self.is_main_process:
            logger.info("训练指标回调已初始化")
            logger.info(f"日志频率: {self.log_freq} episodes")
            logger.info(f"保存频率: {self.save_freq} episodes")
            if self.save_path:
                logger.info(f"指标保存路径: {self.save_path}")
                
    def _on_step(self) -> bool:
        """每个训练步后的回调。"""
        # 在SB3的VecEnv中，episode信息通过不同方式传递
        # 检查self.locals中的infos
        infos = self.locals.get('infos', [])
        
        if self.verbose >= 2 and self.is_main_process:
            logger.debug(f"回调接收到 {len(infos)} 个info，内容: {infos}")
        
        if len(infos) > 0:
            for i, info in enumerate(infos):
                # 检查是否有episode结束信息
                if 'episode' in info:
                    if self.verbose >= 2 and self.is_main_process:
                        logger.debug(f"发现episode结束信息: {info['episode']}")
                    self._process_episode_end(info['episode'])
                
                # 检查SB3标准的episode信息格式
                elif '_episode' in info:
                    episode_data = info['_episode']
                    if self.verbose >= 2 and self.is_main_process:
                        logger.debug(f"发现SB3格式episode信息: {episode_data}")
                    self._process_episode_end(episode_data)
                
                # 检查其他可能的episode信息字段
                elif any(key in info for key in ['terminal_observation', 'episode_info']):
                    if self.verbose >= 2 and self.is_main_process:
                        logger.debug(f"发现其他格式episode信息: {info}")
                    # 尝试构造episode数据
                    episode_data = {}
                    if 'episode_info' in info:
                        episode_data = info['episode_info']
                    self._process_episode_end(episode_data)
        
        return True
        
    def _process_episode_end(self, episode_info: Dict[str, Any]) -> None:
        """处理episode结束时的指标。"""
        self.episode_count += 1
        
        # 提取episode信息
        episode_reward = episode_info.get('r', 0.0)
        episode_length = episode_info.get('l', 0)
        
        # 提取时间成本（如果环境提供了的话）
        time_cost = episode_info.get('time_cost', None)
        layout = episode_info.get('layout', None)
        
        # 记录到滑动窗口
        self.rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if time_cost is not None:
            self.time_costs.append(time_cost)
            
            # 更新最佳结果
            if time_cost < self.best_time_cost:
                self.best_time_cost = time_cost
                self.best_layout = layout
                self.best_episode = self.episode_count
                
                # 只有主进程记录最佳结果发现日志
                if self.verbose >= 1 and self.is_main_process:
                    scaled_reward = -time_cost / 1e4
                    logger.info(f"🎉 发现新的最佳布局！Episode {self.episode_count}: "
                              f"【原始】时间成本 = {time_cost:.2f} 秒 "
                              f"(对应缩放reward: {scaled_reward:.6f})")
        
        # 记录到历史（所有进程都记录，但只有主进程保存文件）
        current_time = time.time()
        self.history['episodes'].append(self.episode_count)
        self.history['time_costs'].append(time_cost if time_cost is not None else float('nan'))
        self.history['rewards'].append(episode_reward)
        self.history['episode_lengths'].append(episode_length)
        self.history['best_time_costs'].append(self.best_time_cost)
        self.history['timestamps'].append(current_time)
        
        # 定期日志输出（只有主进程）
        if self.episode_count % self.log_freq == 0 and self.is_main_process:
            self._log_metrics()
            
        # 定期保存指标（只有主进程）
        if (self.save_freq > 0 and self.episode_count % self.save_freq == 0 and 
            self.is_main_process):
            self._save_metrics()
            
        # TensorBoard日志
        self._log_to_tensorboard(time_cost, episode_reward, episode_length)
        
    def _log_metrics(self) -> None:
        """输出训练指标日志。"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_since_last_log = current_time - self.last_log_time
        
        # 计算统计信息
        if self.time_costs:
            avg_time_cost = np.mean(self.time_costs)
            std_time_cost = np.std(self.time_costs)
            min_time_cost = np.min(self.time_costs)
            max_time_cost = np.max(self.time_costs)
        else:
            avg_time_cost = std_time_cost = min_time_cost = max_time_cost = float('nan')
            
        avg_reward = np.mean(self.rewards) if self.rewards else float('nan')
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else float('nan')
        
        # 计算训练速度
        episodes_per_second = self.log_freq / time_since_last_log if time_since_last_log > 0 else 0
        
        logger.info("=" * 80)
        logger.info(f"📊 训练指标报告 - Episode {self.episode_count}")
        logger.info("-" * 80)
        logger.info(f"⏱️  训练时间: {elapsed_time/3600:.2f} 小时")
        logger.info(f"🚀 训练速度: {episodes_per_second:.2f} episodes/秒")
        logger.info("")
        
        if not np.isnan(avg_time_cost):
            logger.info(f"🎯 【原始】加权时间成本 (最近{len(self.time_costs)}个episodes):")
            logger.info(f"   平均值: {avg_time_cost:.2f} 秒 ± {std_time_cost:.2f}")
            logger.info(f"   范围: [{min_time_cost:.2f}, {max_time_cost:.2f}] 秒")
            logger.info(f"   最佳值: {self.best_time_cost:.2f} 秒 (Episode {self.best_episode})")
            
            # 显示对应的缩放值用于对比
            avg_scaled = -avg_time_cost / 1e4
            best_scaled = -self.best_time_cost / 1e4
            logger.info(f"   对比：平均缩放reward: {avg_scaled:.6f}, 最佳缩放reward: {best_scaled:.6f}")
            logger.info("")
            
        logger.info(f"🏆 奖励统计 (最近{len(self.rewards)}个episodes):")
        logger.info(f"   平均奖励: {avg_reward:.4f}")
        logger.info(f"   平均长度: {avg_length:.1f} 步")
        
        # 添加进度条显示（如果有总目标的话）
        if hasattr(self, 'total_episodes_target') and self.total_episodes_target > 0:
            progress_percent = (self.episode_count / self.total_episodes_target) * 100
            progress_bar_length = 40
            filled_length = int(progress_bar_length * self.episode_count // self.total_episodes_target)
            bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
            logger.info(f"📈 训练进度: [{bar}] {progress_percent:.1f}% ({self.episode_count}/{self.total_episodes_target})")
        
        logger.info("=" * 80)
        
        self.last_log_time = current_time
        
    def _log_to_tensorboard(self, time_cost: Optional[float], reward: float, length: int) -> None:
        """记录指标到TensorBoard。"""
        # 检查是否有可用的logger（在回调正式使用时才有）
        try:
            if hasattr(self, 'logger') and self.logger is not None:
                # 基本指标
                self.logger.record("episode/reward", reward)
                self.logger.record("episode/length", length)
                self.logger.record("episode/count", self.episode_count)
                
                # 时间成本指标
                if time_cost is not None:
                    self.logger.record("episode/time_cost", time_cost)
                    self.logger.record("episode/best_time_cost", self.best_time_cost)
                    
                    # 相对改进
                    if self.best_time_cost > 0:
                        improvement = (time_cost - self.best_time_cost) / self.best_time_cost * 100
                        self.logger.record("episode/time_cost_vs_best_percent", improvement)
                
                # 滑动平均
                if len(self.time_costs) > 0:
                    self.logger.record("episode/avg_time_cost", np.mean(self.time_costs))
                if len(self.rewards) > 0:
                    self.logger.record("episode/avg_reward", np.mean(self.rewards))
                if len(self.episode_lengths) > 0:
                    self.logger.record("episode/avg_length", np.mean(self.episode_lengths))
        except Exception as e:
            # 在测试或其他特殊情况下，logger可能不可用，这是正常的
            if self.verbose >= 2:
                logger.warning(f"TensorBoard日志记录失败: {e}")
            
    def _save_metrics(self) -> None:
        """保存指标历史到文件。"""
        if not self.save_path or not self.is_main_process:
            return
            
        try:
            import json
            
            # 保存完整历史
            history_file = self.save_path / f"training_metrics_ep{self.episode_count}.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                # 转换numpy类型为Python原生类型
                history_to_save = {}
                for key, values in self.history.items():
                    history_to_save[key] = [
                        float(v) if isinstance(v, (np.integer, np.floating)) else v 
                        for v in values
                    ]
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
                
            # 保存最佳结果
            if self.best_layout is not None:
                best_file = self.save_path / f"best_layout_ep{self.episode_count}.json"
                best_info = {
                    "episode": self.best_episode,
                    "time_cost": float(self.best_time_cost),
                    "layout": self.best_layout,
                    "timestamp": time.time()
                }
                with open(best_file, 'w', encoding='utf-8') as f:
                    json.dump(best_info, f, ensure_ascii=False, indent=2)
                    
            if self.verbose >= 2:
                logger.info(f"训练指标已保存到: {self.save_path}")
                
        except Exception as e:
            logger.error(f"保存训练指标时发生错误: {e}", exc_info=True)
            
    def get_best_result(self) -> Dict[str, Any]:
        """获取当前最佳结果。"""
        return {
            "episode": self.best_episode,
            "time_cost": self.best_time_cost,
            "layout": self.best_layout,
            "total_episodes": self.episode_count
        }
        
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息。"""
        stats = {
            "episode_count": self.episode_count,
            "best_time_cost": self.best_time_cost,
            "best_episode": self.best_episode
        }
        
        if self.time_costs:
            stats.update({
                "avg_time_cost": float(np.mean(self.time_costs)),
                "std_time_cost": float(np.std(self.time_costs)),
                "min_time_cost": float(np.min(self.time_costs)),
                "max_time_cost": float(np.max(self.time_costs))
            })
            
        if self.rewards:
            stats.update({
                "avg_reward": float(np.mean(self.rewards)),
                "std_reward": float(np.std(self.rewards))
            })
            
        return stats