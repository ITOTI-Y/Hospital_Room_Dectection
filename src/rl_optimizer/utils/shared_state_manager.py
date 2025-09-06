"""
SharedStateManager - 跨进程共享状态管理器

用于PPO多进程训练环境中维护全局共享的动态基线状态。
"""

import multiprocessing as mp
from multiprocessing import Lock, Manager
from threading import RLock
from typing import Dict, Any, Optional, Union
import time
from src.rl_optimizer.utils.setup import setup_logger
from dataclasses import dataclass

logger = setup_logger(__name__)


@dataclass
class EMAState:
    """指数移动平均状态数据类"""
    value: float = 0.0
    count: int = 0
    last_updated: float = 0.0
    is_initialized: bool = False


class SharedStateManager:
    """
    跨进程共享状态管理器
    
    实现PPO多进程环境之间的全局状态共享，主要用于维护动态基线的
    指数移动平均值和相关统计信息。使用multiprocessing.Manager
    实现真正的跨进程数据共享。
    """

    _instance = None
    _lock = RLock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式，确保全局只有一个实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, alpha: float = 0.1, warmup_episodes: int = 100):
        """
        初始化共享状态管理器
        
        Args:
            alpha: 指数移动平均的平滑因子 (0 < alpha <= 1)
            warmup_episodes: 预热期间episode数量，在此期间不更新基线
        """
        if self._initialized:
            return
            
        self.alpha = alpha
        self.warmup_episodes = warmup_episodes
        
        # 创建进程管理器和共享数据结构
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.update_lock = self.manager.Lock()
        
        # 初始化各种EMA状态
        self._init_ema_states()
        
        # 初始化全局统计信息
        self.shared_dict['global_episode_count'] = 0
        self.shared_dict['total_episodes'] = 0
        self.shared_dict['start_time'] = time.time()
        
        self._initialized = True
        logger.info(f"SharedStateManager初始化完成: alpha={alpha}, warmup_episodes={warmup_episodes}")
    
    def _init_ema_states(self):
        """初始化各种指数移动平均状态"""
        ema_keys = [
            'time_cost_baseline',       # 时间成本基线
            'adjacency_reward_baseline', # 相邻性奖励基线
            'area_match_baseline',      # 面积匹配基线
            'total_reward_baseline',    # 总奖励基线
            'time_cost_std',           # 时间成本标准差
            'adjacency_reward_std',     # 相邻性奖励标准差
            'area_match_std',          # 面积匹配标准差
        ]
        
        for key in ema_keys:
            self.shared_dict[key] = {
                'value': 0.0,
                'count': 0,
                'last_updated': time.time(),
                'is_initialized': False,
                'variance': 0.0  # 用于计算标准差
            }
    
    def update_ema(self, key: str, new_value: float, force_update: bool = False) -> None:
        """
        更新指定键的指数移动平均值
        
        Args:
            key: 要更新的EMA键名
            new_value: 新的观测值
            force_update: 是否强制更新（忽略预热期）
        """
        if key not in self.shared_dict:
            logger.error(f"未知的EMA键: {key}")
            return
            
        with self.update_lock:
            current_episodes = self.shared_dict['global_episode_count']
            
            # 检查是否在预热期内
            if not force_update and current_episodes < self.warmup_episodes:
                # 预热期只收集数据，不更新基线
                ema_data = dict(self.shared_dict[key])
                ema_data['count'] += 1
                ema_data['last_updated'] = time.time()
                self.shared_dict[key] = ema_data
                return
            
            # 获取当前EMA状态
            ema_data = dict(self.shared_dict[key])
            
            if not ema_data['is_initialized']:
                # 首次初始化
                ema_data['value'] = new_value
                ema_data['is_initialized'] = True
                ema_data['variance'] = 0.0
                logger.info(f"初始化EMA基线 {key}: {new_value:.6f}")
            else:
                # 更新EMA值
                old_value = ema_data['value']
                ema_data['value'] = (1 - self.alpha) * old_value + self.alpha * new_value
                
                # 更新方差（用于计算标准差）
                if ema_data['count'] > 0:
                    delta = new_value - old_value
                    ema_data['variance'] = (1 - self.alpha) * ema_data['variance'] + self.alpha * delta * delta
            
            # 更新统计信息
            ema_data['count'] += 1
            ema_data['last_updated'] = time.time()
            
            # 写回共享字典
            self.shared_dict[key] = ema_data
            
            std_dev = ema_data['variance'] ** 0.5 if ema_data['variance'] > 0 else 0.0
            logger.debug(f"更新EMA {key}: 新值={new_value:.6f}, EMA={ema_data['value']:.6f}, 标准差={std_dev:.6f}")
    
    def get_ema_value(self, key: str) -> Optional[float]:
        """
        获取指定键的EMA值
        
        Args:
            key: EMA键名
            
        Returns:
            float: EMA值，如果未初始化返回None
        """
        if key not in self.shared_dict:
            return None
            
        ema_data = self.shared_dict[key]
        if not ema_data['is_initialized']:
            return None
            
        return ema_data['value']
    
    def get_ema_std(self, key: str) -> Optional[float]:
        """
        获取指定键的EMA标准差
        
        Args:
            key: EMA键名
            
        Returns:
            float: 标准差值，如果未初始化返回None
        """
        if key not in self.shared_dict:
            return None
            
        ema_data = self.shared_dict[key]
        if not ema_data['is_initialized'] or ema_data['variance'] <= 0:
            return None
            
        return ema_data['variance'] ** 0.5
    
    def get_ema_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取指定键的完整EMA信息
        
        Args:
            key: EMA键名
            
        Returns:
            dict: 包含value, count, std等完整信息的字典
        """
        if key not in self.shared_dict:
            return None
            
        ema_data = dict(self.shared_dict[key])
        ema_data['std'] = ema_data['variance'] ** 0.5 if ema_data['variance'] > 0 else 0.0
        return ema_data
    
    def is_warmup_complete(self) -> bool:
        """检查预热期是否完成"""
        return self.shared_dict['global_episode_count'] >= self.warmup_episodes
    
    def increment_episode_count(self) -> int:
        """增加全局episode计数并返回当前值"""
        with self.update_lock:
            self.shared_dict['global_episode_count'] += 1
            self.shared_dict['total_episodes'] += 1
            return self.shared_dict['global_episode_count']
    
    def get_episode_count(self) -> int:
        """获取当前的全局episode计数"""
        return self.shared_dict['global_episode_count']
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取所有共享状态的统计信息
        
        Returns:
            dict: 包含所有EMA状态和全局统计信息
        """
        stats = {
            'global_episode_count': self.shared_dict['global_episode_count'],
            'total_episodes': self.shared_dict['total_episodes'],
            'warmup_complete': self.is_warmup_complete(),
            'uptime_seconds': time.time() - self.shared_dict['start_time'],
            'ema_states': {}
        }
        
        # 收集所有EMA状态信息
        for key in self.shared_dict:
            if isinstance(self.shared_dict[key], dict) and 'value' in self.shared_dict[key]:
                stats['ema_states'][key] = self.get_ema_info(key)
        
        return stats
    
    def reset(self):
        """重置所有状态（主要用于测试）"""
        with self.update_lock:
            logger.info("重置SharedStateManager状态")
            self._init_ema_states()
            self.shared_dict['global_episode_count'] = 0
            self.shared_dict['total_episodes'] = 0
            self.shared_dict['start_time'] = time.time()
    
    def update_time_cost_baseline(self, time_cost: float) -> None:
        """更新时间成本基线的便捷方法"""
        self.update_ema('time_cost_baseline', time_cost)
    
    def update_adjacency_baseline(self, adjacency_reward: float) -> None:
        """更新相邻性奖励基线的便捷方法"""
        self.update_ema('adjacency_reward_baseline', adjacency_reward)
    
    def update_area_match_baseline(self, area_match_score: float) -> None:
        """更新面积匹配基线的便捷方法"""
        self.update_ema('area_match_baseline', area_match_score)
    
    def update_total_reward_baseline(self, total_reward: float) -> None:
        """更新总奖励基线的便捷方法"""
        self.update_ema('total_reward_baseline', total_reward)
    
    def get_time_cost_baseline(self) -> Optional[float]:
        """获取时间成本基线的便捷方法"""
        return self.get_ema_value('time_cost_baseline')
    
    def get_adjacency_baseline(self) -> Optional[float]:
        """获取相邻性奖励基线的便捷方法"""
        return self.get_ema_value('adjacency_reward_baseline')
    
    def get_area_match_baseline(self) -> Optional[float]:
        """获取面积匹配基线的便捷方法"""
        return self.get_ema_value('area_match_baseline')
    
    def get_total_reward_baseline(self) -> Optional[float]:
        """获取总奖励基线的便捷方法"""
        return self.get_ema_value('total_reward_baseline')


# 全局实例（延迟初始化）
_global_shared_state_manager = None


def get_shared_state_manager(alpha: float = 0.1, warmup_episodes: int = 100) -> SharedStateManager:
    """
    获取全局共享状态管理器实例
    
    Args:
        alpha: EMA平滑因子（仅在首次调用时有效）
        warmup_episodes: 预热期episodes数量（仅在首次调用时有效）
        
    Returns:
        SharedStateManager: 全局实例
    """
    global _global_shared_state_manager
    
    if _global_shared_state_manager is None:
        _global_shared_state_manager = SharedStateManager(alpha=alpha, warmup_episodes=warmup_episodes)
    
    return _global_shared_state_manager


def reset_shared_state_manager():
    """重置全局共享状态管理器（主要用于测试）"""
    global _global_shared_state_manager
    if _global_shared_state_manager is not None:
        _global_shared_state_manager.reset()