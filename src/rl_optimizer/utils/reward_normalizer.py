"""
RewardNormalizer - 奖励归一化器

实现基于动态基线的奖励归一化，将各种奖励组件标准化到合理范围内，
并支持相对改进奖励机制。
"""

import numpy as np
from src.rl_optimizer.utils.setup import setup_logger
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.config import RLConfig
from src.rl_optimizer.utils.shared_state_manager import SharedStateManager

logger = setup_logger(__name__)


@dataclass
class RewardComponents:
    """奖励组件数据类"""
    time_cost: float = 0.0
    adjacency_reward: float = 0.0
    area_match_reward: float = 0.0
    skip_penalty: float = 0.0
    completion_bonus: float = 0.0
    placement_bonus: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'time_cost': self.time_cost,
            'adjacency_reward': self.adjacency_reward,
            'area_match_reward': self.area_match_reward,
            'skip_penalty': self.skip_penalty,
            'completion_bonus': self.completion_bonus,
            'placement_bonus': self.placement_bonus
        }


@dataclass
class NormalizedRewardInfo:
    """归一化奖励信息数据类"""
    normalized_components: RewardComponents
    total_normalized_reward: float
    raw_components: RewardComponents
    baselines_used: Dict[str, Optional[float]]
    improvement_scores: Dict[str, Optional[float]]
    is_relative_improvement: bool = False


class RewardNormalizer:
    """
    奖励归一化器
    
    负责将各种奖励组件归一化到统一范围，基于动态基线计算相对改进奖励，
    并提供统一的权重组合机制。
    """
    
    def __init__(self, config: RLConfig, shared_state_manager: SharedStateManager):
        """
        初始化奖励归一化器
        
        Args:
            config: RL配置对象
            shared_state_manager: 共享状态管理器
        """
        self.config = config
        self.shared_state = shared_state_manager
        
        # 验证配置
        if not config.ENABLE_DYNAMIC_BASELINE:
            logger.warning("动态基线未启用，将使用固定基线归一化")
        
        logger.info("奖励归一化器初始化完成")
    
    def normalize_reward_components(self, components: RewardComponents) -> NormalizedRewardInfo:
        """
        归一化所有奖励组件
        
        Args:
            components: 原始奖励组件
            
        Returns:
            NormalizedRewardInfo: 归一化后的奖励信息
        """
        normalized_components = RewardComponents()
        baselines_used = {}
        improvement_scores = {}
        
        # 1. 归一化时间成本（越小越好，所以取负值）
        time_cost_normalized, time_baseline, time_improvement = self._normalize_time_cost(
            -abs(components.time_cost)  # 确保时间成本为负值
        )
        normalized_components.time_cost = time_cost_normalized
        baselines_used['time_cost'] = time_baseline
        improvement_scores['time_cost'] = time_improvement
        
        # 2. 归一化相邻性奖励（越大越好）
        adj_normalized, adj_baseline, adj_improvement = self._normalize_adjacency_reward(
            components.adjacency_reward
        )
        normalized_components.adjacency_reward = adj_normalized
        baselines_used['adjacency_reward'] = adj_baseline
        improvement_scores['adjacency_reward'] = adj_improvement
        
        # 3. 归一化面积匹配奖励（越大越好）
        area_normalized, area_baseline, area_improvement = self._normalize_area_match_reward(
            components.area_match_reward
        )
        normalized_components.area_match_reward = area_normalized
        baselines_used['area_match_reward'] = area_baseline
        improvement_scores['area_match_reward'] = area_improvement
        
        # 4. 处理惩罚和奖励项（这些通常是预定义的，不需要基线归一化）
        normalized_components.skip_penalty = self._clip_value(
            components.skip_penalty, -5.0, 0.0
        )
        normalized_components.completion_bonus = self._clip_value(
            components.completion_bonus, 0.0, 5.0
        )
        normalized_components.placement_bonus = self._clip_value(
            components.placement_bonus, 0.0, 5.0
        )
        
        # 5. 计算加权总奖励
        total_normalized_reward = self._compute_weighted_reward(normalized_components, improvement_scores)
        
        # 6. 应用最终裁剪
        if self.config.ENABLE_REWARD_CLIPPING:
            total_normalized_reward = self._clip_value(
                total_normalized_reward, 
                self.config.REWARD_CLIP_RANGE[0],
                self.config.REWARD_CLIP_RANGE[1]
            )
        
        # 7. 检查是否使用了相对改进
        is_relative_improvement = any(score is not None for score in improvement_scores.values())
        
        return NormalizedRewardInfo(
            normalized_components=normalized_components,
            total_normalized_reward=total_normalized_reward,
            raw_components=components,
            baselines_used=baselines_used,
            improvement_scores=improvement_scores,
            is_relative_improvement=is_relative_improvement
        )
    
    def _normalize_time_cost(self, time_cost: float) -> Tuple[float, Optional[float], Optional[float]]:
        """
        归一化时间成本
        
        Args:
            time_cost: 原始时间成本（应为负值）
            
        Returns:
            tuple: (归一化值, 使用的基线, 改进分数)
        """
        baseline = self.shared_state.get_time_cost_baseline()
        
        if baseline is None or not self.config.ENABLE_DYNAMIC_BASELINE:
            # 没有基线时，使用固定缩放
            normalized = time_cost / 10000.0  # 固定缩放因子
            return self._clip_normalized_value(normalized), baseline, None
        
        # 使用动态基线进行相对改进计算
        if self.config.ENABLE_RELATIVE_IMPROVEMENT_REWARD:
            # 计算相对改进（时间成本越小越好）
            improvement_ratio = (baseline + time_cost) / (abs(baseline) + 1e-8)
            improvement_score = improvement_ratio
            normalized = self._clip_normalized_value(improvement_score)
            return normalized, baseline, improvement_ratio
        else:
            # 基于基线的标准化
            time_cost_std = self.shared_state.get_ema_std('time_cost_std')
            if time_cost_std and time_cost_std > self.config.REWARD_NORMALIZATION_MIN_STD:
                normalized = (time_cost - baseline) / time_cost_std
                normalized = self._clip_normalized_value(normalized)
            else:
                normalized = self._clip_normalized_value(time_cost / 10000.0)
            return normalized, baseline, None
    
    def _normalize_adjacency_reward(self, adjacency_reward: float) -> Tuple[float, Optional[float], Optional[float]]:
        """
        归一化相邻性奖励
        
        Args:
            adjacency_reward: 原始相邻性奖励
            
        Returns:
            tuple: (归一化值, 使用的基线, 改进分数)
        """
        baseline = self.shared_state.get_adjacency_baseline()
        
        if baseline is None or not self.config.ENABLE_DYNAMIC_BASELINE:
            # 没有基线时，直接裁剪到合理范围
            normalized = self._clip_value(adjacency_reward, -1.0, 1.0)
            return normalized, baseline, None
        
        # 使用动态基线
        if self.config.ENABLE_RELATIVE_IMPROVEMENT_REWARD:
            # 计算相对改进（相邻性奖励越大越好）
            improvement_ratio = (adjacency_reward - baseline) / (abs(baseline) + 1e-8)
            improvement_score = improvement_ratio
            normalized = self._clip_normalized_value(improvement_score)
            return normalized, baseline, improvement_ratio
        else:
            # 基于基线的标准化
            adj_std = self.shared_state.get_ema_std('adjacency_reward_std')
            if adj_std and adj_std > self.config.REWARD_NORMALIZATION_MIN_STD:
                normalized = (adjacency_reward - baseline) / adj_std
                normalized = self._clip_normalized_value(normalized)
            else:
                normalized = self._clip_value(adjacency_reward, -1.0, 1.0)
            return normalized, baseline, None
    
    def _normalize_area_match_reward(self, area_match_reward: float) -> Tuple[float, Optional[float], Optional[float]]:
        """
        归一化面积匹配奖励
        
        Args:
            area_match_reward: 原始面积匹配奖励
            
        Returns:
            tuple: (归一化值, 使用的基线, 改进分数)
        """
        baseline = self.shared_state.get_area_match_baseline()
        
        if baseline is None or not self.config.ENABLE_DYNAMIC_BASELINE:
            # 没有基线时，直接裁剪到合理范围
            normalized = self._clip_value(area_match_reward, -1.0, 1.0)
            return normalized, baseline, None
        
        # 使用动态基线
        if self.config.ENABLE_RELATIVE_IMPROVEMENT_REWARD:
            # 计算相对改进（面积匹配奖励越大越好）
            improvement_ratio = (area_match_reward - baseline) / (abs(baseline) + 1e-8)
            improvement_score = improvement_ratio
            normalized = self._clip_normalized_value(improvement_score)
            return normalized, baseline, improvement_ratio
        else:
            # 基于基线的标准化
            area_std = self.shared_state.get_ema_std('area_match_std')
            if area_std and area_std > self.config.REWARD_NORMALIZATION_MIN_STD:
                normalized = (area_match_reward - baseline) / area_std
                normalized = self._clip_normalized_value(normalized)
            else:
                normalized = self._clip_value(area_match_reward, -1.0, 1.0)
            return normalized, baseline, None
    
    def _compute_weighted_reward(self, components: RewardComponents, improvement_scores: Dict[str, Optional[float]]) -> float:
        """
        计算加权总奖励
        
        Args:
            components: 归一化后的奖励组件
            
        Returns:
            float: 加权总奖励
        """
        

        if any(improvement_scores.values()):
            total_reward = (
                improvement_scores['time_cost'] * self.config.NORMALIZED_TIME_WEIGHT +
                improvement_scores['adjacency_reward'] * self.config.NORMALIZED_ADJACENCY_WEIGHT +
                improvement_scores['area_match_reward'] * self.config.NORMALIZED_AREA_WEIGHT
            )
        
        else:
            total_reward = (
            components.time_cost * self.config.NORMALIZED_TIME_WEIGHT +
            components.adjacency_reward * self.config.NORMALIZED_ADJACENCY_WEIGHT +
            components.area_match_reward * self.config.NORMALIZED_AREA_WEIGHT +
            components.skip_penalty * self.config.NORMALIZED_SKIP_PENALTY_WEIGHT +
            components.completion_bonus * self.config.NORMALIZED_COMPLETION_BONUS_WEIGHT +
            components.placement_bonus * self.config.NORMALIZED_PLACEMENT_BONUS_WEIGHT
            )
        
        return total_reward
    
    def _clip_normalized_value(self, value: float) -> float:
        """
        裁剪归一化值到合理范围
        
        Args:
            value: 待裁剪的值
            
        Returns:
            float: 裁剪后的值
        """
        clip_range = self.config.REWARD_NORMALIZATION_CLIP_RANGE
        return self._clip_value(value, -clip_range, clip_range)
    
    def _clip_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        裁剪值到指定范围
        
        Args:
            value: 待裁剪的值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            float: 裁剪后的值
        """
        return np.clip(value, min_val, max_val)
    
    def update_baselines(self, components: RewardComponents) -> None:
        """
        更新动态基线
        
        Args:
            components: 当前episode的奖励组件
        """
        if not self.config.ENABLE_DYNAMIC_BASELINE:
            return
        
        # 更新时间成本基线（使用绝对值）
        self.shared_state.update_time_cost_baseline(abs(components.time_cost))
        
        # 更新相邻性奖励基线
        if components.adjacency_reward != 0.0:  # 避免更新零值
            self.shared_state.update_adjacency_baseline(components.adjacency_reward)
        
        # 更新面积匹配基线
        if components.area_match_reward != 0.0:  # 避免更新零值
            self.shared_state.update_area_match_baseline(components.area_match_reward)
        
        # 更新标准差估计（用于计算方差的EMA）
        self._update_variance_baselines(components)
    
    def _update_variance_baselines(self, components: RewardComponents) -> None:
        """
        更新方差基线（用于标准差计算）
        
        Args:
            components: 当前episode的奖励组件
        """
        # 获取当前基线
        time_baseline = self.shared_state.get_time_cost_baseline()
        adj_baseline = self.shared_state.get_adjacency_baseline()
        area_baseline = self.shared_state.get_area_match_baseline()
        
        # 计算偏差的平方并更新方差EMA
        if time_baseline is not None:
            time_deviation_sq = (abs(components.time_cost) - time_baseline) ** 2
            self.shared_state.update_ema('time_cost_std', time_deviation_sq)
        
        if adj_baseline is not None and components.adjacency_reward != 0.0:
            adj_deviation_sq = (components.adjacency_reward - adj_baseline) ** 2
            self.shared_state.update_ema('adjacency_reward_std', adj_deviation_sq)
        
        if area_baseline is not None and components.area_match_reward != 0.0:
            area_deviation_sq = (components.area_match_reward - area_baseline) ** 2
            self.shared_state.update_ema('area_match_std', area_deviation_sq)
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        获取当前归一化统计信息
        
        Returns:
            dict: 包含基线、标准差等统计信息
        """
        stats = {
            'time_cost_baseline': self.shared_state.get_time_cost_baseline(),
            'adjacency_baseline': self.shared_state.get_adjacency_baseline(),
            'area_match_baseline': self.shared_state.get_area_match_baseline(),
            'time_cost_std': self.shared_state.get_ema_std('time_cost_std'),
            'adjacency_reward_std': self.shared_state.get_ema_std('adjacency_reward_std'),
            'area_match_std': self.shared_state.get_ema_std('area_match_std'),
            'warmup_complete': self.shared_state.is_warmup_complete(),
            'episode_count': self.shared_state.get_episode_count(),
            'config': {
                'enable_dynamic_baseline': self.config.ENABLE_DYNAMIC_BASELINE,
                'enable_relative_improvement': self.config.ENABLE_RELATIVE_IMPROVEMENT_REWARD,
                'ema_alpha': self.config.EMA_ALPHA,
                'warmup_episodes': self.config.BASELINE_WARMUP_EPISODES,
                'relative_improvement_scale': self.config.RELATIVE_IMPROVEMENT_SCALE
            }
        }
        
        return stats
    
    def log_reward_info(self, reward_info: NormalizedRewardInfo, episode: int) -> None:
        """
        记录奖励信息到日志
        
        Args:
            reward_info: 归一化奖励信息
            episode: 当前episode编号
        """
        
        raw = reward_info.raw_components
        norm = reward_info.normalized_components
        
        logger.debug(f"Episode {episode} 奖励归一化详情:")
        logger.debug(f"  原始组件: 时间成本={raw.time_cost:.2f}, 相邻性={raw.adjacency_reward:.4f}, "
                    f"面积匹配={raw.area_match_reward:.4f}")
        logger.debug(f"  归一化组件: 时间成本={norm.time_cost:.4f}, 相邻性={norm.adjacency_reward:.4f}, "
                    f"面积匹配={norm.area_match_reward:.4f}")
        logger.debug(f"  惩罚奖励: 跳过惩罚={norm.skip_penalty:.2f}, 完成奖励={norm.completion_bonus:.2f}")
        logger.debug(f"  总归一化奖励: {reward_info.total_normalized_reward:.4f}")
        
        if reward_info.is_relative_improvement:
            logger.debug(f"  改进分数: {reward_info.improvement_scores}")
            logger.debug(f"  使用基线: {reward_info.baselines_used}")