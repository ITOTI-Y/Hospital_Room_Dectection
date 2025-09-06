# src/rl_optimizer/utils/tensorboard_callback.py


from stable_baselines3.common.callbacks import BaseCallback

from src.rl_optimizer.utils.shared_state_manager import get_shared_state_manager
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class TensorboardBaselineCallback(BaseCallback):
    """
    自定义回调，用于将baseline指标记录到tensorboard
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1
    ):
        """
        初始化tensorboard baseline回调
        
        Args:
            log_freq (int): 每多少步记录一次指标
            verbose (int): 日志详细级别
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.shared_state = get_shared_state_manager()
        
    def _init_callback(self) -> None:
        """初始化回调"""
        pass
        
    def _on_step(self) -> bool:
        """每个训练步后的回调"""
        if self.n_calls % self.log_freq == 0:
            self._log_baseline_metrics()
        return True
        
    def _log_baseline_metrics(self) -> None:
        """记录baseline指标到tensorboard"""
        try:
            # 获取各种baseline指标
            time_cost_baseline = self.shared_state.get_time_cost_baseline()
            adjacency_baseline = self.shared_state.get_adjacency_baseline()
            area_match_baseline = self.shared_state.get_area_match_baseline()
            total_reward_baseline = self.shared_state.get_total_reward_baseline()
            
            # 记录到tensorboard
            if time_cost_baseline is not None:
                self.logger.record("baseline/time_cost_baseline", time_cost_baseline)
                
            if adjacency_baseline is not None:
                self.logger.record("baseline/adjacency_baseline", adjacency_baseline)
                
            if area_match_baseline is not None:
                self.logger.record("baseline/area_match_baseline", area_match_baseline)
                
            if total_reward_baseline is not None:
                self.logger.record("baseline/total_reward_baseline", total_reward_baseline)
                
        except Exception as e:
            logger.error(f"记录baseline指标时发生错误: {e}", exc_info=True)