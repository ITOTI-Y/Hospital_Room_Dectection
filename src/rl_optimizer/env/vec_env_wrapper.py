# src/rl_optimizer/env/vec_env_wrapper.py

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class EpisodeInfoVecEnvWrapper(VecEnvWrapper):
    """
    VecEnv包装器，确保episode信息能正确传递给回调函数。
    
    解决SB3中VecEnv环境episode信息传递的问题，特别是自定义info字典的传递。
    """
    
    def __init__(self, venv):
        """
        初始化包装器
        
        Args:
            venv: 要包装的VecEnv
        """
        super().__init__(venv)
        self.episode_rewards = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.episode_count = 0
        
    def step_wait(self) -> VecEnvStepReturn:
        """
        等待并处理环境步骤结果，确保episode信息正确传递
        """
        observations, rewards, dones, infos = self.venv.step_wait()
        
        # 更新episode统计
        self.episode_rewards += rewards
        self.episode_lengths += 1
        
        # 处理episode结束
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.episode_count += 1
                
                # 如果环境提供了episode信息，确保它能被正确传递
                if 'episode' in info:
                    episode_info = info['episode'].copy()
                    
                    # 添加标准的SB3 episode信息字段
                    episode_info['r'] = float(self.episode_rewards[i])
                    episode_info['l'] = int(self.episode_lengths[i])
                    
                    # 确保episode信息在正确的位置
                    info['episode'] = episode_info
                    
                    if logger.isEnabledFor(10):  # DEBUG级别
                        logger.debug(f"环境{i} Episode {self.episode_count}结束")
                        logger.debug(f"包装器传递episode信息: {episode_info}")
                        if 'time_cost' in episode_info:
                            logger.debug(f"时间成本: {episode_info['time_cost']}")
                
                # 重置统计
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
        
        return observations, rewards, dones, infos
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.episode_rewards.fill(0)
        self.episode_lengths.fill(0)
        return self.venv.reset()