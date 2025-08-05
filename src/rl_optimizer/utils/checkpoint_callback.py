# src/rl_optimizer/utils/checkpoint_callback.py

import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)


class CheckpointCallback(BaseCallback):
    """
    自定义checkpoint回调，用于定期保存完整的训练状态。
    
    支持保存：
    - 模型参数
    - 优化器状态
    - 学习率调度器状态
    - 训练元数据（步数、时间等）
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "checkpoint",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 1
    ):
        """
        初始化checkpoint回调。
        
        Args:
            save_freq (int): 每多少训练步保存一次checkpoint
            save_path (str): checkpoint保存目录
            name_prefix (str): checkpoint文件名前缀
            save_replay_buffer (bool): 是否保存replay buffer（如果有）
            save_vecnormalize (bool): 是否保存VecNormalize状态（如果有）
            verbose (int): 日志详细级别
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        
        # 确保保存目录存在
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 训练状态跟踪
        self.start_time = time.time()
        self.checkpoint_count = 0
        
    def _init_callback(self) -> None:
        """初始化回调时的操作。"""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self) -> bool:
        """每个训练步后的回调。"""
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        return True
        
    def _save_checkpoint(self) -> None:
        """保存完整的checkpoint。"""
        self.checkpoint_count += 1
        checkpoint_name = f"{self.name_prefix}_{self.n_calls:08d}_steps"
        checkpoint_path = self.save_path / checkpoint_name
        
        if self.verbose >= 1:
            logger.info(f"正在保存checkpoint: {checkpoint_name}")
            
        try:
            # 1. 保存模型
            model_path = checkpoint_path.with_suffix('.zip')
            self.model.save(str(model_path))
            
            # 2. 保存训练元数据
            metadata = self._collect_metadata()
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            # 3. 保存训练状态（如果启用）
            if hasattr(self.model, 'get_training_state'):
                state_path = checkpoint_path.with_suffix('.pkl')
                training_state = self.model.get_training_state()
                with open(state_path, 'wb') as f:
                    pickle.dump(training_state, f)
                    
            # 4. 保存VecNormalize状态（如果有）
            if self.save_vecnormalize and hasattr(self.training_env, 'normalize_obs'):
                vecnorm_path = checkpoint_path.with_suffix('.vecnorm.pkl')
                self.training_env.save(str(vecnorm_path))
                
            if self.verbose >= 1:
                logger.info(f"Checkpoint保存成功: {checkpoint_name}")
                
        except Exception as e:
            logger.error(f"保存checkpoint时发生错误: {e}", exc_info=True)
            
    def _collect_metadata(self) -> Dict[str, Any]:
        """收集训练元数据。"""
        current_time = time.time()
        training_duration = current_time - self.start_time
        
        metadata = {
            "checkpoint_info": {
                "timestamp": current_time,
                "checkpoint_count": self.checkpoint_count,
                "training_duration_seconds": training_duration,
                "training_duration_hours": training_duration / 3600,
            },
            "training_progress": {
                "n_calls": self.n_calls,
                "num_timesteps": self.model.num_timesteps,
                "total_timesteps": getattr(self.model, '_total_timesteps', None),
                "progress_percent": (
                    self.model.num_timesteps / getattr(self.model, '_total_timesteps', 1) * 100
                    if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0
                    else 0
                ),
            },
            "model_info": {
                "algorithm": self.model.__class__.__name__,
                "learning_rate": self.model.learning_rate,
                "n_envs": getattr(self.model.env, 'num_envs', 1),
            }
        }
        
        # 添加学习率调度器信息（如果有）
        if hasattr(self.model, 'lr_schedule'):
            try:
                current_lr = self.model.lr_schedule(self.model.num_timesteps / 
                                                  getattr(self.model, '_total_timesteps', self.model.num_timesteps))
                metadata["model_info"]["current_learning_rate"] = float(current_lr)
            except:
                pass
                
        return metadata
        
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的checkpoint路径。"""
        if not self.save_path.exists():
            return None
            
        checkpoint_files = list(self.save_path.glob(f"{self.name_prefix}_*_steps.zip"))
        if not checkpoint_files:
            return None
            
        # 按文件名中的步数排序，获取最新的
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-2]))
        return checkpoint_files[-1]
        
    @staticmethod
    def load_checkpoint_metadata(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """加载checkpoint的元数据。"""
        metadata_path = checkpoint_path.with_suffix('.json')
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载checkpoint元数据失败: {e}")
            return None