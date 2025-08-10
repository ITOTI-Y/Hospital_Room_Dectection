"""
PPO优化器 - 基于强化学习的布局优化算法
"""

import torch
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from src.algorithms.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.constraint_manager import ConstraintManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.env.vec_env_wrapper import EpisodeInfoVecEnvWrapper
from src.rl_optimizer.model.policy_network import LayoutTransformer
from src.rl_optimizer.utils.setup import setup_logger, save_json
from src.rl_optimizer.utils.lr_scheduler import get_lr_scheduler
from src.rl_optimizer.utils.checkpoint_callback import CheckpointCallback
from src.rl_optimizer.data.cache_manager import CacheManager
from src.config import RLConfig

logger = setup_logger(__name__)


def get_action_mask_from_info(infos: List[Dict]) -> np.ndarray:
    """从矢量化环境的info字典列表中提取动作掩码"""
    return np.array([info.get("action_mask", []) for info in infos])


class PPOOptimizer(BaseOptimizer):
    """
    PPO优化器
    
    基于强化学习的PPO算法实现布局优化，使用MaskablePPO来处理动作掩码约束。
    """
    
    def __init__(self, 
                 cost_calculator: CostCalculator,
                 constraint_manager: ConstraintManager,
                 config: RLConfig,
                 cache_manager: CacheManager):
        """
        初始化PPO优化器
        
        Args:
            cost_calculator: 成本计算器
            constraint_manager: 约束管理器
            config: RL配置
            cache_manager: 缓存管理器
        """
        super().__init__(cost_calculator, constraint_manager, "PPO")
        self.config = config
        self.cache_manager = cache_manager
        
        # 环境参数
        self.env_kwargs = {
            "config": self.config,
            "cache_manager": self.cache_manager,
            "cost_calculator": self.cost_calculator,
            "constraint_manager": self.constraint_manager
        }
        
        # 训练状态
        self.model = None
        self.vec_env = None
        self.resume_model_path = None
        self.completed_steps = 0
        self.best_model_dir = None  # 保存最佳模型目录路径
    
    def optimize(self, 
                 initial_layout: Optional[List[str]] = None,
                 max_iterations: int = None,
                 total_timesteps: int = None,
                 original_layout: Optional[List[str]] = None,
                 original_cost: Optional[float] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行PPO优化
        
        Args:
            initial_layout: 初始布局（PPO会自动探索）
            max_iterations: 最大迭代次数（使用total_timesteps代替）
            total_timesteps: 总训练步数
            original_layout: 原始布局（未经优化的基准）
            original_cost: 原始布局的成本
            **kwargs: 其他PPO参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        self.start_optimization()
        
        # 保存原始布局信息
        self.original_layout = original_layout
        self.original_cost = original_cost
        
        # 使用配置中的参数或传入的参数
        if total_timesteps is None:
            total_timesteps = self.config.TOTAL_TIMESTEPS
        
        logger.info(f"开始PPO优化，总训练步数: {total_timesteps}")
        
        try:
            # 检查是否需要恢复训练
            self._check_for_resume()
            
            # 计算剩余训练步数
            remaining_steps = max(0, total_timesteps - self.completed_steps)
            if remaining_steps == 0:
                logger.info("训练已完成，加载最佳模型进行评估")
                best_layout, best_cost = self._evaluate_best_model()
                self.update_best_solution(best_layout, best_cost)
                return self.finish_optimization()
            
            # 创建环境和模型
            self._setup_environment_and_model(remaining_steps)
            
            # 执行训练
            self._train_model(remaining_steps)
            
            # 评估最佳模型
            best_layout, best_cost = self._evaluate_best_model()
            self.update_best_solution(best_layout, best_cost)
            
        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
            if self.model:
                self._save_interrupted_model()
        except Exception as e:
            logger.error(f"PPO优化过程中发生错误: {e}", exc_info=True)
            raise
        
        return self.finish_optimization()
    
    def _check_for_resume(self):
        """检查是否需要从checkpoint恢复训练"""
        if not self.config.RESUME_TRAINING:
            return
            
        if self.config.PRETRAINED_MODEL_PATH:
            model_path = Path(self.config.PRETRAINED_MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"指定的预训练模型不存在: {model_path}")
                return
            self.resume_model_path = str(model_path)
        else:
            # 自动查找最新的checkpoint
            checkpoint_callback = CheckpointCallback(
                save_freq=1,
                save_path=self.config.LOG_PATH
            )
            model_path = checkpoint_callback.get_latest_checkpoint()
            if model_path:
                self.resume_model_path = str(model_path)
            else:
                logger.info("未找到可用的checkpoint，将开始全新训练")
                return
        
        # 加载checkpoint元数据
        metadata = CheckpointCallback.load_checkpoint_metadata(self.resume_model_path)
        if metadata:
            self.completed_steps = metadata.get("training_progress", {}).get("num_timesteps", 0)
            logger.info(f"从checkpoint恢复训练，已完成步数: {self.completed_steps}")
        else:
            logger.warning("无法加载checkpoint元数据，从步数0开始")
    
    def _setup_environment_and_model(self, remaining_steps: int):
        """设置环境和模型"""
        logger.info(f"正在创建 {self.config.NUM_ENVS} 个并行环境...")
        
        # 创建矢量化环境
        vec_env = make_vec_env(
            lambda: ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn),
            n_envs=self.config.NUM_ENVS
        )
        
        # 使用自定义包装器确保episode信息正确传递
        self.vec_env = EpisodeInfoVecEnvWrapper(vec_env)
        
        logger.info("矢量化环境创建成功，已添加episode信息包装器")
        
        # 创建或加载模型
        if self.resume_model_path:
            self._load_pretrained_model()
        else:
            self._create_new_model()
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        logger.info(f"正在加载预训练模型: {self.resume_model_path}")
        
        try:
            self.model = MaskablePPO.load(
                self.resume_model_path,
                env=self.vec_env,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 重新设置学习率调度器
            if hasattr(self.model, 'lr_schedule'):
                lr_scheduler = get_lr_scheduler(
                    schedule_type=self.config.LEARNING_RATE_SCHEDULE_TYPE,
                    initial_lr=self.config.LEARNING_RATE_INITIAL,
                    final_lr=self.config.LEARNING_RATE_FINAL
                )
                self.model.lr_schedule = lr_scheduler
                logger.info("学习率调度器已重新设置")
                
            logger.info("预训练模型加载成功")
            
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            raise
    
    def _create_new_model(self):
        """创建新的PPO模型"""
        logger.info("创建全新的PPO模型...")
        
        # 创建学习率调度器
        lr_scheduler = get_lr_scheduler(
            schedule_type=self.config.LEARNING_RATE_SCHEDULE_TYPE,
            initial_lr=self.config.LEARNING_RATE_INITIAL,
            final_lr=self.config.LEARNING_RATE_FINAL
        )
        
        logger.info(f"使用学习率调度器: {self.config.LEARNING_RATE_SCHEDULE_TYPE}")
        logger.info(f"初始学习率: {self.config.LEARNING_RATE_INITIAL}, 最终学习率: {self.config.LEARNING_RATE_FINAL}")
        
        # 定义策略网络参数
        policy_kwargs = {
            "features_extractor_class": LayoutTransformer,
            "features_extractor_kwargs": {
                "features_dim": self.config.FEATURES_DIM,  # 使用统一的配置属性
                "config": self.config
            },
            "net_arch": dict(pi=[self.config.POLICY_NET_ARCH] * self.config.POLICY_NET_LAYERS,
                            vf=[self.config.VALUE_NET_ARCH] * self.config.VALUE_NET_LAYERS)
        }
        
        # 创建PPO模型
        self.model = MaskablePPO(
            MaskableActorCriticPolicy,
            self.vec_env,
            learning_rate=lr_scheduler,
            n_steps=self.config.N_STEPS,
            batch_size=self.config.BATCH_SIZE,
            n_epochs=self.config.N_EPOCHS,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            clip_range=self.config.CLIP_RANGE,
            ent_coef=self.config.ENT_COEF,
            vf_coef=self.config.VF_COEF,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            tensorboard_log=str(self.config.LOG_PATH)
        )
        
        logger.info("PPO模型创建成功")
    
    def _train_model(self, remaining_steps: int):
        """训练模型"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.LOG_PATH / f"ppo_layout_{timestamp}"
        result_dir = self.config.RESULT_PATH / "model" / f"ppo_layout_{timestamp}"
        
        # 如果是恢复训练，使用原有目录
        if self.resume_model_path:
            checkpoint_path = Path(self.resume_model_path)
            if "ppo_layout_" in checkpoint_path.parent.name:
                log_dir = checkpoint_path.parent
                result_dir = self.config.RESULT_PATH / checkpoint_path.parent.name
        
        # 创建目录
        log_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳模型目录路径
        self.best_model_dir = result_dir
        
        # 设置回调
        callbacks = []
        
        
        # Checkpoint回调
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.CHECKPOINT_FREQUENCY,
            save_path=str(log_dir / "checkpoints"),
            name_prefix="checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # 评估回调
        eval_vec_env = make_vec_env(
            lambda: ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn),
            n_envs=1
        )
        eval_env = EpisodeInfoVecEnvWrapper(eval_vec_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(result_dir / "best_model"),
            log_path=str(log_dir / "eval_logs"),
            eval_freq=self.config.EVAL_FREQUENCY,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # 开始训练
        logger.info(f"开始训练，剩余步数: {remaining_steps}")
        logger.info(f"日志保存路径: {log_dir}")
        
        self.model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            tb_log_name="PPO",
            progress_bar=True,
            reset_num_timesteps=False if self.resume_model_path else True
        )
        
        # 训练完成
        logger.info("🎉 训练完成！")
        logger.info("=" * 80)
        
        # 保存最终模型
        final_model_path = log_dir / "final_model.zip"
        self.model.save(str(final_model_path))
        logger.info(f"最终模型已保存到: {final_model_path}")
        
        # 保存训练配置
        config_path = log_dir / "training_config.json"
        config_data = self.config.__dict__.copy()
        save_json(config_data, str(config_path))
        logger.info(f"训练配置已保存到: {config_path}")
    
    def _evaluate_best_model(self) -> tuple[List[str], float]:
        """评估最佳模型并返回最优布局和成本"""
        best_model_path = None
        
        # 1. 确定最佳模型路径
        if self.best_model_dir:
            best_model_path = self.best_model_dir / "best_model" / "best_model.zip"
            if not best_model_path.exists():
                logger.warning(f"最佳模型文件不存在: {best_model_path}")
                best_model_path = None
        
        # 2. 如果没有最佳模型，尝试使用最终模型
        if not best_model_path:
            final_model_paths = [
                self.config.LOG_PATH / f"ppo_layout_*/final_model.zip",
                self.config.LOG_PATH / "final_model.zip"
            ]
            
            for pattern in final_model_paths:
                import glob
                matches = glob.glob(str(pattern))
                if matches:
                    # 使用最新的模型
                    best_model_path = Path(max(matches, key=lambda x: Path(x).stat().st_mtime))
                    logger.info(f"使用最终模型作为备选: {best_model_path}")
                    break
        
        # 3. 如果仍然没有找到模型，返回默认布局
        if not best_model_path or not best_model_path.exists():
            logger.warning("未找到可用的训练模型，返回初始布局")
            best_layout = self.generate_initial_layout()
            best_cost = self.evaluate_layout(best_layout)
            return best_layout, best_cost
        
        try:
            # 4. 加载最佳模型
            logger.info(f"正在加载最佳模型: {best_model_path}")
            best_model = MaskablePPO.load(
                str(best_model_path),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 5. 创建评估环境（使用ActionMasker包装）
            from sb3_contrib.common.wrappers import ActionMasker
            eval_env = ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn)
            
            # 6. 执行评估（可以多次运行取最佳结果）
            best_layout = None
            best_cost = float('inf')
            n_eval_episodes = 5  # 评估5次取最佳
            
            for episode in range(n_eval_episodes):
                obs = eval_env.reset()
                # 处理新版Gym API返回的tuple
                if isinstance(obs, tuple):
                    obs = obs[0]
                terminated = False
                
                while not terminated:
                    # 获取动作掩码（从内部环境）
                    inner_env_for_mask = eval_env.env if hasattr(eval_env, 'env') else eval_env
                    action_mask = inner_env_for_mask.get_action_mask()
                    
                    # 使用掩码进行预测
                    action, _ = best_model.predict(obs, action_masks=action_mask, deterministic=True)
                    result = eval_env.step(int(action))
                    # 处理不同版本API的返回值
                    if len(result) == 5:
                        obs, _, terminated, _, _ = result
                    else:
                        obs, _, terminated, _ = result[:4]
                
                # 获取当前episode的布局和成本
                # ActionMasker包装了原始环境，需要访问内部环境
                inner_env = eval_env.env if hasattr(eval_env, 'env') else eval_env
                current_layout = inner_env._get_final_layout_str()
                current_cost = self.cost_calculator.calculate_total_cost(current_layout)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_layout = current_layout
                    logger.info(f"评估Episode {episode+1}/{n_eval_episodes}: 发现更优布局，成本: {best_cost:.2f}")
                else:
                    logger.info(f"评估Episode {episode+1}/{n_eval_episodes}: 成本: {current_cost:.2f}")
            
            logger.info(f"最佳模型评估完成，最优成本: {best_cost:.2f}")
            return best_layout, best_cost
            
        except Exception as e:
            logger.error(f"加载或评估模型时发生错误: {e}")
            logger.info("使用默认布局作为备选方案")
            best_layout = self.generate_initial_layout()
            best_cost = self.evaluate_layout(best_layout)
            return best_layout, best_cost
    
    def _save_interrupted_model(self):
        """保存被中断的模型"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        interrupted_path = self.config.LOG_PATH / f"interrupted_model_{timestamp}.zip"
        self.model.save(str(interrupted_path))
        logger.info(f"中断的模型已保存到: {interrupted_path}")
    
    def get_additional_metrics(self) -> Dict[str, Any]:
        """获取PPO特定的额外指标"""
        metrics = {
            "total_timesteps": self.config.TOTAL_TIMESTEPS,
            "completed_steps": self.completed_steps,
            "num_envs": self.config.NUM_ENVS,
            "learning_rate_schedule": self.config.LEARNING_RATE_SCHEDULE_TYPE,
            "resume_training": self.config.RESUME_TRAINING
        }
        
        if self.model is not None:
            metrics.update({
                "model_device": str(self.model.device),
                "policy_class": str(type(self.model.policy))
            })
        
        return metrics