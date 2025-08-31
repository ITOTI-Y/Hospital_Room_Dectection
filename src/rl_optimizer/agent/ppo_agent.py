# src/rl_optimizer/agent/ppo_agent.py

import torch
from pathlib import Path
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from typing import Dict, Any, List
import numpy as np
import time

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.model.policy_network import LayoutTransformer
from src.rl_optimizer.utils.setup import setup_logger, save_json
from src.rl_optimizer.utils.lr_scheduler import get_lr_scheduler
from src.rl_optimizer.utils.checkpoint_callback import CheckpointCallback

logger = setup_logger(__name__)

def get_action_mask_from_info(infos: List[Dict]) -> np.ndarray:
    """
    一个辅助函数，用于从矢量化环境的info字典列表中提取动作掩码。
    这是为了适配 `stable-baselines3` 的 `ActionMasker` 包装器。

    Args:
        infos (List[Dict]): 来自矢量化环境的info字典列表。

    Returns:
        np.ndarray: 合法的动作掩码堆叠成的数组。
    """
    return np.array([info.get("action_mask", []) for info in infos])

class PPOAgent:
    """
    PPO智能体，负责编排整个强化学习训练和评估流程。

    它使用stable-baselines3和sb3-contrib库来高效地实现MaskablePPO算法，
    该算法能够处理带有动作掩码的复杂决策问题。
    """

    def __init__(self, config: RLConfig, cache_manager: CacheManager, cost_calculator: CostCalculator):
        """
        初始化PPO智能体。

        Args:
            config (RLConfig): RL优化器的配置对象。
            cache_manager (CacheManager): 已初始化的数据缓存管理器。
            cost_calculator (CostCalculator): 已初始化的成本计算器。
        """
        self.config = config
        self.cm = cache_manager
        self.cc = cost_calculator

        # 将所有需要传递给环境的参数打包成一个字典
        self.env_kwargs = {
            "config": self.config,
            "cache_manager": self.cm,
            "cost_calculator": self.cc
        }

    def _check_for_resume(self) -> tuple[str, int]:
        """
        检查是否需要从checkpoint恢复训练。
        
        Returns:
            tuple: (模型路径, 已完成的训练步数)，如果不需要恢复则返回 (None, 0)
        """
        if not self.config.RESUME_TRAINING:
            return None, 0
            
        if self.config.PRETRAINED_MODEL_PATH:
            # 使用指定的预训练模型路径
            model_path = Path(self.config.PRETRAINED_MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"指定的预训练模型不存在: {model_path}")
                return None, 0
        else:
            # 自动查找最新的checkpoint
            checkpoint_callback = CheckpointCallback(
                save_freq=1,  # 临时值，仅用于查找checkpoint
                save_path=self.config.LOG_PATH
            )
            model_path = checkpoint_callback.get_latest_checkpoint()
            if not model_path:
                logger.info("未找到可用的checkpoint，将开始全新训练")
                return None, 0
                
        # 尝试加载checkpoint元数据获取训练进度
        metadata = CheckpointCallback.load_checkpoint_metadata(model_path)
        completed_steps = 0
        if metadata:
            completed_steps = metadata.get("training_progress", {}).get("num_timesteps", 0)
            logger.info(f"从checkpoint恢复训练，已完成步数: {completed_steps}")
        else:
            logger.warning("无法加载checkpoint元数据，从步数0开始")
            
        return str(model_path), completed_steps
        
    def _load_pretrained_model(self, model_path: str, vec_env) -> MaskablePPO:
        """
        加载预训练模型并准备继续训练。
        
        Args:
            model_path (str): 模型文件路径
            vec_env: 矢量化环境
            
        Returns:
            MaskablePPO: 加载的模型实例
        """
        logger.info(f"正在加载预训练模型: {model_path}")
        
        try:
            # 加载模型
            model = MaskablePPO.load(
                model_path,
                env=vec_env,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 重新设置学习率调度器（如果需要）
            if hasattr(model, 'lr_schedule'):
                lr_scheduler = get_lr_scheduler(
                    schedule_type=self.config.LEARNING_RATE_SCHEDULE_TYPE,
                    initial_lr=self.config.LEARNING_RATE_INITIAL,
                    final_lr=self.config.LEARNING_RATE_FINAL
                )
                model.lr_schedule = lr_scheduler
                logger.info("学习率调度器已重新设置")
                
            logger.info("预训练模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            raise

    def train(self):
        """
        执行PPO智能体的完整训练流程，包括环境创建、模型初始化、训练过程、评估回调和模型保存。
        
        支持断点续训：如果配置了RESUME_TRAINING=True，将自动检查并加载最新的checkpoint继续训练。
        训练过程中会自动创建并行矢量化环境，配置自定义特征提取器和网络结构，定期评估并保存最佳模型，最终保存完整训练后的模型至指定目录。支持手动中断训练并安全保存当前模型进度。
        """
        logger.info("开始配置和启动PPO训练流程...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.LOG_PATH / f"ppo_layout_{timestamp}"
        result_dir = self.config.RESULT_PATH / f"ppo_layout_{timestamp}"

        # 检查是否需要从checkpoint恢复
        resume_model_path, completed_steps = self._check_for_resume()
        if resume_model_path:
            logger.info(f"将从checkpoint恢复训练: {resume_model_path}")
            # 如果是恢复训练，使用原有的日志目录结构
            checkpoint_path = Path(resume_model_path)
            if "ppo_layout_" in checkpoint_path.parent.name:
                log_dir = checkpoint_path.parent
                result_dir = self.config.RESULT_PATH / checkpoint_path.parent.name
        
        remaining_steps = max(0, self.config.TOTAL_TIMESTEPS - completed_steps)
        if remaining_steps == 0:
            logger.info("训练已完成，无需继续训练")
            return

        # --- 1. 创建并行化的、支持掩码的矢量化环境 ---
        logger.info(f"正在创建 {self.config.NUM_ENVS} 个并行环境...")

        # 构建单个带掩码的环境
        vec_env = make_vec_env(
            lambda: ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn),
            n_envs=self.config.NUM_ENVS
        )

        logger.info("矢量化环境创建成功。")

        # --- 2. 配置PPO模型 ---
        model = None
        if resume_model_path:
            # 加载预训练模型
            model = self._load_pretrained_model(resume_model_path, vec_env)
            logger.info(f"剩余训练步数: {remaining_steps}")
        else:
            # 创建新模型
            logger.info("创建全新的PPO模型...")
            # 创建学习率调度器
            lr_scheduler = get_lr_scheduler(
                schedule_type=self.config.LEARNING_RATE_SCHEDULE_TYPE,
                initial_lr=self.config.LEARNING_RATE_INITIAL,
                final_lr=self.config.LEARNING_RATE_FINAL
            )
            logger.info(f"使用学习率调度器: {self.config.LEARNING_RATE_SCHEDULE_TYPE}")
            logger.info(f"初始学习率: {self.config.LEARNING_RATE_INITIAL}, 最终学习率: {self.config.LEARNING_RATE_FINAL}")
            
            # 定义策略网络的关键字参数，指定自定义的特征提取器
            policy_kwargs = {
                "features_extractor_class": LayoutTransformer,
                "features_extractor_kwargs": {
                    "config": self.config,
                    "features_dim": self.config.FEATURES_DIM
                },
                # 定义Actor和Critic网络的隐藏层结构
                "net_arch": [256, 256] 
            }

            logger.info("正在初始化MaskablePPO模型...")
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_scheduler,  # 使用学习率调度器
                n_steps=self.config.NUM_STEPS,
                batch_size=self.config.BATCH_SIZE,
                n_epochs=self.config.NUM_EPOCHS,
                gamma=self.config.GAMMA,
                gae_lambda=self.config.GAE_LAMBDA,
                clip_range=self.config.CLIP_RANGE,
                ent_coef=self.config.ENT_COEF,
                verbose=1,
                tensorboard_log=str(log_dir / 'tensorboard_logs'),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info(f"模型初始化成功，将在 {model.device.type} 设备上进行训练。")

        # --- 3. 配置回调函数 ---
        callbacks = []
        
        # 评估回调
        eval_env = ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(result_dir / 'best_model'),
            log_path=str(log_dir / 'eval_logs'),
            eval_freq=max(self.config.NUM_STEPS * 5, 2048),
            deterministic=True,
            render=False,
            n_eval_episodes=20 # 评估更多回合以获得更稳定的结果
        )
        callbacks.append(eval_callback)
        
        # Checkpoint回调
        if self.config.SAVE_TRAINING_STATE:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.CHECKPOINT_FREQUENCY,
                save_path=str(log_dir / 'checkpoints'),
                name_prefix="checkpoint",
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            logger.info(f"启用checkpoint保存，频率: 每{self.config.CHECKPOINT_FREQUENCY}步")

        # --- 4. 启动训练 ---
        logger.info(f"开始训练，剩余时间步数: {remaining_steps}...")
        logger.info(f"日志和模型将保存在: {log_dir}")
        try:
            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=not bool(resume_model_path)  # 如果是恢复训练则不重置计数器
            )
        except KeyboardInterrupt:
            logger.warning("训练被手动中断。")
        finally:
            # --- 5. 保存最终模型 ---
            final_model_path = log_dir / 'final_model.zip'
            model.save(final_model_path)
            logger.info(f"训练结束，最终模型已保存至: {final_model_path}")
            vec_env.close()

    def evaluate(self, model_path: str) -> Dict[str, Any]:
        """
        加载一个已训练好的模型，进行确定性评估并返回最优布局方案。

        Args:
            model_path (str): 已保存模型的路径 (.zip 文件)。

        Returns:
            Dict[str, Any]: 包含最优布局、总成本和各流程成本的结果字典。
        """
        logger.info(f"正在加载模型进行评估: {model_path}...")
        try:
            model = MaskablePPO.load(model_path, device='cpu')
        except FileNotFoundError:
            logger.error(f"模型文件未找到: {model_path}")
            return {}
        
        eval_env = LayoutEnv(**self.env_kwargs)

        obs, info = eval_env.reset()
        terminated = False

        while not terminated:
            action_mask = eval_env.get_action_mask()
            action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, terminated, _, info = eval_env.step(int(action))

        final_layout_str = eval_env._get_final_layout_str()
        total_cost = self.cc.calculate_total_cost(final_layout_str)
        per_process_cost = self.cc.calculate_per_process_cost(final_layout_str)

        # 构建最终的布局映射：槽位 -> 科室
        final_layout_map = {
            slot: dept for slot, dept in zip(eval_env.placeable_slots, final_layout_str)
            if dept is not None
        }

        results = {
            "best_layout": final_layout_map,
            "total_weighted_cost": total_cost,
            "per_process_unweighted_cost": per_process_cost,
            "final_reward_from_env": reward
        }

        result_path = self.config.LOG_PATH / 'best_layout_result.json'
        save_json(results, result_path)
        logger.info(f"评估完成！最优布局已保存至: {result_path}")
        logger.info(f"最优布局的总加权成本为: {total_cost:.2f}")

        return results
