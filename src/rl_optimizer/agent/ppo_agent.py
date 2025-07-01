# src/rl_optimizer/agent/ppo_agent.py

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from typing import Dict, Any, List, Callable
import numpy as np
import time

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.model.policy_network import LayoutTransformer
from src.rl_optimizer.utils.setup import setup_logger, save_json

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

    def train(self):
        """
        启动并执行完整的训练流程。
        """
        logger.info("开始配置和启动PPO训练流程...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.LOG_PATH / f"ppo_layout_{timestamp}"
        result_dir = self.config.RESULT_PATH / f"ppo_layout_{timestamp}"

        # --- 1. 创建并行化的、支持掩码的矢量化环境 ---
        logger.info(f"正在创建 {self.config.NUM_ENVS} 个并行环境...")

        # 构建单个带掩码的环境
        vec_env = make_vec_env(
            lambda: ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn),
            n_envs=self.config.NUM_ENVS
        )

        logger.info("矢量化环境创建成功。")

        # --- 2. 配置PPO模型 ---
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

        # 定义评估回调函数，用于在训练过程中定期评估模型并保存最优模型
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

        logger.info("正在初始化MaskablePPO模型...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=self.config.LEARNING_RATE,
            n_steps=self.config.NUM_STEPS,
            batch_size=self.config.BATCH_SIZE,
            n_epochs=self.config.NUM_EPOCHS,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            clip_range=self.config.CLIP_EPS,
            ent_coef=self.config.ENT_COEF,
            verbose=1,
            tensorboard_log=str(log_dir / 'tensorboard_logs'),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"模型初始化成功，将在 {model.device.type} 设备上进行训练。")

        # --- 3. 启动训练 ---
        logger.info(f"开始训练，总时间步数: {self.config.TOTAL_TIMESTEPS}...")
        logger.info(f"日志和模型将保存在: {log_dir}")
        try:
            model.learn(
                total_timesteps=self.config.TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=True
            )
        except KeyboardInterrupt:
            logger.warning("训练被手动中断。")
        finally:
            # --- 4. 保存最终模型 ---
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
