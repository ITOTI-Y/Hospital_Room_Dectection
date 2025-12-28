from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from src.config.config_loader import ConfigLoader
from src.pipeline import CostManager, PathwayGenerator
from src.rl.env import LayoutEnv
from src.rl.models.policy import LayoutA2CPolicy
from src.rl.models.ppo_model import LayoutOptimizationModel


class OptimizeManager:
    def __init__(self, config: ConfigLoader, **kwargs):
        self.logger = logger.bind(module=__name__)
        self.config = config
        self.kwargs = kwargs

        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config, is_shuffle=True)
        self.max_departments = self.config.agent.max_departments
        self.max_steps = self.config.agent.max_steps
        self.checkpoint_interval = 10  # Save checkpoint every N epochs

        # Early stopping and best model tracking based on actual improvement
        self.best_actual_improvement = -float("inf")
        self.best_improvement_epoch = 0
        self.improvement_history: list[float] = []
        self.early_stop_patience = getattr(
            self.config.agent, "early_stop_patience", 20
        )  # Stop if no improvement for N epochs
        self.eval_episodes = getattr(
            self.config.agent, "eval_episodes", 10
        )  # Number of episodes for evaluation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def create_env(self):
        train_envs = DummyVectorEnv(
            [
                lambda: LayoutEnv(
                    config=self.config,
                    max_departments=self.max_departments,
                    max_step=self.max_steps,
                )
                for _ in range(self.config.agent.num_train_envs)
            ]
        )

        test_envs = DummyVectorEnv(
            [
                lambda: LayoutEnv(
                    config=self.config,
                    max_departments=self.max_departments,
                    max_step=self.max_steps,
                    is_training=False,
                )
                for _ in range(self.config.agent.num_test_envs)
            ]
        )

        return train_envs, test_envs

    def create_model(self):
        agent_cfg = self.config.agent
        model = LayoutOptimizationModel(
            num_categories=agent_cfg.max_departments,
            embedding_dim=agent_cfg.embedding_dim,
            numerical_feat_dim=agent_cfg.numerical_feat_dim,
            numerical_hidden_dim=agent_cfg.numerical_hidden_dim,
            gnn_hidden_dims=agent_cfg.gnn_hidden_dims,
            gnn_output_dim=agent_cfg.gnn_output_dim,
            gnn_num_layers=agent_cfg.gnn_num_layers,
            gnn_dropout=agent_cfg.gnn_dropout,
            actor_hidden_dim=agent_cfg.actor_hidden_dim,
            actor_dropout=agent_cfg.actor_dropout,
            value_hidden_dim=agent_cfg.value_hidden_dim,
            value_num_layers=agent_cfg.value_num_layers,
            value_pooling_type=agent_cfg.value_pooling_type,
            value_dropout=agent_cfg.value_dropout,
            device=self.device,
            # Flow-aware encoding parameters
            use_flow_aware=getattr(agent_cfg, "use_flow_aware", False),
            flow_attention_heads=getattr(agent_cfg, "flow_attention_heads", 4),
        )

        encoder_type = "FlowAwareGCN" if getattr(agent_cfg, "use_flow_aware", False) else "GCN"
        self.logger.info(
            f"Model created with {encoder_type} encoder, "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
        )
        return model

    def create_scheduler(self, optim: torch.optim.Optimizer):
        """Create learning rate scheduler based on config."""
        agent_cfg = self.config.agent
        scheduler_type = getattr(agent_cfg, "lr_scheduler", "none")

        if scheduler_type == "step":
            step_size = getattr(agent_cfg, "lr_decay_step", 20)
            gamma = getattr(agent_cfg, "lr_decay_gamma", 0.5)
            scheduler = StepLR(optim, step_size=step_size, gamma=gamma)
            self.logger.info(
                f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}"
            )
        elif scheduler_type == "cosine":
            lr_min = getattr(agent_cfg, "lr_min", 1e-6)
            scheduler = CosineAnnealingLR(
                optim, T_max=agent_cfg.max_epoch, eta_min=lr_min
            )
            self.logger.info(
                f"Using CosineAnnealingLR scheduler: T_max={agent_cfg.max_epoch}, eta_min={lr_min}"
            )
        else:
            scheduler = None
            self.logger.info("No learning rate scheduler enabled")

        return scheduler

    def run(self):
        self.logger.info("Starting PPO training with actual improvement tracking...")
        self.logger.info(
            f"Early stopping patience: {self.early_stop_patience} epochs, "
            f"Eval episodes: {self.eval_episodes}"
        )

        train_envs, test_envs = self.create_env()

        model = self.create_model()

        optim = torch.optim.Adam(
            model.parameters(),
            lr=self.config.agent.lr,
        )

        # Create learning rate scheduler
        scheduler = self.create_scheduler(optim)

        # Store policy reference for checkpoint saving
        self._current_policy = None
        self._current_epoch = 0

        # Create train_fn callback for scheduler step, epoch tracking, and checkpoints
        def train_fn(epoch: int, env_step: int):
            self._current_epoch = epoch

            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                self.logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.2e}")

            # Save periodic checkpoint
            if (
                epoch > 0
                and epoch % self.checkpoint_interval == 0
                and self._current_policy is not None
            ):
                self.save_checkpoint(self._current_policy, epoch)

        # Create stop_fn for early stopping based on actual improvement
        def stop_fn(mean_rewards: float) -> bool:
            return self.check_early_stop(self._current_epoch, 0)

        policy = LayoutA2CPolicy(
            model=model,
            optim=optim,
            action_space=train_envs.workers[0].action_space,
            discount_factor=self.config.agent.discount_factor,
            gae_lambda=self.config.agent.gae_lambda,
            vf_coef=self.config.agent.vf_coef,
            ent_coef=self.config.agent.ent_coef,
            max_grad_norm=self.config.agent.max_grad_norm,
            value_clip=self.config.agent.value_clip,
            advantage_normalization=self.config.agent.advantage_normalization,
            recompute_advantage=self.config.agent.recompute_advantage,
            dual_clip=self.config.agent.dual_clip,
            reward_normalization=self.config.agent.reward_normalization,
            eps_clip=self.config.agent.eps_clip,
            max_batchsize=self.config.agent.max_batchsize,
            deterministic_eval=self.config.agent.deterministic_eval,
        )

        # Store policy reference for checkpoint saving in train_fn
        self._current_policy = policy

        train_collector = Collector(
            policy=policy,
            env=train_envs,
            buffer=VectorReplayBuffer(
                total_size=self.config.agent.buffer_size,
                buffer_num=train_envs.env_num,
            ),
            exploration_noise=False,
        )

        test_collector = Collector(
            policy=policy,
            env=test_envs,
        )

        writer = SummaryWriter(log_dir=self.config.paths.tensorboard_dir)

        tensorboard_logger = TensorboardLogger(writer)

        try:
            trainer = OnpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                max_epoch=self.config.agent.max_epoch,
                batch_size=self.config.agent.batch_size,
                step_per_epoch=self.config.agent.step_per_epoch,
                step_per_collect=self.config.agent.step_per_collect,
                repeat_per_collect=self.config.agent.repeat_per_collect,
                episode_per_test=self.config.agent.episode_per_test,
                logger=tensorboard_logger,
                verbose=True,
                show_progress=True,
                save_best_fn=self.save_best_model,
                train_fn=train_fn,
                stop_fn=stop_fn,
            )

            result = trainer.run()
        except Exception:
            self.logger.exception("Training failed with error")
            raise

        self.logger.info(
            f"Training completed! Best actual improvement: {self.best_actual_improvement:.2f}% "
            f"at epoch {self.best_improvement_epoch}"
        )
        self.save(policy=policy, file_name="final_ppo_layout_model.pth")
        writer.flush()
        writer.close()

        return trainer

    def evaluate_actual_improvement(
        self, policy: LayoutA2CPolicy, num_episodes: int | None = None
    ) -> tuple[float, float]:
        """Evaluate policy on test episodes and return actual layout improvement.

        Args:
            policy: The policy to evaluate
            num_episodes: Number of episodes to run (default: self.eval_episodes)

        Returns:
            mean_improvement: Average improvement rate (%)
            std_improvement: Standard deviation of improvement rate
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        policy.eval()
        improvements = []

        for _ in range(num_episodes):
            env = LayoutEnv(
                config=self.config,
                max_departments=self.max_departments,
                max_step=self.max_steps,
                is_training=False,
            )
            obs, info = env.reset()
            initial_cost = info["initial_cost"]
            done = False

            while not done:
                obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
                batch = Batch(obs=obs_batch)
                with torch.no_grad():
                    result = policy(batch)
                    action = result.act[0].cpu().numpy()
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            final_cost = info["current_cost"]
            improvement = (initial_cost - final_cost) / initial_cost * 100
            improvements.append(improvement)

        mean_improvement = float(np.mean(improvements))
        std_improvement = float(np.std(improvements))

        policy.train()
        return mean_improvement, std_improvement

    def save_best_model(self, policy):
        """Save model only if actual improvement is better than previous best.

        This method is called by Tianshou when reward improves, but we override
        the logic to use actual layout improvement instead.
        """
        # Evaluate actual improvement
        mean_improvement, std_improvement = self.evaluate_actual_improvement(policy)
        self._current_epoch = getattr(self, "_current_epoch", 0)

        self.logger.info(
            f"Epoch {self._current_epoch}: Actual improvement = "
            f"{mean_improvement:.2f}% ± {std_improvement:.2f}%"
        )

        # Track improvement history
        self.improvement_history.append(mean_improvement)

        # Save if this is the best improvement so far
        if mean_improvement > self.best_actual_improvement:
            self.best_actual_improvement = mean_improvement
            self.best_improvement_epoch = self._current_epoch

            best_model_path = (
                Path(self.config.paths.model_dir) / "best_ppo_layout_model.pth"
            )
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), best_model_path)
            self.logger.info(
                f"New best model saved! Improvement: {mean_improvement:.2f}% "
                f"(epoch {self._current_epoch})"
            )
        else:
            epochs_since_best = self._current_epoch - self.best_improvement_epoch
            self.logger.info(
                f"No improvement over best ({self.best_actual_improvement:.2f}% "
                f"at epoch {self.best_improvement_epoch}). "
                f"Patience: {epochs_since_best}/{self.early_stop_patience}"
            )

    def check_early_stop(self, epoch: int, env_step: int) -> bool:
        """Check if training should stop based on actual improvement.

        Args:
            epoch: Current epoch number
            env_step: Current environment step

        Returns:
            True if training should stop, False otherwise
        """
        if epoch < 5:  # Don't stop too early
            return False

        epochs_since_best = epoch - self.best_improvement_epoch
        should_stop = epochs_since_best >= self.early_stop_patience

        if should_stop:
            self.logger.warning(
                f"Early stopping triggered! No improvement for {epochs_since_best} epochs. "
                f"Best improvement: {self.best_actual_improvement:.2f}% at epoch {self.best_improvement_epoch}"
            )

        return should_stop

    def save_checkpoint(self, policy, epoch: int):
        """Save a checkpoint at the given epoch."""
        checkpoint_dir = Path(self.config.paths.model_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(policy.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def save(self, policy: torch.nn.Module, file_name: str):
        model_path = Path(self.config.paths.model_dir) / file_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
