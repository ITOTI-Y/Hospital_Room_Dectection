import torch
from pathlib import Path
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

from src.pipeline import PathwayGenerator, CostManager
from src.config.config_loader import ConfigLoader
from src.rl.env import LayoutEnv
from src.rl.models.ppo_model import LayoutOptimizationModel
from src.rl.models.policy import LayoutPolicy


class OptimizeManager:
    def __init__(self, config: ConfigLoader, **kwargs):
        self.logger = logger.bind(module=__name__)
        self.config = config
        self.kwargs = kwargs

        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config, is_shuffle=True)
        self.max_departments = self.config.agent.max_departments
        self.max_steps = self.config.agent.max_steps

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
        )

        self.logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
        )
        return model

    def run(self):

        self.logger.info("Starting PPO training...")

        train_envs, test_envs = self.create_env()

        model = self.create_model()

        optim = torch.optim.Adam(
            model.parameters(),
            lr=self.config.agent.lr,
        )

        policy = LayoutPolicy(
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

        train_collector = Collector(
            policy=policy,
            env=train_envs,
            buffer=VectorReplayBuffer(
                total_size=self.config.agent.buffer_size,
                buffer_num=len(train_envs),
            ),
            exploration_noise=False,
        )

        test_collector = Collector(
            policy=policy,
            env=test_envs,
        )

        writer = SummaryWriter(
            log_dir=self.config.paths.tensorboard_dir
        )

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
            )

            result = trainer.run()
        except Exception:
            self.logger.exception("Training failed with error")
            raise

        self.logger.info(
            f"Training completed!, best reward: {result.best_reward}")

        self.save(policy=policy, file_name="final_ppo_layout_model.pth")

        return trainer

    def save_best_model(self, policy):
        best_model_path = Path(self.config.paths.model_dir) / \
            "best_ppo_layout_model.pth"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), best_model_path)
        self.logger.info(f"best model saved to {best_model_path}")

    def save(self, policy: torch.nn.Module, file_name: str):

        model_path = Path(self.config.paths.model_dir) / file_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
