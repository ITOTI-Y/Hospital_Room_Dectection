"""
Fast Adaptation Module for quick policy updates when patient flow changes.

Key idea: Instead of retraining from scratch when flow patterns change,
we fine-tune only the flow-related layers with a few episodes.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Any
from loguru import logger


class FastAdaptationWrapper:
    """
    Fast adaptation wrapper for few-shot policy updates.

    When patient flow distribution changes, this module enables quick adaptation
    through selective fine-tuning of flow-related parameters.
    """

    def __init__(
        self,
        policy: nn.Module,
        adaptation_lr: float = 1e-3,
        adaptation_steps: int = 10,
        freeze_pattern: str = "flow|attention",
    ):
        """
        Args:
            policy: The policy network to adapt
            adaptation_lr: Learning rate for adaptation
            adaptation_steps: Number of gradient steps per episode
            freeze_pattern: Regex pattern for layers to keep trainable
                           (others will be frozen during adaptation)
        """
        self.base_policy = policy
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.freeze_pattern = freeze_pattern

        # Save base policy parameters
        self.base_state_dict = deepcopy(policy.state_dict())

        self.logger = logger.bind(module=__name__)

    def adapt(
        self,
        new_flow_matrix: torch.Tensor,
        env: Any,
        n_episodes: int = 5,
        return_metrics: bool = False,
    ) -> nn.Module | tuple[nn.Module, dict]:
        """
        Quickly adapt policy to new flow distribution.

        Args:
            new_flow_matrix: New patient flow matrix (max_depts, max_depts)
            env: Environment instance
            n_episodes: Number of episodes for adaptation
            return_metrics: Whether to return adaptation metrics

        Returns:
            Adapted policy (and optionally metrics dict)
        """
        import re

        # Clone policy for adaptation
        adapted_policy = deepcopy(self.base_policy)
        adapted_policy.load_state_dict(self.base_state_dict)

        # Identify trainable parameters (flow-related layers)
        trainable_params = []
        frozen_count = 0
        trainable_count = 0

        for name, param in adapted_policy.named_parameters():
            if re.search(self.freeze_pattern, name.lower()):
                param.requires_grad = True
                trainable_params.append(param)
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        self.logger.info(
            f"Adaptation: {trainable_count} trainable params, {frozen_count} frozen"
        )

        if not trainable_params:
            self.logger.warning("No trainable parameters found for adaptation!")
            if return_metrics:
                return adapted_policy, {"episodes": 0, "final_reward": 0}
            return adapted_policy

        optimizer = torch.optim.Adam(trainable_params, lr=self.adaptation_lr)

        # Adaptation loop
        metrics = {
            "episode_rewards": [],
            "episode_improvements": [],
        }

        for ep in range(n_episodes):
            # Reset with new flow matrix
            obs, info = env.reset(options={"flow_matrix": new_flow_matrix.cpu().numpy()})

            episode_reward = 0
            episode_log_probs = []
            episode_rewards = []
            done = False

            adapted_policy.train()

            while not done:
                # Convert observation to tensor
                obs_tensor = self._obs_to_tensor(obs, adapted_policy)

                # Get action from policy
                with torch.enable_grad():
                    action, log_prob = self._get_action(adapted_policy, obs_tensor)

                # Step environment
                obs_next, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_reward += reward

                obs = obs_next

            # REINFORCE update
            returns = self._compute_returns(episode_rewards, gamma=0.99)
            policy_loss = 0

            for log_prob, R in zip(episode_log_probs, returns):
                policy_loss -= log_prob * R

            if isinstance(policy_loss, torch.Tensor):
                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimizer.step()

            # Track metrics
            improvement = (info.get("initial_cost", 1) - info.get("best_cost", 1)) / (
                info.get("initial_cost", 1) + 1e-6
            )
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_improvements"].append(improvement * 100)

            self.logger.debug(
                f"Adaptation episode {ep + 1}/{n_episodes}: "
                f"reward={episode_reward:.2f}, improvement={improvement * 100:.2f}%"
            )

        adapted_policy.eval()

        if return_metrics:
            metrics["final_reward"] = metrics["episode_rewards"][-1] if metrics["episode_rewards"] else 0
            metrics["final_improvement"] = metrics["episode_improvements"][-1] if metrics["episode_improvements"] else 0
            return adapted_policy, metrics

        return adapted_policy

    def _obs_to_tensor(self, obs: dict, policy: nn.Module) -> dict:
        """Convert numpy observation to tensor."""
        device = next(policy.parameters()).device
        return {
            k: torch.from_numpy(v).unsqueeze(0).to(device) if hasattr(v, '__array__') else v
            for k, v in obs.items()
        }

    def _get_action(
        self, policy: nn.Module, obs: dict
    ) -> tuple[list, torch.Tensor]:
        """Get action from policy with log probability."""
        # This should be implemented based on your policy's interface
        # Here's a generic implementation
        if hasattr(policy, "get_action"):
            return policy.get_action(obs)
        else:
            # Fallback: use forward pass
            logits = policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.cpu().numpy().tolist(), log_prob

    def _compute_returns(self, rewards: list, gamma: float = 0.99) -> list:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def reset_to_base(self) -> None:
        """Reset policy to base (pre-adaptation) state."""
        self.base_policy.load_state_dict(self.base_state_dict)

    def update_base(self) -> None:
        """Update base state to current policy state."""
        self.base_state_dict = deepcopy(self.base_policy.state_dict())


class MetaAdaptationWrapper(FastAdaptationWrapper):
    """
    Meta-learning inspired adaptation using MAML-like approach.

    Learns initialization that is amenable to fast adaptation,
    requiring even fewer episodes for good performance.
    """

    def __init__(
        self,
        policy: nn.Module,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-4,
        inner_steps: int = 5,
    ):
        super().__init__(policy, adaptation_lr=inner_lr)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        # Outer optimizer for meta-learning
        self.meta_optimizer = torch.optim.Adam(
            self.base_policy.parameters(), lr=outer_lr
        )

    def meta_train_step(
        self,
        task_batch: list[torch.Tensor],
        env: Any,
    ) -> float:
        """
        One step of meta-training across multiple flow scenarios.

        Args:
            task_batch: List of flow matrices representing different scenarios
            env: Environment instance

        Returns:
            Average meta-loss across tasks
        """
        meta_loss = 0

        for flow_matrix in task_batch:
            # Inner loop: adapt to this task
            adapted_policy, metrics = self.adapt(
                flow_matrix, env, n_episodes=self.inner_steps, return_metrics=True
            )

            # Outer loop: evaluate on same task after adaptation
            obs, info = env.reset(options={"flow_matrix": flow_matrix.cpu().numpy()})
            episode_reward = 0
            done = False

            while not done:
                obs_tensor = self._obs_to_tensor(obs, adapted_policy)
                action, log_prob = self._get_action(adapted_policy, obs_tensor)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            # Meta-loss is negative reward (we want to maximize)
            meta_loss -= episode_reward

        # Meta-update
        meta_loss = meta_loss / len(task_batch)

        self.meta_optimizer.zero_grad()
        if isinstance(meta_loss, torch.Tensor):
            meta_loss.backward()
            self.meta_optimizer.step()

        # Update base state after meta-update
        self.update_base()

        return meta_loss.item() if isinstance(meta_loss, torch.Tensor) else meta_loss


def perturb_flow_matrix(
    flow: torch.Tensor,
    strength: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Perturb flow matrix to simulate patient flow changes.

    Args:
        flow: Original flow matrix (n_depts, n_depts)
        strength: Perturbation strength (0.0 = no change, 1.0 = ±100% change)
        seed: Random seed for reproducibility

    Returns:
        Perturbed flow matrix (symmetric, non-negative)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Multiplicative noise
    noise = torch.empty_like(flow).uniform_(1 - strength, 1 + strength)
    perturbed = flow * noise

    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2

    # Ensure non-negative
    perturbed = torch.clamp(perturbed, min=0)

    # Re-normalize
    max_val = perturbed.max()
    if max_val > 0:
        perturbed = perturbed / max_val

    return perturbed
