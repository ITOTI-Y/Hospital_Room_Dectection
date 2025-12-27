"""Fast adaptation mechanisms for dynamic patient flow changes.

This module implements few-shot adaptation techniques that allow the model
to quickly adjust to changing patient flow patterns without full retraining.
"""

import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class FastAdaptationWrapper:
    """Fast adaptation wrapper for policy networks.

    When patient flow distribution changes, this wrapper enables quick policy
    adaptation through few-shot fine-tuning, avoiding expensive retraining.

    Key idea: Only fine-tune flow-related layers (attention, flow encoder),
    keeping static feature encoders frozen.

    Args:
        policy: Base policy network to adapt
        adaptation_lr: Learning rate for adaptation
        adaptation_steps: Number of gradient steps for adaptation
        freeze_static: Whether to freeze static feature layers
    """

    def __init__(
        self,
        policy: nn.Module,
        adaptation_lr: float = 1e-3,
        adaptation_steps: int = 10,
        freeze_static: bool = True,
    ):
        self.base_policy = policy
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.freeze_static = freeze_static

        # Save base policy parameters for reset
        self.base_state_dict = deepcopy(policy.state_dict())

        self.logger = logger.bind(module=__name__)

    def adapt(
        self,
        new_flow_matrix: torch.Tensor | np.ndarray,
        env: Any,
        n_episodes: int = 5,
        verbose: bool = True,
    ) -> nn.Module:
        """Quickly adapt policy to new flow distribution.

        Args:
            new_flow_matrix: New patient flow demand matrix
            env: Environment instance
            n_episodes: Number of episodes for adaptation
            verbose: Whether to log adaptation progress

        Returns:
            Adapted policy network
        """
        if verbose:
            self.logger.info(f"Starting fast adaptation with {n_episodes} episodes")

        # Convert to tensor if needed
        if isinstance(new_flow_matrix, np.ndarray):
            new_flow_matrix = torch.from_numpy(new_flow_matrix).float()

        # Create adapted policy from base
        adapted_policy = deepcopy(self.base_policy)
        adapted_policy.load_state_dict(self.base_state_dict)

        # Setup trainable parameters (only flow-related layers)
        trainable_params = self._get_trainable_params(adapted_policy)

        if len(trainable_params) == 0:
            self.logger.warning(
                "No flow-related parameters found! Adapting all parameters."
            )
            trainable_params = list(adapted_policy.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=self.adaptation_lr)

        # Adaptation loop
        total_reward = 0.0
        for ep in range(n_episodes):
            obs, _ = env.reset()

            # Inject new flow matrix into observation
            obs = self._inject_flow_matrix(obs, new_flow_matrix, env)

            episode_rewards = []
            episode_losses = []
            done = False
            step = 0

            while not done and step < 100:  # Max steps per episode
                # Get action from adapted policy
                with torch.enable_grad():
                    obs_tensor = self._prepare_obs(obs, adapted_policy.device)
                    action, log_prob, value = self._get_action_and_value(
                        adapted_policy, obs_tensor
                    )

                # Step environment
                obs_next, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy()
                )
                done = terminated or truncated

                # Simple policy gradient loss
                advantage = reward - value.item()
                loss = -log_prob * advantage

                episode_losses.append(loss.item())
                episode_rewards.append(reward)

                # Update policy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimizer.step()

                obs = self._inject_flow_matrix(obs_next, new_flow_matrix, env)
                step += 1

            total_reward += sum(episode_rewards)

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                avg_reward = np.mean(episode_rewards)
                avg_loss = np.mean(episode_losses)
                self.logger.info(
                    f"Adaptation Episode {ep + 1}/{n_episodes}: "
                    f"Avg Reward={avg_reward:.3f}, Avg Loss={avg_loss:.4f}"
                )

        if verbose:
            avg_total_reward = total_reward / n_episodes
            self.logger.info(
                f"Adaptation complete. Average episode reward: {avg_total_reward:.3f}"
            )

        return adapted_policy

    def _get_trainable_params(self, policy: nn.Module) -> list[nn.Parameter]:
        """Get parameters related to flow encoding for fine-tuning.

        Args:
            policy: Policy network

        Returns:
            List of trainable parameters
        """
        trainable_params = []

        for name, param in policy.named_parameters():
            # Freeze static layers if requested
            if self.freeze_static and any(
                keyword in name.lower()
                for keyword in ["dept_encoder", "fixable", "categorical"]
            ):
                param.requires_grad = False
                continue

            # Only fine-tune flow-related layers
            if any(
                keyword in name.lower()
                for keyword in ["flow", "attention", "cross_attention", "fusion"]
            ):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        return trainable_params

    def _inject_flow_matrix(
        self,
        obs: dict[str, np.ndarray],
        flow_matrix: torch.Tensor,
        env: Any,
    ) -> dict[str, np.ndarray]:
        """Inject new flow matrix into observation.

        Args:
            obs: Current observation dict
            flow_matrix: New flow matrix to inject
            env: Environment (for max_departments info)

        Returns:
            Updated observation with new flow matrix
        """
        obs = obs.copy()

        # Convert flow matrix to numpy and pad to max_departments
        flow_np = flow_matrix.cpu().numpy()
        max_depts = env.max_departments

        if "flow_matrix" not in obs:
            # Create padded flow matrix
            padded_flow = np.zeros((max_depts, max_depts), dtype=np.float32)
            n = min(flow_np.shape[0], max_depts)
            padded_flow[:n, :n] = flow_np[:n, :n]
            obs["flow_matrix"] = padded_flow
        else:
            # Update existing flow matrix
            n = min(flow_np.shape[0], max_depts)
            obs["flow_matrix"][:n, :n] = flow_np[:n, :n]

        return obs

    def _prepare_obs(
        self, obs: dict[str, np.ndarray], device: torch.device | None
    ) -> dict[str, torch.Tensor]:
        """Convert observation to tensors.

        Args:
            obs: Observation dict
            device: Target device

        Returns:
            Tensorized observation
        """
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            else:
                obs_tensor[key] = value

        return obs_tensor

    def _get_action_and_value(
        self, policy: nn.Module, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, and value from policy.

        Args:
            policy: Policy network
            obs: Observation tensors

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Forward pass
        value = policy(obs)

        # Get action
        action, log_prob, _ = policy.forward_actor(obs)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def reset_to_base(self):
        """Reset policy to base (pre-adaptation) state."""
        self.base_policy.load_state_dict(self.base_state_dict)
        self.logger.info("Policy reset to base state")


def benchmark_adaptation_speed(
    policy: nn.Module,
    env: Any,
    flow_perturbations: list[float],
    n_trials: int = 3,
) -> dict[str, list[float]]:
    """Benchmark adaptation speed vs performance trade-off.

    Args:
        policy: Base policy to adapt
        env: Environment instance
        flow_perturbations: List of perturbation strengths to test
        n_trials: Number of trials per perturbation

    Returns:
        Dict with adaptation times and final costs
    """
    wrapper = FastAdaptationWrapper(policy)
    results = {
        "perturbation": flow_perturbations,
        "adapt_time": [],
        "adapt_cost": [],
    }

    original_flow = env.cost_manager.pair_weights  # Get original flow

    for pert in flow_perturbations:
        times = []
        costs = []

        for _ in range(n_trials):
            # Perturb flow
            perturbed_flow = _perturb_flow_dict(original_flow, pert)

            # Time adaptation
            start = time.time()
            adapted_policy = wrapper.adapt(
                perturbed_flow, env, n_episodes=5, verbose=False
            )
            adapt_time = time.time() - start

            # Evaluate adapted policy
            final_cost = _evaluate_policy(adapted_policy, env, perturbed_flow)

            times.append(adapt_time)
            costs.append(final_cost)

        results["adapt_time"].append(np.mean(times))
        results["adapt_cost"].append(np.mean(costs))

    return results


def _perturb_flow_dict(
    flow_dict: dict[tuple[str, str], float], strength: float
) -> dict[tuple[str, str], float]:
    """Perturb flow dictionary by random noise.

    Args:
        flow_dict: Original flow dictionary
        strength: Perturbation strength (0-1)

    Returns:
        Perturbed flow dictionary
    """
    perturbed = {}
    for key, value in flow_dict.items():
        noise = np.random.uniform(1 - strength, 1 + strength)
        perturbed[key] = value * noise

    return perturbed


def _evaluate_policy(
    policy: nn.Module, env: Any, flow_dict: dict[tuple[str, str], float]
) -> float:
    """Evaluate policy on environment with given flow.

    Args:
        policy: Policy to evaluate
        env: Environment
        flow_dict: Flow dictionary

    Returns:
        Final cost achieved
    """
    # TODO: Implement proper evaluation logic
    # This is a placeholder
    return 0.0
