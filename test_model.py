"""Test script to evaluate the best model on test cases."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tianshou.data import Batch

from src.config import config_loader
from src.rl.env import LayoutEnv
from src.rl.models.policy import LayoutA2CPolicy
from src.rl.models.ppo_model import LayoutOptimizationModel
from src.utils.logger import setup_logger

setup_logger()


def load_model(config, model_path: Path, device: torch.device) -> LayoutA2CPolicy:
    """Load trained model from checkpoint."""
    agent_cfg = config.agent

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
        device=device,
    )

    # Create a dummy optimizer for policy initialization
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create environment to get action space
    env = LayoutEnv(
        config=config,
        max_departments=agent_cfg.max_departments,
        max_step=agent_cfg.max_steps,
        is_training=False,
    )

    policy = LayoutA2CPolicy(
        model=model,
        optim=optim,
        action_space=env.action_space,
        discount_factor=agent_cfg.discount_factor,
        gae_lambda=agent_cfg.gae_lambda,
        vf_coef=agent_cfg.vf_coef,
        ent_coef=agent_cfg.ent_coef,
        max_grad_norm=agent_cfg.max_grad_norm,
        value_clip=agent_cfg.value_clip,
        advantage_normalization=agent_cfg.advantage_normalization,
        recompute_advantage=agent_cfg.recompute_advantage,
        dual_clip=agent_cfg.dual_clip,
        reward_normalization=agent_cfg.reward_normalization,
        eps_clip=agent_cfg.eps_clip,
        max_batchsize=agent_cfg.max_batchsize,
        deterministic_eval=True,  # Use deterministic for testing
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    logger.info(f"Model loaded from {model_path}")
    return policy, env


def run_test_episode(policy: LayoutA2CPolicy, env: LayoutEnv, case_id: int) -> dict:
    """Run a single test episode and record all swaps."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test Case #{case_id}")
    logger.info(f"{'=' * 60}")

    obs, info = env.reset()
    initial_cost = info["initial_cost"]

    logger.info(f"Initial cost: {initial_cost:.2f}")
    logger.info(f"Number of departments: {info['num_departments']}")
    logger.info("\nSwap Sequence:")
    logger.info(f"{'-' * 60}")

    swap_history = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        step += 1

        # Convert observation to Batch format for Tianshou
        obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        batch = Batch(obs=obs_batch)

        # Get action from policy
        with torch.no_grad():
            result = policy(batch)
            action = result.act[0].cpu().numpy()  # Get first (only) action

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Get department names
        idx1, idx2 = int(action[0]), int(action[1])

        if idx1 < env.num_total_slot and idx2 < env.num_total_slot:
            dept1 = env.index_to_dept_id.get(idx1, f"Index{idx1}")
            dept2 = env.index_to_dept_id.get(idx2, f"Index{idx2}")
        else:
            dept1 = f"Invalid({idx1})"
            dept2 = f"Invalid({idx2})"

        cost_change = (
            info.get("reward_components", {}).get("improvement", 0)
            * env.initial_cost
            / 100
        )

        swap_record = {
            "step": step,
            "dept1": dept1,
            "dept2": dept2,
            "current_cost": info["current_cost"],
            "reward": reward,
        }
        swap_history.append(swap_record)
        # Log swap
        cost_diff = (
            swap_history[-2]["current_cost"] - info["current_cost"]
            if len(swap_history) > 1
            else initial_cost - info["current_cost"]
        )
        logger.info(
            f"Step {step:2d}: {dept1:25s} <-> {dept2:25s} | "
            f"Cost: {info['current_cost']:8.1f} | Change: {cost_diff:+7.1f} | "
            f"Reward: {reward:+6.2f}"
        )

    # Calculate final results
    final_cost = info["current_cost"]
    improvement = (initial_cost - final_cost) / initial_cost * 100
    best_cost = env.best_cost
    best_improvement = (initial_cost - best_cost) / initial_cost * 100

    logger.info(f"{'-' * 60}")
    logger.info("\nResults Summary:")
    logger.info(f"  Initial cost:     {initial_cost:.2f}")
    logger.info(f"  Final cost:       {final_cost:.2f}")
    logger.info(f"  Best cost:        {best_cost:.2f}")
    logger.info(f"  Improvement:      {improvement:.2f}%")
    logger.info(f"  Best improvement: {best_improvement:.2f}%")
    logger.info(f"  Total reward:     {total_reward:.2f}")
    logger.info(f"  Total swaps:      {env.total_swaps}")
    logger.info(f"  Invalid swaps:    {env.invalid_swaps}")
    logger.info(f"  No-change swaps:  {env.no_change_swaps}")

    return {
        "case_id": case_id,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "best_cost": best_cost,
        "improvement": improvement,
        "best_improvement": best_improvement,
        "total_reward": total_reward,
        "total_swaps": env.total_swaps,
        "invalid_swaps": env.invalid_swaps,
        "no_change_swaps": env.no_change_swaps,
        "swap_history": swap_history,
    }


def main():
    """Main test function."""
    config = config_loader.ConfigLoader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load best model
    model_path = Path(config.paths.model_dir) / "best_ppo_layout_model.pth"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    policy, _ = load_model(config, model_path, device)

    # Run 2 test cases
    results = []
    for case_id in range(1, 3):
        # Create fresh environment for each test
        env = LayoutEnv(
            config=config,
            max_departments=config.agent.max_departments,
            max_step=config.agent.max_steps,
            is_training=False,
        )
        result = run_test_episode(policy, env, case_id)
        results.append(result)

    # Print overall summary
    logger.info(f"\n{'=' * 60}")
    logger.info("OVERALL TEST SUMMARY")
    logger.info(f"{'=' * 60}")

    avg_improvement = np.mean([r["improvement"] for r in results])
    avg_best_improvement = np.mean([r["best_improvement"] for r in results])
    avg_no_change = np.mean([r["no_change_swaps"] for r in results])

    for r in results:
        logger.info(
            f"Case #{r['case_id']}: "
            f"Improvement={r['improvement']:.2f}%, "
            f"Best={r['best_improvement']:.2f}%, "
            f"No-change={r['no_change_swaps']}"
        )

    logger.info("\nAverages:")
    logger.info(f"  Improvement:      {avg_improvement:.2f}%")
    logger.info(f"  Best improvement: {avg_best_improvement:.2f}%")
    logger.info(f"  No-change swaps:  {avg_no_change:.1f}")


if __name__ == "__main__":
    main()
