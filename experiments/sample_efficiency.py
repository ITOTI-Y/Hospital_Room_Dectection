"""Sample efficiency comparison experiments.

This module compares the sample efficiency of RL vs traditional algorithms:
- RL (PPO with flow-aware encoding)
- Simulated Annealing
- Greedy Local Search
- Random Search
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.baselines import GreedySwap, SimulatedAnnealing


def run_sample_efficiency_experiment(
    env: Any,
    rl_policy: Any,
    max_steps: int = 1000,
    n_runs: int = 10,
    save_dir: str | Path | None = None,
) -> dict[str, list[float]]:
    """Compare sample efficiency of different optimization methods.

    Args:
        env: LayoutEnv instance
        rl_policy: Trained RL policy
        max_steps: Maximum number of environment interactions
        n_runs: Number of independent runs for each method
        save_dir: Directory to save results and plots

    Returns:
        Dictionary containing results for each method
    """
    logger.info(
        f"Running sample efficiency experiment: {n_runs} runs, {max_steps} max steps"
    )

    # Prepare cost function for baselines
    def cost_function(permutation: list[int]) -> float:
        """Evaluate cost of a permutation."""
        # Map permutation to department IDs
        dept_ids = [env.index_to_dept_id[i] for i in permutation]

        # Calculate cost using environment's cost engine
        layout_dict = {
            slot_id: dept_id for slot_id, dept_id in zip(env.cost_engine.slots, dept_ids)
        }

        # Create temporary cost engine with this layout
        temp_cost = env.cost_engine.current_travel_cost
        return temp_cost

    # Initialize baseline algorithms
    sa_solver = SimulatedAnnealing(
        cost_fn=cost_function,
        n_items=env.num_total_slot,
        T_init=1000.0,
        alpha=0.995,
    )

    greedy_solver = GreedySwap(
        cost_fn=cost_function,
        n_items=env.num_total_slot,
    )

    # Storage for results
    results = {
        "RL": [],
        "SA": [],
        "Greedy": [],
        "steps": list(range(0, max_steps + 1, 10)),
    }

    # Run experiments
    for run in range(n_runs):
        logger.info(f"Run {run + 1}/{n_runs}")

        # Reset environment
        env.reset()

        # RL evaluation
        logger.info("  Evaluating RL...")
        rl_history = _evaluate_rl_policy(env, rl_policy, max_steps)
        results["RL"].append(rl_history)

        # Simulated Annealing
        logger.info("  Evaluating SA...")
        _, _, sa_history = sa_solver.solve(max_steps=max_steps)
        # Resample to match step intervals
        sa_resampled = _resample_history(sa_history, results["steps"])
        results["SA"].append(sa_resampled)

        # Greedy Search
        logger.info("  Evaluating Greedy...")
        _, _, greedy_history = greedy_solver.solve(max_steps=max_steps)
        greedy_resampled = _resample_history(greedy_history, results["steps"])
        results["Greedy"].append(greedy_resampled)

    # Plot results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_sample_efficiency(results, save_dir / "sample_efficiency.png")

    logger.info("Sample efficiency experiment complete")
    return results


def _evaluate_rl_policy(env: Any, policy: Any, max_steps: int) -> list[float]:
    """Evaluate RL policy and return cost history.

    Args:
        env: Environment
        policy: RL policy
        max_steps: Maximum steps

    Returns:
        List of best costs at each step
    """
    obs, _ = env.reset()
    best_cost = env.initial_cost
    history = [best_cost]

    for step in range(max_steps):
        # Get action from policy
        action, _, _ = policy.forward_actor(
            {k: np.expand_dims(v, 0) for k, v in obs.items()},
            deterministic=True,
        )

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action[0])

        # Track best cost
        current_cost = info.get("current_cost", env.current_cost)
        if current_cost < best_cost:
            best_cost = current_cost

        history.append(best_cost)

        if terminated or truncated:
            # Reset if episode ends
            obs, _ = env.reset()

    return _resample_history(history, list(range(0, max_steps + 1, 10)))


def _resample_history(history: list[float], target_steps: list[int]) -> list[float]:
    """Resample history to match target step intervals.

    Args:
        history: Original cost history
        target_steps: Target step indices

    Returns:
        Resampled history
    """
    resampled = []
    for step in target_steps:
        if step < len(history):
            resampled.append(history[step])
        else:
            resampled.append(history[-1])

    return resampled


def plot_sample_efficiency(
    results: dict[str, list[float]], save_path: str | Path
) -> None:
    """Plot sample efficiency comparison.

    Args:
        results: Results dictionary from experiment
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = results["steps"]
    colors = {"RL": "blue", "SA": "red", "Greedy": "green"}

    for method in ["RL", "SA", "Greedy"]:
        data = np.array(results[method])
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        ax.plot(steps, mean, label=method, color=colors[method], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=colors[method])

    ax.set_xlabel("Environment Interactions (Steps)", fontsize=12)
    ax.set_ylabel("Best Cost Found", fontsize=12)
    ax.set_title("Sample Efficiency Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Sample efficiency plot saved to {save_path}")


def compute_auc_efficiency(results: dict[str, list[float]]) -> dict[str, float]:
    """Compute area under curve as efficiency metric.

    Lower AUC = better (finds low cost faster).

    Args:
        results: Results dictionary

    Returns:
        Dictionary of AUC scores per method
    """
    auc_scores = {}
    steps = np.array(results["steps"])

    for method in ["RL", "SA", "Greedy"]:
        data = np.array(results[method])
        mean_costs = data.mean(axis=0)

        # Compute AUC using trapezoidal rule
        auc = np.trapz(mean_costs, steps)
        auc_scores[method] = auc

    return auc_scores
