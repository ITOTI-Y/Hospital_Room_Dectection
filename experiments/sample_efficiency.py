"""
Sample Efficiency Experiment

Compares RL agent against traditional optimization baselines:
- Simulated Annealing (SA)
- Greedy Swap
- Random Search
- Tabu Search

Measures: Steps to reach target improvement, final solution quality
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable
from loguru import logger

from src.baselines.traditional import (
    SimulatedAnnealing,
    GreedySwap,
    RandomSearch,
    TabuSearch,
)


def evaluate_rl_agent(
    env: Any,
    policy: Any,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> tuple[list[float], float]:
    """
    Evaluate RL agent and record cost history.

    Args:
        env: Environment instance
        policy: Trained RL policy
        max_steps: Maximum steps to run
        deterministic: Whether to use deterministic actions

    Returns:
        (cost_history, final_improvement)
    """
    obs, info = env.reset()
    initial_cost = info["initial_cost"]
    best_cost = initial_cost
    cost_history = [initial_cost]

    for step in range(max_steps):
        # Get action from policy
        if hasattr(policy, "get_action"):
            action = policy.get_action(obs, deterministic=deterministic)
        else:
            # Assume policy returns action directly
            action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        current_best = info.get("best_cost", info["current_cost"])
        if current_best < best_cost:
            best_cost = current_best

        cost_history.append(best_cost)

        if terminated or truncated:
            # Pad remaining steps with final cost
            remaining = max_steps - step - 1
            cost_history.extend([best_cost] * remaining)
            break

    final_improvement = (initial_cost - best_cost) / (initial_cost + 1e-6) * 100
    return cost_history, final_improvement


def create_cost_function(env: Any) -> Callable[[list[int]], float]:
    """
    Create a cost function for traditional optimizers from the environment.

    Args:
        env: Environment instance (must be reset first)

    Returns:
        cost_fn: Function that takes permutation and returns cost
    """
    # Get reference to cost engine
    cost_engine = env.cost_engine
    original_layout = cost_engine.get_layout_snapshot()

    def cost_fn(permutation: list[int]) -> float:
        # Apply permutation to layout
        # This is a simplified version - actual implementation may vary
        slots = list(cost_engine.layout.keys())
        depts = list(cost_engine.layout.values())

        # Reorder departments according to permutation
        new_depts = [depts[i] for i in permutation]

        # Create new layout
        new_layout = dict(zip(slots, new_depts))

        # Temporarily set layout and compute cost
        cost = cost_engine.restore_layout(new_layout)

        return cost

    return cost_fn


def run_sample_efficiency_experiment(
    env: Any,
    rl_policy: Any,
    n_runs: int = 10,
    max_steps: int = 1000,
    seed_base: int = 42,
) -> dict[str, Any]:
    """
    Run sample efficiency comparison experiment.

    Args:
        env: Environment instance
        rl_policy: Trained RL policy
        n_runs: Number of independent runs
        max_steps: Maximum steps per run
        seed_base: Base seed for reproducibility

    Returns:
        Dictionary with results for each method
    """
    results = {
        "RL": {"histories": [], "improvements": []},
        "SA": {"histories": [], "improvements": []},
        "Greedy": {"histories": [], "improvements": []},
        "Random": {"histories": [], "improvements": []},
        "Tabu": {"histories": [], "improvements": []},
        "steps": list(range(max_steps + 1)),
    }

    for run in range(n_runs):
        seed = seed_base + run
        logger.info(f"Run {run + 1}/{n_runs} (seed={seed})")

        # Reset environment with seed
        obs, info = env.reset(seed=seed)
        initial_cost = info["initial_cost"]
        n_depts = info["num_departments"]

        # Create cost function for traditional methods
        cost_fn = create_cost_function(env)

        # RL Agent
        env.reset(seed=seed)  # Reset to same state
        rl_history, rl_improvement = evaluate_rl_agent(env, rl_policy, max_steps)
        results["RL"]["histories"].append(rl_history)
        results["RL"]["improvements"].append(rl_improvement)

        # Simulated Annealing
        sa = SimulatedAnnealing(cost_fn, n_depts, seed=seed)
        _, sa_best_cost, sa_history = sa.solve(max_steps)
        sa_improvement = (initial_cost - sa_best_cost) / (initial_cost + 1e-6) * 100
        results["SA"]["histories"].append([initial_cost] + sa_history)
        results["SA"]["improvements"].append(sa_improvement)

        # Greedy Swap
        greedy = GreedySwap(cost_fn, n_depts, seed=seed)
        _, greedy_best_cost, greedy_history = greedy.solve(max_steps)
        greedy_improvement = (initial_cost - greedy_best_cost) / (initial_cost + 1e-6) * 100
        # Pad greedy history (it may terminate early)
        greedy_full = [initial_cost] + greedy_history
        if len(greedy_full) < max_steps + 1:
            greedy_full.extend([greedy_full[-1]] * (max_steps + 1 - len(greedy_full)))
        results["Greedy"]["histories"].append(greedy_full)
        results["Greedy"]["improvements"].append(greedy_improvement)

        # Random Search
        random = RandomSearch(cost_fn, n_depts, seed=seed)
        _, random_best_cost, random_history = random.solve(max_steps)
        random_improvement = (initial_cost - random_best_cost) / (initial_cost + 1e-6) * 100
        results["Random"]["histories"].append([initial_cost] + random_history)
        results["Random"]["improvements"].append(random_improvement)

        # Tabu Search
        tabu = TabuSearch(cost_fn, n_depts, seed=seed)
        _, tabu_best_cost, tabu_history = tabu.solve(max_steps)
        tabu_improvement = (initial_cost - tabu_best_cost) / (initial_cost + 1e-6) * 100
        tabu_full = [initial_cost] + tabu_history
        if len(tabu_full) < max_steps + 1:
            tabu_full.extend([tabu_full[-1]] * (max_steps + 1 - len(tabu_full)))
        results["Tabu"]["histories"].append(tabu_full)
        results["Tabu"]["improvements"].append(tabu_improvement)

    # Compute statistics
    for method in ["RL", "SA", "Greedy", "Random", "Tabu"]:
        histories = np.array(results[method]["histories"])
        results[method]["mean_history"] = histories.mean(axis=0)
        results[method]["std_history"] = histories.std(axis=0)
        results[method]["mean_improvement"] = np.mean(results[method]["improvements"])
        results[method]["std_improvement"] = np.std(results[method]["improvements"])

    return results


def plot_sample_efficiency(
    results: dict[str, Any],
    save_path: str | Path,
    title: str = "Sample Efficiency Comparison",
) -> None:
    """
    Plot sample efficiency comparison.

    Args:
        results: Results from run_sample_efficiency_experiment
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color scheme
    colors = {
        "RL": "#2ecc71",      # Green
        "SA": "#3498db",      # Blue
        "Greedy": "#e74c3c",  # Red
        "Random": "#95a5a6",  # Gray
        "Tabu": "#9b59b6",    # Purple
    }

    # Left: Cost over steps
    ax1 = axes[0]
    steps = results["steps"]

    for method in ["RL", "SA", "Greedy", "Tabu", "Random"]:
        mean = results[method]["mean_history"]
        std = results[method]["std_history"]

        # Normalize by initial cost for comparison
        initial = mean[0]
        mean_norm = mean / initial
        std_norm = std / initial

        ax1.plot(steps, mean_norm, label=method, color=colors[method], linewidth=2)
        ax1.fill_between(
            steps,
            mean_norm - std_norm,
            mean_norm + std_norm,
            color=colors[method],
            alpha=0.2,
        )

    ax1.set_xlabel("Steps (Environment Interactions)", fontsize=12)
    ax1.set_ylabel("Normalized Cost (lower is better)", fontsize=12)
    ax1.set_title("Cost Reduction Over Steps", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(steps) - 1)

    # Right: Final improvement bar chart
    ax2 = axes[1]
    methods = ["RL", "SA", "Greedy", "Tabu", "Random"]
    improvements = [results[m]["mean_improvement"] for m in methods]
    stds = [results[m]["std_improvement"] for m in methods]

    bars = ax2.bar(
        methods,
        improvements,
        yerr=stds,
        capsize=5,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=1,
    )

    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_title("Final Layout Improvement", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.annotate(
            f"{imp:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Sample efficiency plot saved to {save_path}")


def print_summary(results: dict[str, Any]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SAMPLE EFFICIENCY EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<10} {'Improvement':<20} {'Std':<10}")
    print("-" * 40)

    for method in ["RL", "SA", "Greedy", "Tabu", "Random"]:
        mean_imp = results[method]["mean_improvement"]
        std_imp = results[method]["std_improvement"]
        print(f"{method:<10} {mean_imp:>6.2f}% {std_imp:>10.2f}%")

    print("=" * 60)
