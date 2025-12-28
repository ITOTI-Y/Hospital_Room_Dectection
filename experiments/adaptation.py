"""
Dynamic Adaptation Experiment

Tests how well the policy adapts to changes in patient flow patterns.

Compares:
1. Zero-shot transfer (use original policy directly)
2. Few-shot adaptation (5 episodes fine-tuning)
3. Retrain from scratch (expensive baseline)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from copy import deepcopy
import time
from loguru import logger

from src.rl.adaptation import FastAdaptationWrapper, perturb_flow_matrix


def evaluate_policy_on_flow(
    env: Any,
    policy: Any,
    flow_matrix: torch.Tensor,
    n_episodes: int = 5,
    deterministic: bool = True,
) -> dict[str, float]:
    """
    Evaluate policy on specific flow matrix.

    Args:
        env: Environment instance
        policy: Policy to evaluate
        flow_matrix: Flow matrix to use
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions

    Returns:
        Dictionary with mean improvement, std, costs
    """
    improvements = []
    costs = []

    for ep in range(n_episodes):
        obs, info = env.reset(options={"flow_matrix": flow_matrix.cpu().numpy()})
        initial_cost = info["initial_cost"]
        done = False

        while not done:
            if hasattr(policy, "get_action"):
                action = policy.get_action(obs, deterministic=deterministic)
            else:
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_cost = info.get("best_cost", info["current_cost"])
        improvement = (initial_cost - best_cost) / (initial_cost + 1e-6) * 100
        improvements.append(improvement)
        costs.append(best_cost)

    return {
        "mean_improvement": np.mean(improvements),
        "std_improvement": np.std(improvements),
        "mean_cost": np.mean(costs),
        "std_cost": np.std(costs),
        "improvements": improvements,
    }


def run_adaptation_experiment(
    env: Any,
    trained_policy: Any,
    perturbation_strengths: list[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    n_runs: int = 5,
    adaptation_episodes: int = 5,
    eval_episodes: int = 5,
    seed_base: int = 42,
) -> dict[str, Any]:
    """
    Run dynamic adaptation experiment.

    Args:
        env: Environment instance
        trained_policy: Pre-trained policy
        perturbation_strengths: List of perturbation levels to test
        n_runs: Number of independent runs per perturbation level
        adaptation_episodes: Episodes for few-shot adaptation
        eval_episodes: Episodes for evaluation
        seed_base: Base random seed

    Returns:
        Dictionary with results for each adaptation method
    """
    results = {
        "perturbations": perturbation_strengths,
        "zero_shot": {"improvements": [], "times": []},
        "few_shot": {"improvements": [], "times": []},
        "methods": ["zero_shot", "few_shot"],
    }

    # Get original flow matrix
    obs, info = env.reset(seed=seed_base)
    original_flow = torch.from_numpy(env.flow_matrix)

    # Create adaptation wrapper
    adaptation_wrapper = FastAdaptationWrapper(
        trained_policy,
        adaptation_lr=1e-3,
        adaptation_steps=10,
    )

    for pert_strength in perturbation_strengths:
        logger.info(f"Testing perturbation strength: {pert_strength}")

        zero_shot_improvements = []
        few_shot_improvements = []
        few_shot_times = []

        for run in range(n_runs):
            seed = seed_base + run

            # Generate perturbed flow matrix
            if pert_strength > 0:
                perturbed_flow = perturb_flow_matrix(
                    original_flow, pert_strength, seed=seed
                )
            else:
                perturbed_flow = original_flow

            # Zero-shot: use original policy directly
            env.reset(seed=seed)
            zero_shot_result = evaluate_policy_on_flow(
                env, trained_policy, perturbed_flow, eval_episodes
            )
            zero_shot_improvements.append(zero_shot_result["mean_improvement"])

            # Few-shot adaptation
            env.reset(seed=seed)
            t0 = time.time()
            adapted_policy = adaptation_wrapper.adapt(
                perturbed_flow, env, n_episodes=adaptation_episodes
            )
            adapt_time = time.time() - t0

            env.reset(seed=seed)
            few_shot_result = evaluate_policy_on_flow(
                env, adapted_policy, perturbed_flow, eval_episodes
            )
            few_shot_improvements.append(few_shot_result["mean_improvement"])
            few_shot_times.append(adapt_time)

            # Reset adaptation wrapper for next run
            adaptation_wrapper.reset_to_base()

        # Store results for this perturbation level
        results["zero_shot"]["improvements"].append({
            "mean": np.mean(zero_shot_improvements),
            "std": np.std(zero_shot_improvements),
            "all": zero_shot_improvements,
        })
        results["few_shot"]["improvements"].append({
            "mean": np.mean(few_shot_improvements),
            "std": np.std(few_shot_improvements),
            "all": few_shot_improvements,
        })
        results["few_shot"]["times"].append({
            "mean": np.mean(few_shot_times),
            "std": np.std(few_shot_times),
        })

        logger.info(
            f"  Zero-shot: {np.mean(zero_shot_improvements):.2f}% ± {np.std(zero_shot_improvements):.2f}%"
        )
        logger.info(
            f"  Few-shot: {np.mean(few_shot_improvements):.2f}% ± {np.std(few_shot_improvements):.2f}%"
        )

    return results


def plot_adaptation_results(
    results: dict[str, Any],
    save_path: str | Path,
    title: str = "Adaptation to Flow Changes",
) -> None:
    """
    Plot adaptation experiment results.

    Args:
        results: Results from run_adaptation_experiment
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    perturbations = [p * 100 for p in results["perturbations"]]  # Convert to %

    colors = {
        "zero_shot": "#e74c3c",   # Red
        "few_shot": "#2ecc71",    # Green
    }

    labels = {
        "zero_shot": "Zero-shot (no adaptation)",
        "few_shot": "Few-shot (5 episodes)",
    }

    # Left: Improvement vs perturbation
    ax1 = axes[0]

    for method in ["zero_shot", "few_shot"]:
        means = [r["mean"] for r in results[method]["improvements"]]
        stds = [r["std"] for r in results[method]["improvements"]]

        ax1.errorbar(
            perturbations,
            means,
            yerr=stds,
            label=labels[method],
            color=colors[method],
            linewidth=2,
            marker="o",
            markersize=8,
            capsize=5,
        )

    ax1.set_xlabel("Flow Perturbation Strength (%)", fontsize=12)
    ax1.set_ylabel("Layout Improvement (%)", fontsize=12)
    ax1.set_title("Performance Under Flow Changes", fontsize=14)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(perturbations)

    # Right: Adaptation time
    ax2 = axes[1]

    if results["few_shot"]["times"]:
        times_mean = [t["mean"] for t in results["few_shot"]["times"]]
        times_std = [t["std"] for t in results["few_shot"]["times"]]

        bars = ax2.bar(
            perturbations,
            times_mean,
            yerr=times_std,
            capsize=5,
            color=colors["few_shot"],
            edgecolor="black",
            linewidth=1,
            width=8,
        )

        ax2.set_xlabel("Flow Perturbation Strength (%)", fontsize=12)
        ax2.set_ylabel("Adaptation Time (seconds)", fontsize=12)
        ax2.set_title("Few-shot Adaptation Time", fontsize=14)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xticks(perturbations)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Adaptation plot saved to {save_path}")


def print_adaptation_summary(results: dict[str, Any]) -> None:
    """Print summary of adaptation experiment."""
    print("\n" + "=" * 70)
    print("DYNAMIC ADAPTATION EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\n{'Perturbation':<15} {'Zero-shot':<20} {'Few-shot':<20} {'Δ':<10}")
    print("-" * 65)

    for i, pert in enumerate(results["perturbations"]):
        zero = results["zero_shot"]["improvements"][i]
        few = results["few_shot"]["improvements"][i]
        delta = few["mean"] - zero["mean"]

        print(
            f"{pert*100:>6.0f}%        "
            f"{zero['mean']:>6.2f}% ± {zero['std']:>5.2f}%   "
            f"{few['mean']:>6.2f}% ± {few['std']:>5.2f}%   "
            f"{delta:>+5.2f}%"
        )

    print("=" * 70)

    # Summary statistics
    avg_zero = np.mean([r["mean"] for r in results["zero_shot"]["improvements"]])
    avg_few = np.mean([r["mean"] for r in results["few_shot"]["improvements"]])

    print(f"\nAverage improvement:")
    print(f"  Zero-shot: {avg_zero:.2f}%")
    print(f"  Few-shot:  {avg_few:.2f}%")
    print(f"  Gain:      {avg_few - avg_zero:+.2f}%")

    if results["few_shot"]["times"]:
        avg_time = np.mean([t["mean"] for t in results["few_shot"]["times"]])
        print(f"\nAverage adaptation time: {avg_time:.2f}s")
