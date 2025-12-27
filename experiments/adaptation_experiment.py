"""Dynamic adaptation capability experiments.

This module tests the model's ability to adapt to changing patient flow patterns:
- Zero-shot transfer (no adaptation)
- Few-shot adaptation (5-10 episodes)
- Full retraining baseline
"""

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.rl.adaptation import FastAdaptationWrapper


def run_adaptation_experiment(
    base_env: Any,
    trained_policy: Any,
    flow_perturbations: list[float] | None = None,
    n_runs: int = 5,
    n_adapt_episodes: int = 5,
    save_dir: str | Path | None = None,
) -> dict[str, list[float]]:
    """Test adaptation capability to flow changes.

    Compares three approaches:
    1. Zero-shot: Use base policy without adaptation
    2. Few-shot: Quick adaptation with 5-10 episodes
    3. Retrain: Full retraining baseline (expensive)

    Args:
        base_env: Base environment
        trained_policy: Pre-trained policy on original flow
        flow_perturbations: List of perturbation strengths (0-1)
        n_runs: Number of runs per perturbation
        n_adapt_episodes: Episodes for few-shot adaptation
        save_dir: Directory to save results

    Returns:
        Dictionary with adaptation results
    """
    if flow_perturbations is None:
        flow_perturbations = [0.1, 0.2, 0.3, 0.5]

    logger.info(
        f"Running adaptation experiment: perturbations={flow_perturbations}, "
        f"runs={n_runs}, adapt_episodes={n_adapt_episodes}"
    )

    # Get original flow from environment
    original_flow = _extract_flow_matrix(base_env)

    # Initialize adaptation wrapper
    adaptation_wrapper = FastAdaptationWrapper(
        policy=trained_policy,
        adaptation_lr=1e-3,
        adaptation_steps=10,
    )

    # Storage for results
    results = {
        "perturbation": flow_perturbations,
        "zero_shot": [],
        "few_shot": [],
        "adapt_time": [],
        "adapt_episodes": n_adapt_episodes,
    }

    for pert in flow_perturbations:
        logger.info(f"\nTesting perturbation: {pert * 100:.0f}%")

        zero_costs = []
        adapt_costs = []
        adapt_times = []

        for run in range(n_runs):
            logger.info(f"  Run {run + 1}/{n_runs}")

            # Generate perturbed flow
            perturbed_flow = _perturb_flow_matrix(original_flow, pert)

            # Zero-shot evaluation
            logger.info("    Zero-shot evaluation...")
            zero_cost = _evaluate_policy_on_flow(
                trained_policy, base_env, perturbed_flow
            )
            zero_costs.append(zero_cost)

            # Few-shot adaptation
            logger.info(f"    Few-shot adaptation ({n_adapt_episodes} episodes)...")
            t_start = time.time()
            adapted_policy = adaptation_wrapper.adapt(
                new_flow_matrix=perturbed_flow,
                env=base_env,
                n_episodes=n_adapt_episodes,
                verbose=False,
            )
            adapt_time = time.time() - t_start
            adapt_times.append(adapt_time)

            # Evaluate adapted policy
            adapt_cost = _evaluate_policy_on_flow(
                adapted_policy, base_env, perturbed_flow
            )
            adapt_costs.append(adapt_cost)

            logger.info(
                f"    Zero-shot: {zero_cost:.2f}, "
                f"Adapted: {adapt_cost:.2f} ({adapt_time:.2f}s), "
                f"Improvement: {(zero_cost - adapt_cost) / zero_cost * 100:.1f}%"
            )

        # Store aggregated results
        results["zero_shot"].append(np.mean(zero_costs))
        results["few_shot"].append(np.mean(adapt_costs))
        results["adapt_time"].append(np.mean(adapt_times))

    # Plot results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_adaptation_results(results, save_dir / "adaptation_results.png")
        plot_adaptation_time(results, save_dir / "adaptation_time.png")

    logger.info("\nAdaptation experiment complete")
    _print_adaptation_summary(results)

    return results


def _extract_flow_matrix(env: Any) -> np.ndarray:
    """Extract flow matrix from environment.

    Args:
        env: Environment instance

    Returns:
        Flow matrix (n_depts, n_depts)
    """
    n = env.num_total_slot
    flow_matrix = np.zeros((n, n), dtype=np.float32)

    # Extract from pair_weights
    for (dept1, dept2), weight in env.cost_manager.pair_weights.items():
        if dept1 in env.dept_id_to_index and dept2 in env.dept_id_to_index:
            i = env.dept_id_to_index[dept1]
            j = env.dept_id_to_index[dept2]
            flow_matrix[i, j] = weight
            flow_matrix[j, i] = weight

    # Normalize
    if flow_matrix.max() > 0:
        flow_matrix = flow_matrix / flow_matrix.max()

    return flow_matrix


def _perturb_flow_matrix(flow: np.ndarray, strength: float) -> np.ndarray:
    """Perturb flow matrix to simulate distribution shift.

    Args:
        flow: Original flow matrix
        strength: Perturbation strength (0-1)

    Returns:
        Perturbed flow matrix
    """
    noise = np.random.uniform(1 - strength, 1 + strength, flow.shape)
    perturbed = flow * noise

    # Maintain symmetry
    perturbed = (perturbed + perturbed.T) / 2

    # Normalize
    if perturbed.max() > 0:
        perturbed = perturbed / perturbed.max()

    return perturbed


def _evaluate_policy_on_flow(
    policy: Any, env: Any, flow_matrix: np.ndarray, n_episodes: int = 3
) -> float:
    """Evaluate policy on environment with specified flow matrix.

    Args:
        policy: Policy to evaluate
        env: Environment
        flow_matrix: Flow matrix to use
        n_episodes: Number of evaluation episodes

    Returns:
        Average final cost
    """
    costs = []

    for _ in range(n_episodes):
        # Reset environment with custom flow matrix
        obs, _ = env.reset(flow_matrix=flow_matrix)

        # Run episode
        done = False
        step = 0
        max_steps = 100

        while not done and step < max_steps:
            # Get action
            action, _, _ = policy.forward_actor(
                {k: np.expand_dims(v, 0) for k, v in obs.items()},
                deterministic=True,
            )

            # Step
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            step += 1

        # Record final cost
        final_cost = info.get("current_cost", env.current_cost)
        costs.append(final_cost)

    return np.mean(costs)


def _pad_flow_matrix(flow: np.ndarray, max_depts: int) -> np.ndarray:
    """Pad flow matrix to max_departments size.

    Args:
        flow: Flow matrix
        max_depts: Maximum departments

    Returns:
        Padded flow matrix
    """
    padded = np.zeros((max_depts, max_depts), dtype=np.float32)
    n = min(flow.shape[0], max_depts)
    padded[:n, :n] = flow[:n, :n]
    return padded


def plot_adaptation_results(
    results: dict[str, list[float]], save_path: str | Path
) -> None:
    """Plot adaptation performance vs perturbation strength.

    Args:
        results: Results dictionary
        save_path: Save path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    perturbations = np.array(results["perturbation"]) * 100  # Convert to percentage
    zero_shot = results["zero_shot"]
    few_shot = results["few_shot"]

    # Bar plot
    x = np.arange(len(perturbations))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, zero_shot, width, label="Zero-shot", color="skyblue", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, few_shot, width, label="Few-shot Adapted", color="orange", alpha=0.8
    )

    # Add improvement percentage annotations
    for i, (z, f) in enumerate(zip(zero_shot, few_shot)):
        improvement = (z - f) / z * 100
        ax.text(
            i,
            max(z, f) + 5,
            f"+{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Flow Perturbation Strength (%)", fontsize=12)
    ax.set_ylabel("Final Layout Cost", fontsize=12)
    ax.set_title(
        "Adaptation Performance vs Flow Changes", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}%" for p in perturbations])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Adaptation results plot saved to {save_path}")


def plot_adaptation_time(results: dict[str, list[float]], save_path: str | Path) -> None:
    """Plot adaptation time vs perturbation strength.

    Args:
        results: Results dictionary
        save_path: Save path for plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    perturbations = np.array(results["perturbation"]) * 100
    adapt_times = results["adapt_time"]

    ax.plot(
        perturbations,
        adapt_times,
        marker="o",
        linewidth=2,
        markersize=8,
        color="green",
    )

    ax.set_xlabel("Flow Perturbation Strength (%)", fontsize=12)
    ax.set_ylabel("Adaptation Time (seconds)", fontsize=12)
    ax.set_title("Adaptation Speed", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add time annotations
    for p, t in zip(perturbations, adapt_times):
        ax.text(p, t + 0.1, f"{t:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Adaptation time plot saved to {save_path}")


def _print_adaptation_summary(results: dict[str, list[float]]) -> None:
    """Print summary of adaptation results.

    Args:
        results: Results dictionary
    """
    logger.info("\n" + "=" * 60)
    logger.info("ADAPTATION EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for i, pert in enumerate(results["perturbation"]):
        zero = results["zero_shot"][i]
        adapted = results["few_shot"][i]
        time_taken = results["adapt_time"][i]
        improvement = (zero - adapted) / zero * 100

        logger.info(f"\nPerturbation: {pert * 100:.0f}%")
        logger.info(f"  Zero-shot cost:     {zero:.2f}")
        logger.info(f"  Adapted cost:       {adapted:.2f}")
        logger.info(f"  Improvement:        {improvement:.1f}%")
        logger.info(f"  Adaptation time:    {time_taken:.2f}s")

    logger.info("\n" + "=" * 60)
