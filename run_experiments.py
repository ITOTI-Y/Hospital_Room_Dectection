#!/usr/bin/env python
"""
Run experiments for the layout optimization paper.

Usage:
    python run_experiments.py --experiment sample_efficiency
    python run_experiments.py --experiment adaptation
    python run_experiments.py --experiment all
"""

import argparse
from pathlib import Path

import torch
from loguru import logger

from src.config import config_loader
from src.optimize_manager import OptimizeManager
from src.rl.env import LayoutEnv


def load_trained_policy(config, model_path: str | Path):
    """Load trained policy from checkpoint."""
    manager = OptimizeManager(config)
    model = manager.create_model()

    state_dict = torch.load(model_path, map_location=manager.device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_sample_efficiency_experiment(config, model_path: str | Path, output_dir: Path):
    """Run sample efficiency comparison."""
    from experiments.sample_efficiency import (
        run_sample_efficiency_experiment as run_exp,
        plot_sample_efficiency,
        print_summary,
    )

    logger.info("Running sample efficiency experiment...")

    # Create environment
    env = LayoutEnv(
        config=config,
        max_departments=config.agent.max_departments,
        max_step=config.agent.max_steps,
        is_training=False,
    )

    # Load policy
    policy = load_trained_policy(config, model_path)

    # Run experiment
    results = run_exp(
        env=env,
        rl_policy=policy,
        n_runs=10,
        max_steps=100,
        seed_base=42,
    )

    # Plot results
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_sample_efficiency(results, output_dir / "sample_efficiency.png")
    print_summary(results)

    logger.info(f"Sample efficiency results saved to {output_dir}")
    return results


def run_adaptation_experiment(config, model_path: str | Path, output_dir: Path):
    """Run dynamic adaptation experiment."""
    from experiments.adaptation import (
        run_adaptation_experiment as run_exp,
        plot_adaptation_results,
        print_adaptation_summary,
    )

    logger.info("Running adaptation experiment...")

    # Create environment
    env = LayoutEnv(
        config=config,
        max_departments=config.agent.max_departments,
        max_step=config.agent.max_steps,
        is_training=False,
    )

    # Load policy
    policy = load_trained_policy(config, model_path)

    # Run experiment
    results = run_exp(
        env=env,
        trained_policy=policy,
        perturbation_strengths=[0.0, 0.1, 0.2, 0.3, 0.5],
        n_runs=5,
        adaptation_episodes=5,
        eval_episodes=5,
        seed_base=42,
    )

    # Plot results
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_adaptation_results(results, output_dir / "adaptation.png")
    print_adaptation_summary(results)

    logger.info(f"Adaptation results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run layout optimization experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["sample_efficiency", "adaptation", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/model/best_ppo_layout_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load config
    config = config_loader.ConfigLoader()
    output_dir = Path(args.output_dir)

    if args.experiment in ["sample_efficiency", "all"]:
        run_sample_efficiency_experiment(config, args.model_path, output_dir)

    if args.experiment in ["adaptation", "all"]:
        run_adaptation_experiment(config, args.model_path, output_dir)

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
