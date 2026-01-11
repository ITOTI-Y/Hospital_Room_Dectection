"""
Baseline Runner for Hospital Layout Optimization

Provides a unified interface to run and compare baseline algorithms
(Genetic Algorithm and Simulated Annealing).
"""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.baseline.base import OptimizationResult
from src.baseline.genetic import GAConfig, GeneticAlgorithm
from src.baseline.simulated_annealing import (
    SAConfig,
    SimulatedAnnealing,
    estimate_initial_temperature,
)
from src.config.config_loader import ConfigLoader
from src.pipeline.cost_manager_v2 import CostManager
from src.pipeline.pathway_generator import PathwayGenerator


def _convert_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(item) for item in obj]
    return obj


class BaselineRunner:
    """Runner for baseline optimization algorithms.

    Handles:
    - Initialization of cost manager and pathway generator
    - Running GA and SA algorithms
    - Multiple runs with different seeds
    - Result aggregation and export

    Example:
        >>> runner = BaselineRunner(config)
        >>> results = runner.run_comparison(n_runs=5, max_iterations=1000)
        >>> runner.export_results(results, 'results/baseline')
    """

    def __init__(
        self,
        config: ConfigLoader,
        shuffle_initial_layout: bool = False,
        eval_mode: Literal['smart', 'traditional'] = 'smart',
    ):
        self.config = config
        self.logger = logger.bind(module='BaselineRunner')

        self.pathway_generator = PathwayGenerator(config, is_training=False, eval_mode=eval_mode)
        self.cost_manager = CostManager(
            config, shuffle_initial_layout=shuffle_initial_layout
        )

        self._pathways_initialized = False

    def initialize_pathways(self) -> None:
        """Initialize patient pathways for optimization."""
        pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways)
        self._pathways_initialized = True
        self.logger.info(f'Initialized {len(pathways)} patient pathways')

    def run_genetic_algorithm(
        self,
        max_iterations: int = 200,
        ga_config: GAConfig | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run Genetic Algorithm optimization.

        Args:
            max_iterations: Maximum generations
            ga_config: GA configuration (uses defaults if None)
            seed: Random seed

        Returns:
            OptimizationResult with best layout and statistics
        """
        if not self._pathways_initialized:
            self.initialize_pathways()

        ga = GeneticAlgorithm(self.cost_manager, config=ga_config)
        return ga.optimize(max_iterations=max_iterations, seed=seed)

    def run_simulated_annealing(
        self,
        max_iterations: int = 10000,
        sa_config: SAConfig | None = None,
        auto_temp: bool = True,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run Simulated Annealing optimization.

        Args:
            max_iterations: Maximum iterations
            sa_config: SA configuration (uses defaults if None)
            auto_temp: Whether to auto-estimate initial temperature
            seed: Random seed

        Returns:
            OptimizationResult with best layout and statistics
        """
        if not self._pathways_initialized:
            self.initialize_pathways()

        if auto_temp and sa_config is None:
            initial_temp = estimate_initial_temperature(
                self.cost_manager, acceptance_prob=0.8, seed=seed
            )
            sa_config = SAConfig(initial_temp=initial_temp)
            self.logger.info(f'Auto-estimated initial temperature: {initial_temp:.2f}')

        sa = SimulatedAnnealing(self.cost_manager, config=sa_config)
        return sa.optimize(max_iterations=max_iterations, seed=seed)

    def run_comparison(
        self,
        n_runs: int = 5,
        ga_iterations: int = 200,
        sa_iterations: int = 10000,
        ga_config: GAConfig | None = None,
        sa_config: SAConfig | None = None,
        base_seed: int = 42,
    ) -> dict[str, list[OptimizationResult]]:
        """Run multiple trials of both algorithms for comparison.

        Args:
            n_runs: Number of runs per algorithm
            ga_iterations: Max generations for GA
            sa_iterations: Max iterations for SA
            ga_config: GA configuration
            sa_config: SA configuration
            base_seed: Base random seed (incremented per run)

        Returns:
            Dictionary with 'ga' and 'sa' keys containing result lists
        """
        if not self._pathways_initialized:
            self.initialize_pathways()

        results: dict[str, list[OptimizationResult]] = {'ga': [], 'sa': []}

        self.logger.info(f'Starting comparison: {n_runs} runs per algorithm')

        for i in range(n_runs):
            seed = base_seed + i
            self.logger.info(f'Run {i + 1}/{n_runs} (seed={seed})')

            ga_result = self.run_genetic_algorithm(
                max_iterations=ga_iterations,
                ga_config=ga_config,
                seed=seed,
            )
            results['ga'].append(ga_result)

            sa_result = self.run_simulated_annealing(
                max_iterations=sa_iterations,
                sa_config=sa_config,
                seed=seed,
            )
            results['sa'].append(sa_result)

        self._log_comparison_summary(results)
        return results

    def _log_comparison_summary(
        self, results: dict[str, list[OptimizationResult]]
    ) -> None:
        """Log summary statistics for comparison results."""
        self.logger.info('=' * 60)
        self.logger.info('COMPARISON SUMMARY')
        self.logger.info('=' * 60)

        for algo, result_list in results.items():
            if not result_list:
                continue

            costs = [r.best_cost for r in result_list]
            improvements = [r.improvement_ratio for r in result_list]
            times = [r.time_seconds for r in result_list]

            algo_name = 'Genetic Algorithm' if algo == 'ga' else 'Simulated Annealing'
            self.logger.info(f'\n{algo_name}:')
            self.logger.info(
                f'  Best Cost: {np.min(costs):.2f} '
                f'(mean={np.mean(costs):.2f}, std={np.std(costs):.2f})'
            )
            self.logger.info(
                f'  Improvement: {np.max(improvements):.2%} '
                f'(mean={np.mean(improvements):.2%}, std={np.std(improvements):.2%})'
            )
            self.logger.info(
                f'  Time: {np.mean(times):.1f}s (total={np.sum(times):.1f}s)'
            )

    def export_results(
        self,
        results: dict[str, list[OptimizationResult]],
        output_dir: str | Path,
        prefix: str = 'baseline',
    ) -> dict[str, Path]:
        """Export optimization results to files.

        Args:
            results: Results from run_comparison
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Dictionary of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files: dict[str, Path] = {}

        summary_data = []
        for algo, result_list in results.items():
            for i, result in enumerate(result_list):
                summary_data.append(
                    {
                        'algorithm': algo,
                        'run': i + 1,
                        'initial_cost': result.initial_cost,
                        'best_cost': result.best_cost,
                        'improvement_ratio': result.improvement_ratio,
                        'iterations': result.iterations,
                        'evaluations': result.evaluations,
                        'time_seconds': result.time_seconds,
                        'converged': result.converged,
                    }
                )

        summary_path = output_dir / f'{prefix}_summary.csv'
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        output_files['summary'] = summary_path
        self.logger.info(f'Saved summary to {summary_path}')

        for algo, result_list in results.items():
            for i, result in enumerate(result_list):
                layout_path = output_dir / f'{prefix}_{algo}_run{i + 1}_layout.npy'
                np.save(layout_path, result.best_layout)
                output_files[f'{algo}_layout_{i + 1}'] = layout_path

                history_path = output_dir / f'{prefix}_{algo}_run{i + 1}_history.json'
                with open(history_path, 'w') as f:
                    json.dump(
                        _convert_to_native(
                            {
                                'cost_history': result.cost_history,
                                'metadata': result.metadata,
                            }
                        ),
                        f,
                        indent=2,
                    )
                output_files[f'{algo}_history_{i + 1}'] = history_path

        self.logger.info(f'Exported {len(output_files)} files to {output_dir}')
        return output_files

    def run_single(
        self,
        algorithm: Literal['ga', 'sa'],
        max_iterations: int | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run a single algorithm.

        Args:
            algorithm: 'ga' for Genetic Algorithm, 'sa' for Simulated Annealing
            max_iterations: Maximum iterations (uses default if None)
            seed: Random seed

        Returns:
            OptimizationResult
        """
        if algorithm == 'ga':
            iterations = max_iterations or 200
            return self.run_genetic_algorithm(max_iterations=iterations, seed=seed)
        elif algorithm == 'sa':
            iterations = max_iterations or 10000
            return self.run_simulated_annealing(max_iterations=iterations, seed=seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'ga' or 'sa'")
