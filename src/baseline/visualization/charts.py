"""Chart generator for baseline optimization algorithm comparison."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ultraplot as uplt

from src.baseline.base import OptimizationResult
from src.baseline.visualization.journal_style import (
    JOURNAL_STYLE,
    FigureWidth,
    ImageType,
    JournalStyle,
)


def load_results_from_dir(
    results_dir: Path | str,
    prefix: str = 'baseline',
) -> dict[str, list[OptimizationResult]]:
    """Load optimization results from saved files.

    Args:
        results_dir: Directory containing result files.
        prefix: Filename prefix.

    Returns:
        Dict mapping algorithm names to list of OptimizationResult.
    """
    results_dir = Path(results_dir)
    results: dict[str, list[OptimizationResult]] = {'GA': [], 'SA': []}

    summary_path = results_dir / f'{prefix}_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')

    summary_df = pd.read_csv(summary_path)

    for _, row in summary_df.iterrows():
        algo = row['algorithm']
        run_id = int(row['run'])

        history_path = results_dir / f'{prefix}_{algo}_run{run_id}_history.json'
        layout_path = results_dir / f'{prefix}_{algo}_run{run_id}_layout.npy'

        with open(history_path) as f:
            history_data = json.load(f)

        best_layout = np.load(layout_path)

        result = OptimizationResult(
            best_cost=row['best_cost'],
            initial_cost=row['initial_cost'],
            improvement_ratio=row['improvement_ratio'],
            best_layout=best_layout,
            cost_history=history_data['cost_history'],
            iterations=row['iterations'],
            evaluations=row['evaluations'],
            time_seconds=row['time_seconds'],
            converged=row['converged'],
            metadata=history_data.get('metadata', {}),
        )

        algo_key = 'GA' if algo == 'ga' else 'SA'
        results[algo_key].append(result)

    return results


@dataclass
class MultiObjectiveResult:
    """Result container for multi-objective optimization.

    Attributes:
        objectives: Array of shape (n_solutions, n_objectives)
        solutions: List of solution layouts
        algorithm: Algorithm name
        metadata: Additional data
    """

    objectives: np.ndarray
    solutions: list[np.ndarray]
    algorithm: str
    metadata: dict[str, Any] | None = None


class BaselineChartGenerator:
    """Chart generator for baseline algorithm comparison visualization."""

    def __init__(
        self,
        output_dir: Path | str,
        style: JournalStyle | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style or JOURNAL_STYLE
        uplt.rc.update(self.style.get_rc_params())

    def create_figure(
        self,
        width: FigureWidth = FigureWidth.SINGLE_COLUMN,
        aspect_ratio: float = 0.618,
        ncols: int = 1,
        nrows: int = 1,
        **kwargs,
    ) -> tuple[uplt.Figure, Any]:
        """Create figure with journal-compliant dimensions.

        Args:
            width: Total figure width (not per-subplot).
            aspect_ratio: Height/width ratio for each subplot.
            ncols: Number of columns.
            nrows: Number of rows.
        """
        subplot_width = width.value / ncols
        subplot_height = subplot_width * aspect_ratio
        return uplt.subplots(
            ncols=ncols,
            nrows=nrows,
            refwidth=subplot_width,
            refheight=subplot_height,
            **kwargs,
        )

    def save(
        self,
        fig: uplt.Figure,
        name: str,
        image_type: ImageType = ImageType.LINE_ART,
        fmt: str | None = None,
    ) -> Path:
        """Save figure with journal-compliant format and DPI."""
        output_format = fmt or self.style.default_format
        path = self.output_dir / f'{name}.{output_format}'
        fig.save(path, dpi=image_type.value)
        return path

    def convergence_comparison(
        self,
        results: dict[str, OptimizationResult],
        normalize: bool = True,
        log_scale: bool = False,
        x_axis: str = 'iterations',
    ) -> Path:
        """Generate convergence speed comparison chart.

        Compares how quickly different algorithms converge to optimal solution.

        Args:
            results: Dict mapping algorithm names to OptimizationResult.
            normalize: Whether to normalize costs relative to initial cost.
            log_scale: Whether to use log scale for y-axis.
            x_axis: What to use for x-axis ('iterations' or 'evaluations').

        Returns:
            Path to saved figure.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
        )

        try:
            xlabel = 'Iterations'
            for name, result in results.items():
                cost_history = np.array(result.cost_history)

                if normalize:
                    cost_history = cost_history / result.initial_cost

                if x_axis == 'evaluations':
                    n_points = len(cost_history)
                    evals_per_iter = result.evaluations / max(result.iterations, 1)
                    x = np.arange(n_points) * evals_per_iter
                    xlabel = 'Evaluations'
                else:
                    x = np.arange(len(cost_history))
                    xlabel = 'Iterations'

                color = self.style.get_algorithm_color(name)
                ax.plot(
                    x,
                    cost_history,
                    label=name,
                    color=color,
                    linewidth=self.style.line_width_thick,
                )

            ax.set_xlabel(xlabel)
            ylabel = 'Normalized Cost' if normalize else 'Travel Cost'
            ax.set_ylabel(ylabel)

            if log_scale:
                ax.set_yscale('log')

            ax.legend(
                loc='upper right',
                frameon=False,
            )

            ax.format(
                title='Convergence Comparison',
            )

            return self.save(fig, 'convergence_comparison')
        finally:
            uplt.close(fig)

    def solution_quality_comparison(
        self,
        results: dict[str, list[OptimizationResult]],
        metric: str = 'improvement_ratio',
    ) -> Path:
        """Generate solution quality comparison chart using box plots.

        Compares final solution quality across multiple runs of each algorithm.

        Args:
            results: Dict mapping algorithm names to list of OptimizationResult
                     from multiple runs.
            metric: Which metric to compare ('improvement_ratio', 'best_cost',
                    'time_seconds', 'evaluations').

        Returns:
            Path to saved figure.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.SINGLE_COLUMN,
            aspect_ratio=0.8,
        )

        try:
            algorithm_names = list(results.keys())
            positions = np.arange(len(algorithm_names))

            for i, name in enumerate(algorithm_names):
                run_results = results[name]
                values = np.array([getattr(r, metric) for r in run_results])
                color = self.style.get_algorithm_color(name)

                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(values))
                ax.scatter(
                    positions[i] + jitter,
                    values,
                    c=color,
                    alpha=0.6,
                    s=self.style.marker_size**2,
                    zorder=2,
                )

                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.errorbar(
                    positions[i],
                    mean_val,
                    yerr=std_val,
                    fmt='_',
                    color='black',
                    markersize=15,
                    markeredgewidth=2,
                    capsize=6,
                    capthick=self.style.line_width_thick,
                    elinewidth=self.style.line_width,
                    zorder=3,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(algorithm_names)

            ylabel_map = {
                'improvement_ratio': 'Improvement Ratio',
                'best_cost': 'Best Cost',
                'time_seconds': 'Time (seconds)',
                'evaluations': 'Evaluations',
            }
            ax.set_ylabel(ylabel_map.get(metric, metric))

            ax.format(
                title=f'Solution Quality Comparison ({metric})',
            )

            return self.save(fig, f'quality_comparison_{metric}')
        finally:
            uplt.close(fig)

    def solution_quality_bar(
        self,
        results: dict[str, list[OptimizationResult]],
        metric: str = 'improvement_ratio',
        show_error: bool = True,
    ) -> Path:
        """Generate solution quality comparison as bar chart with error bars.

        Args:
            results: Dict mapping algorithm names to list of OptimizationResult.
            metric: Which metric to compare.
            show_error: Whether to show standard deviation as error bars.

        Returns:
            Path to saved figure.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.SINGLE_COLUMN,
            aspect_ratio=0.7,
        )

        try:
            algorithm_names = list(results.keys())
            means = []
            stds = []
            colors = []

            for name in algorithm_names:
                run_results = results[name]
                values = [getattr(r, metric) for r in run_results]
                means.append(np.mean(values))
                stds.append(np.std(values))
                colors.append(self.style.get_algorithm_color(name))

            positions = np.arange(len(algorithm_names))

            ax.bar(
                positions,
                means,
                width=0.6,
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=self.style.axis_line_width,
            )

            if show_error and len(results[algorithm_names[0]]) > 1:
                ax.errorbar(
                    positions,
                    means,
                    yerr=stds,
                    fmt='none',
                    color='black',
                    capsize=4,
                    capthick=self.style.line_width,
                    elinewidth=self.style.line_width,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(algorithm_names)

            ylabel_map = {
                'improvement_ratio': 'Improvement Ratio',
                'best_cost': 'Best Cost',
                'time_seconds': 'Time (seconds)',
                'evaluations': 'Evaluations',
            }
            ax.set_ylabel(ylabel_map.get(metric, metric))

            ax.format(
                title=f'Algorithm Comparison ({metric})',
            )

            return self.save(fig, f'bar_comparison_{metric}')
        finally:
            uplt.close(fig)

    def pareto_front(
        self,
        results: dict[str, MultiObjectiveResult],
        objective_names: tuple[str, str] = ('Objective 1', 'Objective 2'),
        show_dominated: bool = True,
    ) -> Path:
        """Generate Pareto front comparison chart for multi-objective optimization.

        Args:
            results: Dict mapping algorithm names to MultiObjectiveResult.
            objective_names: Names for the two objectives being plotted.
            show_dominated: Whether to show dominated solutions (in lighter color).

        Returns:
            Path to saved figure.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.618,
        )

        try:
            for name, result in results.items():
                objectives = result.objectives
                color = self.style.get_algorithm_color(name)

                if objectives.shape[1] < 2:
                    continue

                pareto_mask = self._compute_pareto_mask(objectives)

                if show_dominated:
                    dominated = ~pareto_mask
                    ax.scatter(
                        objectives[dominated, 0],
                        objectives[dominated, 1],
                        c=color,
                        alpha=0.2,
                        s=self.style.marker_size_small**2,
                        marker='o',
                    )

                pareto_points = objectives[pareto_mask]
                sorted_idx = np.argsort(pareto_points[:, 0])
                pareto_sorted = pareto_points[sorted_idx]

                ax.scatter(
                    pareto_sorted[:, 0],
                    pareto_sorted[:, 1],
                    c=color,
                    s=self.style.marker_size**2,
                    marker='o',
                    label=name,
                    edgecolors='black',
                    linewidths=0.5,
                )

                ax.plot(
                    pareto_sorted[:, 0],
                    pareto_sorted[:, 1],
                    c=color,
                    linewidth=self.style.line_width,
                    alpha=0.6,
                    linestyle='--',
                )

            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])

            ax.legend(
                loc='upper right',
                frameon=False,
            )

            ax.format(
                title='Pareto Front Comparison',
            )

            return self.save(fig, 'pareto_front')
        finally:
            uplt.close(fig)

    def pareto_spread_comparison(
        self,
        results: dict[str, MultiObjectiveResult],
    ) -> Path:
        """Generate Pareto front spread comparison bar chart.

        Compares the spread/diversity of Pareto fronts across algorithms.

        Args:
            results: Dict mapping algorithm names to MultiObjectiveResult.

        Returns:
            Path to saved figure.
        """
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
            ncols=2,
        )

        try:
            algorithm_names = list(results.keys())
            spreads = []
            hypervolumes = []
            colors = []

            for name in algorithm_names:
                result = results[name]
                objectives = result.objectives
                pareto_mask = self._compute_pareto_mask(objectives)
                pareto_points = objectives[pareto_mask]

                spread = self._compute_spread(pareto_points)
                spreads.append(spread)

                hv = self._compute_hypervolume_2d(pareto_points)
                hypervolumes.append(hv)

                colors.append(self.style.get_algorithm_color(name))

            positions = np.arange(len(algorithm_names))

            axs[0].bar(
                positions,
                spreads,
                width=0.6,
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=self.style.axis_line_width,
            )
            axs[0].set_xticks(positions)
            axs[0].set_xticklabels(algorithm_names)
            axs[0].set_ylabel('Spread')
            axs[0].set_title('Pareto Front Spread')

            axs[1].bar(
                positions,
                hypervolumes,
                width=0.6,
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=self.style.axis_line_width,
            )
            axs[1].set_xticks(positions)
            axs[1].set_xticklabels(algorithm_names)
            axs[1].set_ylabel('Hypervolume')
            axs[1].set_title('Hypervolume Indicator')

            axs.format(abc='a.', abcloc='ul')

            return self.save(fig, 'pareto_spread_comparison')
        finally:
            uplt.close(fig)

    def convergence_with_confidence(
        self,
        results: dict[str, list[OptimizationResult]],
        normalize: bool = True,
        confidence: float = 0.95,
    ) -> Path:
        """Generate convergence chart with confidence intervals from multiple runs.

        Args:
            results: Dict mapping algorithm names to list of OptimizationResult
                     from multiple runs.
            normalize: Whether to normalize costs relative to initial cost.
            confidence: Confidence level for interval (e.g., 0.95 for 95%).

        Returns:
            Path to saved figure.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
        )

        try:
            for name, run_results in results.items():
                max_len = max(len(r.cost_history) for r in run_results)
                histories = np.full((len(run_results), max_len), np.nan)

                for i, r in enumerate(run_results):
                    h = np.array(r.cost_history)
                    if normalize:
                        h = h / r.initial_cost
                    histories[i, : len(h)] = h
                    histories[i, len(h) :] = h[-1]

                mean = np.nanmean(histories, axis=0)
                std = np.nanstd(histories, axis=0)

                z = 1.96 if confidence == 0.95 else 1.645
                lower = mean - z * std / np.sqrt(len(run_results))
                upper = mean + z * std / np.sqrt(len(run_results))

                x = np.arange(max_len)
                color = self.style.get_algorithm_color(name)

                ax.plot(
                    x,
                    mean,
                    label=name,
                    color=color,
                    linewidth=self.style.line_width_thick,
                )

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2,
                )

            ax.set_xlabel('Iterations')
            ylabel = 'Normalized Cost' if normalize else 'Travel Cost'
            ax.set_ylabel(ylabel)

            ax.legend(
                loc='upper right',
                frameon=False,
            )

            ax.format(
                title=f'Convergence with {int(confidence * 100)}% CI',
            )

            return self.save(fig, 'convergence_with_ci')
        finally:
            uplt.close(fig)

    def efficiency_comparison(
        self,
        results: dict[str, list[OptimizationResult]],
    ) -> Path:
        """Generate efficiency comparison chart (improvement per evaluation).

        Args:
            results: Dict mapping algorithm names to list of OptimizationResult.

        Returns:
            Path to saved figure.
        """
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN, aspect_ratio=0.5, ncols=2, sharey=False
        )

        try:
            algorithm_names = list(results.keys())
            improvement_rates = []
            eval_efficiency = []
            colors = []

            for name in algorithm_names:
                run_results = results[name]

                rates = [
                    r.improvement_ratio / max(r.time_seconds, 0.001)
                    for r in run_results
                ]
                improvement_rates.append(np.mean(rates))

                efficiencies = [
                    r.improvement_ratio / max(r.evaluations, 1) * 1000
                    for r in run_results
                ]
                eval_efficiency.append(np.mean(efficiencies))

                colors.append(self.style.get_algorithm_color(name))

            positions = np.arange(len(algorithm_names))

            axs[0].bar(
                positions,
                improvement_rates,
                width=0.6,
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=self.style.axis_line_width,
            )
            axs[0].set_xticks(positions)
            axs[0].set_xticklabels(algorithm_names)
            axs[0].set_ylabel('Improvement Rate (per second)')
            axs[0].set_title('Time Efficiency')

            axs[1].bar(
                positions,
                eval_efficiency,
                width=0.6,
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=self.style.axis_line_width,
            )
            axs[1].set_xticks(positions)
            axs[1].set_xticklabels(algorithm_names)
            axs[1].set_ylabel('Improvement per 1000 Evaluations')
            axs[1].set_title('Evaluation Efficiency')

            axs.format(abc='a.', abcloc='ul')

            return self.save(fig, 'efficiency_comparison')
        finally:
            uplt.close(fig)

    @staticmethod
    def _compute_pareto_mask(objectives: np.ndarray) -> np.ndarray:
        """Compute mask for Pareto-optimal solutions (minimization).

        Args:
            objectives: Array of shape (n_solutions, n_objectives).

        Returns:
            Boolean mask of non-dominated solutions.
        """
        n = len(objectives)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                if np.all(objectives[j] <= objectives[i]) and np.any(
                    objectives[j] < objectives[i]
                ):
                    is_pareto[i] = False
                    break

        return is_pareto

    @staticmethod
    def _compute_spread(pareto_points: np.ndarray) -> float:
        """Compute spread metric for Pareto front.

        Args:
            pareto_points: Array of Pareto-optimal points.

        Returns:
            Spread value (range of objectives covered).
        """
        if len(pareto_points) < 2:
            return 0.0

        ranges = np.ptp(pareto_points, axis=0)
        return float(np.sqrt(np.sum(ranges**2)))

    @staticmethod
    def _compute_hypervolume_2d(
        pareto_points: np.ndarray,
        reference_point: np.ndarray | None = None,
    ) -> float:
        """Compute 2D hypervolume indicator.

        Args:
            pareto_points: Array of Pareto-optimal points (2D).
            reference_point: Reference point for hypervolume calculation.

        Returns:
            Hypervolume value.
        """
        if len(pareto_points) == 0 or pareto_points.shape[1] < 2:
            return 0.0

        ref = (
            reference_point
            if reference_point is not None
            else np.max(pareto_points, axis=0) * 1.1
        )

        sorted_idx = np.argsort(pareto_points[:, 0])
        points = pareto_points[sorted_idx]

        hv = 0.0
        prev_x = float(ref[0])

        for i in range(len(points) - 1, -1, -1):
            width = prev_x - points[i, 0]
            height = ref[1] - points[i, 1]
            hv += width * height
            prev_x = float(points[i, 0])

        return hv
