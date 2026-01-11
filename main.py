from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch.multiprocessing as mp
import typer
from loguru import logger

from src.config import config_loader
from src.network_generator import NetworkGenerator
from src.utils.logger import setup_logger

mp.set_start_method('spawn', force=True)
config = config_loader.ConfigLoader()
setup_logger(
    log_file=Path(config.paths.log_dir) / f'{datetime.now():%Y-%m-%d_%H-%M-%S}.log'
)

logger = logger.bind(module=__name__)
app = typer.Typer()


@app.command()
def network(
    image_dir: Annotated[
        Path | None,
        typer.Option(
            '--image-dir',
            '-i',
            help='Floor annotation images directory',
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    vis_output: Annotated[
        str,
        typer.Option(
            '--vis-output', '-v', help='Network visualization output filename'
        ),
    ] = 'hospital_network_3d.html',
    travel_times_output: Annotated[
        str,
        typer.Option(
            '--travel-times-output', '-t', help='Travel times matrix output filename'
        ),
    ] = 'hospital_travel_times.csv',
    slots_output: Annotated[
        str, typer.Option('--slots-output', '-s', help='SLOT nodes output filename')
    ] = 'slots.csv',
):
    generator = NetworkGenerator(config)

    try:
        success = generator.run_complete_generation(
            image_dir=image_dir,
            visualization_filename=vis_output,
            travel_times_filename=travel_times_output,
            slots_filename=slots_output,
        )

        if success:
            logger.info('System execution completed successfully')
        else:
            logger.error('System execution failed')
            raise typer.Exit(code=1)

    except KeyboardInterrupt as e:
        logger.warning('Interrupted by user')
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.exception('System execution error')
        raise typer.Exit(code=1) from e


@app.command()
def train():
    from src.rl.actor_critic import create_actor_critic
    from src.rl.encoder import DualStreamGNNEncoder
    from src.rl.env import create_eval_env, create_train_env
    from src.rl.trainer import create_trainer

    encoder = DualStreamGNNEncoder()
    actor_critic = create_actor_critic(encoder)

    trainer = create_trainer(
        env_maker=lambda: create_train_env(config),
        actor_critic=actor_critic,
        eval_env_maker=lambda: create_eval_env(config),
    )
    trainer.train()


@app.command()
def baseline(
    algorithm: Annotated[
        str,
        typer.Option(
            '--algorithm',
            '-a',
            help='Algorithm to run: ga (Genetic Algorithm), sa (Simulated Annealing), or compare (both)',
        ),
    ] = 'compare',
    n_runs: Annotated[
        int,
        typer.Option('--n-runs', '-n', help='Number of runs per algorithm'),
    ] = 5,
    ga_iterations: Annotated[
        int,
        typer.Option('--ga-iter', help='Max generations for GA'),
    ] = 200,
    sa_iterations: Annotated[
        int,
        typer.Option('--sa-iter', help='Max iterations for SA'),
    ] = 10000,
    seed: Annotated[
        int,
        typer.Option('--seed', '-s', help='Random seed'),
    ] = 42,
    output_dir: Annotated[
        str,
        typer.Option('--output', '-o', help='Output directory for results'),
    ] = 'results/baseline',
):
    """Run baseline optimization algorithms (GA and/or SA)."""
    from src.baseline import BaselineRunner

    runner = BaselineRunner(config, shuffle_initial_layout=True)
    runner.initialize_pathways()

    if algorithm == 'compare':
        results = runner.run_comparison(
            n_runs=n_runs,
            ga_iterations=ga_iterations,
            sa_iterations=sa_iterations,
            base_seed=seed,
        )
        runner.export_results(results, output_dir)

    elif algorithm == 'ga':
        result = runner.run_genetic_algorithm(
            max_iterations=ga_iterations,
            seed=seed,
        )
        logger.info(
            f'GA Result: cost={result.best_cost:.2f}, '
            f'improvement={result.improvement_ratio:.2%}'
        )

    elif algorithm == 'sa':
        result = runner.run_simulated_annealing(
            max_iterations=sa_iterations,
            seed=seed,
        )
        logger.info(
            f'SA Result: cost={result.best_cost:.2f}, '
            f'improvement={result.improvement_ratio:.2%}'
        )

    else:
        logger.error(f"Unknown algorithm: {algorithm}. Use 'ga', 'sa', or 'compare'")
        raise typer.Exit(code=1)


if __name__ == '__main__':
    app()
