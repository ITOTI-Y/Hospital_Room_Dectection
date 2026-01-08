from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch.multiprocessing as mp
import typer
from loguru import logger

from src.config import config_loader
from src.network_generator import NetworkGenerator
from src.optimize_manager import OptimizeManager
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
    optimize_manager = OptimizeManager(config)
    optimize_manager.run()


@app.command()
def train_trl():
    from src.trl.actor_critic import create_actor_critic
    from src.trl.encoder import DualStreamGNNEncoder
    from src.trl.env import create_eval_env, create_train_env
    from src.trl.trainer import create_trainer

    encoder = DualStreamGNNEncoder()
    actor_critic = create_actor_critic(encoder)

    trainer = create_trainer(
        env_maker=lambda: create_train_env(config),
        actor_critic=actor_critic,
        eval_env_maker=lambda: create_eval_env(config),
    )
    trainer.train()


if __name__ == '__main__':
    app()
