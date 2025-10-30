from pathlib import Path
import typer
from datetime import datetime
from typing import Optional
from typing_extensions import Annotated
from loguru import logger

from src.network_generator import NetworkGenerator
from src.optimize_manager import OptimizeManager
from src.utils.logger import setup_logger
from src.config import config_loader

config = config_loader.ConfigLoader()
setup_logger(log_file=Path(config.paths.log_dir) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log")

logger = logger.bind(module=__name__)
app = typer.Typer()


@app.command()
def network(
    image_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--image-dir",
            "-i",
            help="Floor annotation images directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    vis_output: Annotated[
        str, typer.Option("--vis-output", "-v", help="Network visualization output filename")
    ] = "hospital_network_3d.html",
    travel_times_output: Annotated[
        str, typer.Option("--travel-times-output", "-t", help="Travel times matrix output filename")
    ] = "hospital_travel_times.csv",
    slots_output: Annotated[
        str, typer.Option("--slots-output", "-s", help="SLOT nodes output filename")
    ] = "slots.csv",
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
            logger.info("System execution completed successfully")
        else:
            logger.error("System execution failed")
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("System execution error")
        raise typer.Exit(code=1)

@app.command()
def train():
    optimize_manager = OptimizeManager(config)
    optimize_manager.run()

if __name__ == "__main__":
    app()
