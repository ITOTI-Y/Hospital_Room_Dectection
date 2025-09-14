"""
医院网络生成系统主入口

一个用于生成医院多楼层网络图的命令行工具。
"""

from pathlib import Path
import typer
from typing_extensions import Annotated

# 导入核心模块
from src.network_generator import NetworkGenerator
from src.utils.setup import setup_logger

logger = setup_logger(__name__)
app = typer.Typer()


@app.command()
def generate_network(
    image_dir: Annotated[
        Path,
        typer.Option(
            "--image-dir",
            "-i",
            help="楼层标注图像目录",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("./data/label/"),
    vis_output: Annotated[
        str, typer.Option("--vis-output", "-v", help="网络可视化输出文件名")
    ] = "hospital_network_3d.html",
    travel_times_output: Annotated[
        str, typer.Option("--travel-times-output", "-t", help="行程时间矩阵输出文件名")
    ] = "hospital_travel_times.csv",
    slots_output: Annotated[
        str, typer.Option("--slots-output", "-s", help="SLOT节点信息输出文件名")
    ] = "slots.csv",
):
    """
    从楼层图像生成医院网络。
    """

    logger.info("=== 医院网络生成系统启动 ===")

    generator = NetworkGenerator()

    try:
        success = generator.run_complete_generation(
            image_dir=str(image_dir),
            visualization_filename=vis_output,
            travel_times_filename=travel_times_output,
            slots_filename=slots_output,
        )

        if success:
            logger.info("=== 系统执行成功完成 ===")
        else:
            logger.error("=== 系统执行失败 ===")
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"系统执行异常: {e}", exc_info=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
