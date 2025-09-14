"""网络生成器 - 整合医院网络生成功能."""

from pathlib import Path
from typing import Dict, Optional, Any
import networkx as nx

from src.utils.setup import setup_logger
from src.config import path_manager
from src.network.super_network import SuperNetwork
from src.plotting.plotter import PlotlyPlotter
from src.analysis.travel_time import calculate_room_travel_times
from src.analysis.slots_exporter import export_slots_to_csv

logger = setup_logger(__name__)


class NetworkGenerator:
    """
    网络生成器类

    整合原有的医院网络生成功能，从楼层平面图生成多层医院网络图，
    包括图像处理、网络构建和行程时间计算。
    """

    def __init__(self):
        """
        初始化网络生成器.
        """
        self.super_network: Optional[SuperNetwork] = None
        self.super_graph: Optional[nx.Graph] = None

        logger.info("网络生成器初始化完成")
        logger.info(f"结果将保存在: {path_manager.get_path('network_dir')}")
        logger.info(f"调试图像将保存在: {path_manager.get_path('debug_dir')}")

    def generate_network(
        self,
        image_dir: Optional[str] = None,
        base_floor: int = 0,
        num_processes: Optional[int] = None,
    ) -> bool:
        """
        生成多层医院网络

        Args:
            image_dir: 楼层标注图像目录 (可选, 默认为 paths.yaml 中的 'label_dir').
            base_floor: 基准楼层.
            num_processes: 并行处理进程数.

        Returns:
            bool: 生成是否成功.
        """
        logger.info("=== 开始生成多层医院网络 ===")

        label_dir = Path(image_dir) if image_dir else path_manager.get_path("label_dir")
        if not label_dir.is_dir():
            logger.error(f"标注图像目录不存在: {label_dir}")
            return False

        image_file_paths = [
            str(p) for p in sorted(label_dir.glob("*.png")) if p.is_file()
        ]

        if not image_file_paths:
            logger.warning(f"在 {label_dir} 中未找到图像文件")
            return False

        logger.info(f"找到 {len(image_file_paths)} 个图像文件: {image_file_paths}")

        try:
            self.super_network = SuperNetwork(
                base_floor=base_floor, num_processes=num_processes
            )

            self.super_graph = self.super_network.run(image_file_paths=image_file_paths)

            if self.super_graph:
                logger.info("SuperNetwork生成成功:")
                logger.info(f"  节点数: {self.super_graph.number_of_nodes()}")
                logger.info(f"  边数: {self.super_graph.number_of_edges()}")
            logger.info(
                f"  图像尺寸: 宽度={self.super_network.width}, 高度={self.super_network.height}"
            )

            return True

        except Exception as e:
            logger.error(f"生成网络时发生错误: {e}", exc_info=True)
            return False

    def visualize_network(
        self, output_filename: str = "hospital_network_3d.html"
    ) -> bool:
        """
        可视化生成的网络

        Args:
            output_filename: 输出文件名

        Returns:
            bool: 可视化是否成功
        """
        if self.super_graph is None or self.super_network is None:
            logger.error("未生成网络，无法进行可视化")
            return False

        try:
            logger.info("正在生成网络可视化...")

            plotter = PlotlyPlotter()

            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            plot_output_path = network_path / output_filename
            plotter.plot(
                graph=self.super_graph,
                output_path=str(plot_output_path),
                title="多层医院网络",
                graph_width=self.super_network.width,
                floor_z_map=self.super_network.floor_z_map,
            )

            logger.info(f"网络可视化已保存到: {plot_output_path}")
            return True

        except Exception as e:
            logger.error(f"网络可视化时发生错误: {e}", exc_info=True)
            return False

    def calculate_travel_times(
        self, output_filename: str = "hospital_travel_times.csv"
    ) -> bool:
        """
        计算并保存房间间行程时间

        Args:
            output_filename: 输出文件名

        Returns:
            bool: 计算是否成功
        """
        if self.super_graph is None:
            logger.error("未生成网络，无法计算行程时间")
            return False

        try:
            logger.info("正在计算房间间行程时间...")

            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            calculate_room_travel_times(
                graph=self.super_graph,
                output_dir=network_path,
                output_filename=output_filename,
            )

            travel_times_path = network_path / output_filename
            logger.info(f"行程时间矩阵已保存到: {travel_times_path}")
            return True

        except Exception as e:
            logger.error(f"计算行程时间时发生错误: {e}", exc_info=True)
            return False

    def export_slots(self, output_filename: str = "slots.csv") -> bool:
        """
        Exports SLOT nodes to a CSV file.

        Args:
            output_filename: The name of the output CSV file.

        Returns:
            bool: True if export was successful, False otherwise.
        """
        if self.super_graph is None:
            logger.error("未生成网络，无法导出 slots")
            return False

        try:
            logger.info("正在导出 SLOT 节点...")
            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            export_slots_to_csv(
                graph=self.super_graph,
                output_dir=network_path,
                output_filename=output_filename,
            )
            slots_path = network_path / output_filename
            logger.info(f"SLOT 节点已导出到: {slots_path}")
            return True
        except Exception as e:
            logger.error(f"导出 SLOT 节点时发生错误: {e}", exc_info=True)
            return False

    def get_network_info(self) -> Dict[str, Any]:
        """
        获取网络信息

        Returns:
            Dict: 网络信息字典
        """
        if self.super_graph is None or self.super_network is None:
            return {}

        return {
            "nodes_count": self.super_graph.number_of_nodes(),
            "edges_count": self.super_graph.number_of_edges(),
            "width": self.super_network.width,
            "height": self.super_network.height,
            "floor_z_map": self.super_network.floor_z_map,
            "ground_floor_z": self.super_network.designated_ground_floor_z,
        }

    def run_complete_generation(
        self,
        image_dir: Optional[str] = None,
        visualization_filename: str = "hospital_network_3d.html",
        travel_times_filename: str = "hospital_travel_times.csv",
        slots_filename: str = "slots.csv",
    ) -> bool:
        """
        运行完整的网络生成流程

        Args:
            image_dir: 楼层标注图像目录
            visualization_filename: 可视化输出文件名
            travel_times_filename: 行程时间输出文件名
            slots_filename: SLOT 节点输出文件名

        Returns:
            bool: 完整流程是否成功
        """
        logger.info("=== 开始完整网络生成流程 ===")

        if not self.generate_network(image_dir):
            logger.error("网络生成失败，中止流程")
            return False

        if not self.visualize_network(visualization_filename):
            logger.error("网络可视化失败，但继续后续步骤")

        if not self.calculate_travel_times(travel_times_filename):
            logger.error("行程时间计算失败")
            # Do not return False, as other steps might succeed

        if not self.export_slots(slots_filename):
            logger.error("SLOT 节点导出失败")

        logger.info("=== 完整网络生成流程成功完成 ===")
        network_info = self.get_network_info()
        logger.info(f"网络统计信息: {network_info}")

        return True
