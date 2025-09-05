"""
网络生成器 - 整合医院网络生成功能
"""

from src.rl_optimizer.utils.setup import setup_logger
import pathlib
from typing import Dict, List, Optional

from src.config import NetworkConfig, COLOR_MAP
from src.network.super_network import SuperNetwork
from src.plotting.plotter import PlotlyPlotter
from src.analysis.travel_time import calculate_room_travel_times

logger = setup_logger(__name__)


class NetworkGenerator:
    """
    网络生成器类
    
    整合原有的医院网络生成功能，从楼层平面图生成多层医院网络图，
    包括图像处理、网络构建和行程时间计算。
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        初始化网络生成器
        
        Args:
            config: 网络配置，如果为None则使用默认配置
        """
        self.config = config if config is not None else NetworkConfig(color_map_data=COLOR_MAP)
        self.color_map_data = COLOR_MAP
        self.super_network = None
        self.super_graph = None
        
        logger.info("网络生成器初始化完成")
        logger.info(f"结果将保存在: {self.config.RESULT_PATH}")
        logger.info(f"调试图像将保存在: {self.config.DEBUG_PATH}")
    
    def generate_network(self, 
                        image_dir: str = "./data/label/",
                        base_floor: int = 0,
                        num_processes: Optional[int] = None) -> bool:
        """
        生成多层医院网络
        
        Args:
            image_dir: 楼层标注图像目录
            base_floor: 基准楼层
            num_processes: 并行处理进程数
            
        Returns:
            bool: 生成是否成功
        """
        logger.info("=== 开始生成多层医院网络 ===")
        
        # 检查图像目录
        label_dir = pathlib.Path(image_dir)
        if not label_dir.is_dir():
            logger.error(f"标注图像目录不存在: {label_dir}")
            return False
        
        # 收集图像文件
        image_file_paths = [
            str(p) for p in sorted(label_dir.glob('*.png')) if p.is_file()
        ]
        
        if not image_file_paths:
            logger.warning(f"在 {label_dir} 中未找到图像文件")
            return False
        
        logger.info(f"找到 {len(image_file_paths)} 个图像文件: {image_file_paths}")
        
        try:
            # 初始化SuperNetwork构建器
            self.super_network = SuperNetwork(
                config=self.config,
                color_map_data=self.color_map_data,
                base_floor=base_floor,
                num_processes=num_processes
            )
            
            # 生成SuperNetwork
            self.super_graph = self.super_network.run(
                image_file_paths=image_file_paths
            )
            
            logger.info(f"SuperNetwork生成成功:")
            logger.info(f"  节点数: {self.super_graph.number_of_nodes()}")
            logger.info(f"  边数: {self.super_graph.number_of_edges()}")
            logger.info(f"  图像尺寸: 宽度={self.super_network.width}, 高度={self.super_network.height}")
            
            return True
            
        except Exception as e:
            logger.error(f"生成网络时发生错误: {e}", exc_info=True)
            return False
    
    def visualize_network(self, output_filename: str = "hospital_network_3d.html") -> bool:
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
            
            plotter = PlotlyPlotter(
                config=self.config, 
                color_map_data=self.color_map_data
            )
            
            plot_output_path = self.config.RESULT_PATH / output_filename
            plotter.plot(
                graph=self.super_graph,
                output_path=plot_output_path,
                title="多层医院网络",
                graph_width=self.super_network.width,
                graph_height=self.super_network.height,
                floor_z_map=self.super_network.floor_z_map
            )
            
            logger.info(f"网络可视化已保存到: {plot_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"网络可视化时发生错误: {e}", exc_info=True)
            return False
    
    def calculate_travel_times(self, output_filename: str = "hospital_travel_times.csv") -> bool:
        """
        计算并保存房间间行程时间
        
        Args:
            output_filename: 输出文件名
            
        Returns:
            bool: 计算是否成功
        """
        if self.super_graph is None or self.super_network is None:
            logger.error("未生成网络，无法计算行程时间")
            return False
        
        try:
            logger.info("正在计算房间间行程时间...")
            
            # 获取地面楼层Z值用于过滤
            ground_floor_z = self.super_network.designated_ground_floor_z
            if ground_floor_z is not None:
                logger.info(f"使用地面楼层 Z={ground_floor_z:.2f} 进行外部区域过滤")
            else:
                logger.warning("无法确定地面楼层Z值，外部区域过滤可能受影响")
            
            # 计算并保存行程时间
            travel_times_output_dir = self.config.RESULT_PATH
            calculate_room_travel_times(
                graph=self.super_graph,
                config=self.config,
                output_dir=travel_times_output_dir,
                output_filename=output_filename,
                ground_floor_z=ground_floor_z
            )
            
            travel_times_path = travel_times_output_dir / output_filename
            logger.info(f"行程时间矩阵已保存到: {travel_times_path}")
            return True
            
        except Exception as e:
            logger.error(f"计算行程时间时发生错误: {e}", exc_info=True)
            return False
    
    def get_network_info(self) -> Dict[str, any]:
        """
        获取网络信息
        
        Returns:
            Dict: 网络信息字典
        """
        if self.super_graph is None or self.super_network is None:
            return {}
        
        return {
            'nodes_count': self.super_graph.number_of_nodes(),
            'edges_count': self.super_graph.number_of_edges(),
            'width': self.super_network.width,
            'height': self.super_network.height,
            'floor_z_map': self.super_network.floor_z_map,
            'ground_floor_z': self.super_network.designated_ground_floor_z
        }
    
    def run_complete_generation(self, 
                               image_dir: str = "./data/label/",
                               visualization_filename: str = "hospital_network_3d.html",
                               travel_times_filename: str = "hospital_travel_times.csv") -> bool:
        """
        运行完整的网络生成流程
        
        Args:
            image_dir: 楼层标注图像目录
            visualization_filename: 可视化输出文件名
            travel_times_filename: 行程时间输出文件名
            
        Returns:
            bool: 完整流程是否成功
        """
        logger.info("=== 开始完整网络生成流程 ===")
        
        # 1. 生成网络
        if not self.generate_network(image_dir):
            logger.error("网络生成失败，中止流程")
            return False
        
        # 2. 可视化网络
        if not self.visualize_network(visualization_filename):
            logger.error("网络可视化失败，但继续后续步骤")
        
        # 3. 计算行程时间
        if not self.calculate_travel_times(travel_times_filename):
            logger.error("行程时间计算失败，但网络生成成功")
            return False
        
        logger.info("=== 完整网络生成流程成功完成 ===")
        network_info = self.get_network_info()
        logger.info(f"网络统计信息: {network_info}")
        
        return True