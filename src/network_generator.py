from pathlib import Path
from typing import Dict, Optional, Any, Tuple
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
    NetworkGenerator handles the complete process of generating a multi-floor hospital network,
    including network generation, visualization, travel time calculation, and SLOT export.
    """

    def __init__(self):
        """
        Initializes the NetworkGenerator.
        """
        self.super_network: Optional[SuperNetwork] = None
        self.super_graph: Optional[nx.Graph] = None

        logger.info("NetworkGenerator initialized.")
        logger.info(f"Results will be saved to: {path_manager.get_path('network_dir')}")
        logger.info(f"Debug images will be saved to: {path_manager.get_path('debug_dir')}")

    def generate_network(
        self,
        image_dir: Optional[Path] = None,
        base_floor: int = 0,
        num_processes: Optional[int] = None,
    ) -> bool:
        """
        Generates the multi-floor hospital network from floor annotation images.

        Args:
            image_dir: Floor annotation images directory.
            base_floor: Base floor.
            num_processes: Number of parallel processing.

        Returns:
            bool: Whether the generation was successful.
        """
        logger.info("Starting multi floor network generation...")

        if image_dir is None:
            image_dir = path_manager.get_path("label_dir")

        if not image_dir.is_dir():
            logger.error(f"Floor annotation images directory does not exist: {image_dir}")
            return False

        image_file_paths = [
            p for p in sorted(image_dir.glob("*.png")) if p.is_file()
        ]

        if not image_file_paths:
            logger.warning(f"Image files is not found in {image_dir}")
            return False

        logger.info(f"Found {len(image_file_paths)} image files: {image_file_paths}")

        try:
            self.super_network = SuperNetwork(
                base_floor=base_floor, num_processes=num_processes
            )

            self.super_graph = self.super_network.run(image_file_paths=image_file_paths)

            if self.super_graph:
                logger.info("Complete multi floor network generation successful")
                logger.info(f"  Nodes: {self.super_graph.number_of_nodes()}")
                logger.info(f"  Edges: {self.super_graph.number_of_edges()}")
            logger.info(
                f"  Image size: Width={self.super_network.width}, Height={self.super_network.height}"
            )

            return True

        except Exception as e:
            logger.error(f"Error occurred while generating network: {e}", exc_info=True)
            return False

    def visualize_network(
        self, output_filename: str = "hospital_network_3d.html"
    ) -> bool:
        """
        Visualizes the generated network.

        Args:
            output_filename: Output file name.

        Returns:
            bool: whether the visualization was successful.
        """
        if self.super_graph is None or self.super_network is None:
            logger.error("No generated network to visualize")
            return False

        try:
            plotter = PlotlyPlotter()

            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            plot_output_path = network_path / output_filename
            plotter.plot(
                graph=self.super_graph,
                output_path=str(plot_output_path),
                title="Multi-Floor Hospital Network",
                graph_width=self.super_network.width,
                floor_z_map=self.super_network.floor_z_map,
            )

            logger.info(f"Network visualization saved to: {plot_output_path}")
            return True

        except Exception as e:
            logger.error(f"Error occurred while visualizing network: {e}", exc_info=True)
            return False

    def calculate_travel_times(
        self, output_filename: str = "hospital_travel_times.csv"
    ) -> bool:
        """
        Calculates and saves travel times between rooms.

        Args:
            output_filename: Output file name.

        Returns:
            bool: whether the calculation was successful.
        """
        if self.super_graph is None:
            logger.error("No generated network to calculate travel times")
            return False

        try:
            logger.info("Calculating travel times between rooms...")

            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            calculate_room_travel_times(
                graph=self.super_graph,
                output_dir=network_path,
                output_filename=output_filename,
            )

            travel_times_path = network_path / output_filename
            logger.info(f"Travel times matrix saved to: {travel_times_path}")
            return True

        except Exception as e:
            logger.error(f"Error occurred while calculating travel times: {e}", exc_info=True)
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
            logger.error("No generated network to export slots")
            return False

        try:
            logger.info("Exporting SLOT nodes...")
            network_path = path_manager.get_path(
                "network_dir", create_if_not_exist=True
            )
            export_slots_to_csv(
                graph=self.super_graph,
                output_dir=network_path,
                output_filename=output_filename,
            )
            slots_path = network_path / output_filename
            logger.info(f"SLOT Nodes exported to: {slots_path}")
            return True
        except Exception as e:
            logger.error(f"Error occurred while exporting SLOT nodes: {e}", exc_info=True)
            return False

    def get_network_info(self) -> Dict[str, Any]:
        """
        Gets basic information about the generated network.

        Returns:
            Dict: Network information dictionary.
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
        image_dir: Optional[Path] = None,
        visualization_filename: str = "hospital_network_3d.html",
        travel_times_filename: str = "hospital_travel_times.csv",
        slots_filename: str = "slots.csv",
    ) -> bool:
        """
        Runs the complete network generation process.

        Args:
            image_dir: floor annotation images directory
            visualization_filename: network visualization output filename
            travel_times_filename: travel times output filename
            slots_filename: SLOT nodes output filename

        Returns:
            bool: whether the complete process was successful
        """
        logger.info("Starting complete network generation process...")

        if not self.generate_network(image_dir):
            logger.error("Network generation failed, aborting process")
            return False

        if not self.visualize_network(visualization_filename):
            logger.error("Error occurred while visualizing network")

        if not self.calculate_travel_times(travel_times_filename):
            logger.error("Error occurred while calculating travel times")
            # Do not return False, as other steps might succeed

        if not self.export_slots(slots_filename):
            logger.error("Error occurred while exporting SLOT nodes")

        if self.super_graph:
            import pickle
            graph_path = path_manager.get_path("network_dir", create_if_not_exist=True)
            with open(graph_path / "hospital_network.pkl", "wb") as f:
                pickle.dump(self.super_graph, f)

        logger.info("Complete network generation process completed successfully")
        network_info = self.get_network_info()
        logger.info(f"Network statistics: {network_info}")

        return True
