"""
Main entry point for the network generation and analysis application.

This script demonstrates how to:
1. Configure logging.
2. Load application configurations.
3. Build a single-floor network (optional).
4. Build a multi-floor super-network.
5. Plot the generated network(s) using Plotly.
6. Calculate and save room-to-room/out-door travel times.
"""
import logging
import sys
import pathlib # For path operations
import os # For os.path.join if needed, but prefer pathlib
from typing import Dict, List, Optional

# Import necessary modules from the 'src' package
from src.config import NetworkConfig, COLOR_MAP # Assuming COLOR_MAP is still defined in config
from src.network.network import Network
from src.network.super_network import SuperNetwork
from src.plotting.plotter import PlotlyPlotter
from src.analysis.travel_time import calculate_room_travel_times

# --- Global Logger Setup ---
def setup_logging(level=logging.INFO, log_file: Optional[pathlib.Path] = None):
    """
    Configures basic logging for the application.

    Args:
        level: The minimum logging level to output (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should also be written.
    """
    root_logger = logging.getLogger() # Get the root logger
    
    # Prevent adding handlers multiple times if this function is called again
    if root_logger.hasHandlers():
        # If you want to clear and reconfigure, uncomment the next line
        # for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        # For now, if handlers exist, assume it's configured.
        # Or, more robustly, check for specific handler types if needed.
        pass # Already configured

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level) # Console can have its own level
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level) # File can also have its own level
        root_logger.addHandler(file_handler)
        
    root_logger.setLevel(level) # Set the overall level for the root logger

# --- Main Application Logic ---
def run_single_floor_example(app_config: NetworkConfig, app_color_map: Dict):
    """
    Demonstrates building and plotting a single-floor network.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Running Single-Floor Network Example ---")

    # Define the image path for the single floor
    # Ensure this path is correct for your setup
    single_image_path_str = "./data/label/1F-meng.png" # Example path
    single_image_path = pathlib.Path(single_image_path_str)

    if not single_image_path.exists():
        logger.error(f"Single floor image not found: {single_image_path}. Skipping single-floor example.")
        return

    try:
        # Initialize Network builder for a single floor
        # id_generator_start_value can be 1 for a standalone single network
        network_builder = Network(
            config=app_config,
            color_map_data=app_color_map,
            id_generator_start_value=1
        )

        # Run the network generation
        # process_outside_nodes=True to generate outside mesh if applicable for this floor
        graph, width, height, _ = network_builder.run(
            image_path=str(single_image_path), # Network.run expects str
            z_level=0, # Example Z-level for the single floor
            process_outside_nodes=False
        )

        logger.info(f"Single-floor network generated with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # Plot the single-floor network
        plotter = PlotlyPlotter(config=app_config, color_map_data=app_color_map)
        plot_output_path = app_config.RESULT_PATH / "single_floor_network_3d.html"
        plotter.plot(
            graph=graph,
            output_path=plot_output_path,
            title=f"Single-Floor Network: {single_image_path.name}",
            graph_width=width,
            graph_height=height
        )
        logger.info(f"Single-floor network plot saved to {plot_output_path}")

    except Exception as e:
        logger.error(f"Error in single-floor example: {e}", exc_info=True)


def run_multi_floor_example(app_config: NetworkConfig, app_color_map: Dict):
    """
    Demonstrates building, plotting, and analyzing a multi-floor super-network.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Running Multi-Floor SuperNetwork Example ---")

    # Define image paths for multiple floors
    # Ensure this directory and images exist
    label_dir_str = "./data/label/"
    label_dir = pathlib.Path(label_dir_str)

    if not label_dir.is_dir():
        logger.error(f"Label directory not found: {label_dir}. Skipping multi-floor example.")
        return

    # Collect image paths (e.g., all PNG files in the directory)
    # Using a list of strings as SuperNetwork.run expects List[str]
    image_file_paths_str: List[str] = [str(p) for p in sorted(label_dir.glob('*.png')) if p.is_file()]
    # Example: image_file_paths_str = ["./data/label/B1.png", "./data/label/1F.png", "./data/label/2F.png"]


    if not image_file_paths_str:
        logger.warning(f"No image files found in {label_dir}. Skipping multi-floor example.")
        return
    
    logger.info(f"Found {len(image_file_paths_str)} images for SuperNetwork: {image_file_paths_str}")


    try:
        # Initialize SuperNetwork builder
        # base_floor=-1 if your floors are like B-1, F1, F2...
        # num_processes can be app_config.NUM_PROCESSES if defined, or None for auto
        super_network_builder = SuperNetwork(
            config=app_config,
            color_map_data=app_color_map,
            base_floor=-1, # Example: if B-1 is the lowest floor detected/assigned as -1
            num_processes=None # Use os.cpu_count() or 1 if single core
        )

        # Run the SuperNetwork generation
        super_graph = super_network_builder.run(image_file_paths=image_file_paths_str)

        logger.info(f"SuperNetwork generated with {super_graph.number_of_nodes()} nodes and {super_graph.number_of_edges()} edges.")
        logger.info(f"Detected image dimensions for SuperNetwork: Width={super_network_builder.width}, Height={super_network_builder.height}")


        # Plot the SuperNetwork
        plotter = PlotlyPlotter(config=app_config, color_map_data=app_color_map)
        plot_output_path = app_config.RESULT_PATH / "super_network_3d.html"
        plotter.plot(
            graph=super_graph,
            output_path=plot_output_path,
            title="Multi-Floor SuperNetwork",
            graph_width=super_network_builder.width, # Pass determined width
            graph_height=super_network_builder.height, # Pass determined height
            floor_z_map=super_network_builder.floor_z_map # Pass for slider labels
        )
        logger.info(f"SuperNetwork plot saved to {plot_output_path}")

        # Calculate and save room travel times for the SuperNetwork
        logger.info("Calculating travel times for SuperNetwork...")
        travel_times_output_dir = app_config.RESULT_PATH
        calculate_room_travel_times(
            graph=super_graph,
            config=app_config,
            output_dir=travel_times_output_dir,
            output_filename="super_network_travel_times.csv" # Specific name for super_network results
        )

    except Exception as e:
        logger.error(f"Error in multi-floor example: {e}", exc_info=True)


if __name__ == "__main__":
    # --- Setup Logging ---
    # Create a log directory if it doesn't exist (for file logging)
    log_dir = pathlib.Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    # Setup logging to console and a file
    setup_logging(level=logging.INFO, log_file=log_dir / "application.log")
    
    main_logger = logging.getLogger(__name__) # Logger for this main script
    main_logger.info("Application started.")

    # --- Load Configuration ---
    # COLOR_MAP is imported directly, NetworkConfig uses it by default
    # If COLOR_MAP were in a separate file, it would be loaded here.
    app_config = NetworkConfig(color_map_data=COLOR_MAP)
    app_color_map_data = COLOR_MAP # Pass explicitly if needed, or rely on config's internal copy

    main_logger.info(f"Results will be saved in: {app_config.RESULT_PATH}")
    main_logger.info(f"Debug images (if any) will be saved in: {app_config.DEBUG_PATH}")


    # --- Run Examples ---
    # You can choose to run one or both examples.

    # Example 1: Single-floor network (optional)
    # main_logger.info("Attempting to run single-floor example...")
    # run_single_floor_example(app_config, app_color_map_data)
    
    # Example 2: Multi-floor SuperNetwork (primary use case)
    main_logger.info("Attempting to run multi-floor SuperNetwork example...")
    run_multi_floor_example(app_config, app_color_map_data)

    main_logger.info("Application finished.")