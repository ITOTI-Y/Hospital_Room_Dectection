"""
Main entry point for the network generation and analysis application.

This script demonstrates how to:
1. Configure logging.
2. Load application configurations.
3. Build a single-floor network (optional).
4. Build a multi-floor super-network.
5. Plot the generated network(s) using Plotly.
6. Calculate and save room-to-room/out-door travel times.
7. Generate and evaluate workflow paths.
"""
import logging
import sys
import pathlib  # For path operations
import os  # For os.path.join if needed, but prefer pathlib
from typing import Dict, List, Optional

# 
import cProfile
import snakeviz

# Import necessary modules from the 'src' package
# Assuming COLOR_MAP is still defined in config
from src.config import NetworkConfig, COLOR_MAP
from src.network.network import Network
from src.network.super_network import SuperNetwork
from src.plotting.plotter import PlotlyPlotter
from src.analysis.travel_time import calculate_room_travel_times

# Analysis modules
from src.analysis.process_flow import PathFinder
from src.analysis.word_detect import WordDetect

# Optimization modules
from src.optimization.optimizer import (
    PhysicalLocation,
    FunctionalAssignment,
    WorkflowDefinition,
    LayoutObjectiveCalculator,  # Not directly used by main, but optimizer uses it
    LayoutOptimizer,
    EvaluatedWorkflowOutcome
)

# --- Global Logger Setup ---


def setup_logging(level=logging.INFO, log_file: Optional[pathlib.Path] = None):
    """
    Configures basic logging for the application.

    Args:
        level: The minimum logging level to output (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should also be written.
    """
    root_logger = logging.getLogger()  # Get the root logger

    # Prevent adding handlers multiple times if this function is called again
    if root_logger.hasHandlers():
        # If you want to clear and reconfigure, uncomment the next line
        # for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        # For now, if handlers exist, assume it's configured.
        # Or, more robustly, check for specific handler types if needed.
        pass  # Already configured

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)  # Console can have its own level
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)  # File can also have its own level
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)  # Set the overall level for the root logger

# --- Main Application Logic ---


def run_single_floor_example(app_config: NetworkConfig, app_color_map: Dict):
    """
    Demonstrates building and plotting a single-floor network.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Running Single-Floor Network Example ---")

    # Define the image path for the single floor
    # Ensure this path is correct for your setup
    single_image_path_str = "./data/label/1F-meng.png"  # Example path
    single_image_path = pathlib.Path(single_image_path_str)

    if not single_image_path.exists():
        logger.error(
            f"Single floor image not found: {single_image_path}. Skipping single-floor example.")
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
            image_path=str(single_image_path),  # Network.run expects str
            z_level=0,  # Example Z-level for the single floor
            process_outside_nodes=False
        )

        logger.info(
            f"Single-floor network generated with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # Plot the single-floor network
        plotter = PlotlyPlotter(
            config=app_config, color_map_data=app_color_map)
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
        logger.error(
            f"Label directory not found: {label_dir}. Skipping multi-floor example.")
        return

    # Collect image paths (e.g., all PNG files in the directory)
    # Using a list of strings as SuperNetwork.run expects List[str]
    image_file_paths_str: List[str] = [
        str(p) for p in sorted(label_dir.glob('*.png')) if p.is_file()]
    # Example: image_file_paths_str = ["./data/label/B1.png", "./data/label/1F.png", "./data/label/2F.png"]

    if not image_file_paths_str:
        logger.warning(
            f"No image files found in {label_dir}. Skipping multi-floor example.")
        return

    logger.info(
        f"Found {len(image_file_paths_str)} images for SuperNetwork: {image_file_paths_str}")

    try:
        # Initialize SuperNetwork builder
        # base_floor=-1 if your floors are like B-1, F1, F2...
        # num_processes can be app_config.NUM_PROCESSES if defined, or None for auto
        super_network_builder = SuperNetwork(
            config=app_config,
            color_map_data=app_color_map,
            base_floor=-1,  # Example: if B-1 is the lowest floor detected/assigned as -1
            num_processes=None  # Use os.cpu_count() or 1 if single core
        )

        # Run the SuperNetwork generation
        super_graph = super_network_builder.run(
            image_file_paths=image_file_paths_str)

        ground_floor_z_for_travel_calc = super_network_builder.designated_ground_floor_z
        if ground_floor_z_for_travel_calc is not None:
            logger.info(
                f"Using designated ground floor Z={ground_floor_z_for_travel_calc:.2f} for travel time 'OutDoor' filtering.")
        else:
            logger.warning(
                "Could not determine designated ground floor Z from SuperNetwork. 'OutDoor' filtering in travel times may be affected.")

        logger.info(
            f"SuperNetwork generated with {super_graph.number_of_nodes()} nodes and {super_graph.number_of_edges()} edges.")
        logger.info(
            f"Detected image dimensions for SuperNetwork: Width={super_network_builder.width}, Height={super_network_builder.height}")

        # Plot the SuperNetwork
        plotter = PlotlyPlotter(
            config=app_config, color_map_data=app_color_map)
        plot_output_path = app_config.RESULT_PATH / "super_network_3d.html"
        plotter.plot(
            graph=super_graph,
            output_path=plot_output_path,
            title="Multi-Floor SuperNetwork",
            graph_width=super_network_builder.width,  # Pass determined width
            graph_height=super_network_builder.height,  # Pass determined height
            floor_z_map=super_network_builder.floor_z_map  # Pass for slider labels
        )
        logger.info(f"SuperNetwork plot saved to {plot_output_path}")

        # Calculate and save room travel times for the SuperNetwork
        logger.info("Calculating travel times for SuperNetwork...")
        travel_times_output_dir = app_config.RESULT_PATH
        calculate_room_travel_times(
            graph=super_graph,
            config=app_config,
            output_dir=travel_times_output_dir,
            # Specific name for super_network results
            output_filename="super_network_travel_times.csv",
            ground_floor_z=ground_floor_z_for_travel_calc
        )

    except Exception as e:
        logger.error(f"Error in multi-floor example: {e}", exc_info=True)


def run_layout_optimization_example(app_config: NetworkConfig):
    main_logger = logging.getLogger(__name__)
    main_logger.info("--- Running Facility Layout Optimization Example ---")

    # 1. Initialize PathFinder (it loads the travel_times.csv)
    # This CSV represents the fixed physical network.
    try:
        # Assumes CSV is at default path
        path_finder = PathFinder(config=app_config)
        if path_finder.travel_times_df is None:
            main_logger.error(
                "Failed to load travel times data in PathFinder. Optimization cannot proceed.")
            return
    except Exception as e:
        main_logger.error(f"Error initializing PathFinder: {e}", exc_info=True)
        return

    # 2. Define Workflows
    # These are sequences of *functional types*.
    word_detect = WordDetect(config=app_config)
    workflow_defs = [
        WorkflowDefinition(workflow_id='WF_GynecologyA',
                           functional_sequence=word_detect.detect_nearest_word(
                               ['入口', '妇科', '采血处', '超声科', '妇科', '门诊药房', '入口']),
                           weight=1.0),
        # WorkflowDefinition(workflow_id='WF_GynecologyB',
        #                    functional_sequence=word_detect.detect_nearest_word(
        #                        ['入口', '挂号收费', '妇科', '挂号收费', '采血处', '超声科', '妇科', '挂号收费', '门诊药房', '入口']),
        #                    weight=1.0),
        # WorkflowDefinition(workflow_id='WF_PulmonologyA',
        #                    functional_sequence=word_detect.detect_nearest_word(
        #                        ['入口', '呼吸内科', '采血处', '放射科', '呼吸内科', '门诊药房', '入口']),
        #                    weight=0.5),
        # WorkflowDefinition(workflow_id='WF_PulmonologyB',
        #                    functional_sequence=word_detect.detect_nearest_word(
        #                        ['入口', '挂号收费', '呼吸内科', '挂号收费', '采血处', '放射科', '呼吸内科', '挂号收费', '门诊药房', '入口']),
        #                    weight=0.5),
        # WorkflowDefinition(workflow_id='WF_CardiologyA',
        #                    functional_sequence=word_detect.detect_nearest_word(
        #                        ['入口', '心血管内科', '采血处', '超声科', '放射科', '心血管内科', '门诊药房', '入口']),
        #                    weight=1.2),
        # WorkflowDefinition(workflow_id='WF_CardiologyB',
        #                    functional_sequence=word_detect.detect_nearest_word(
        #                        ['入口', '挂号收费', '心血管内科', '挂号收费', '采血处', '超声科', '放射科', '心血管内科', '挂号收费', '门诊药房', '入口']),
        #                    weight=1.2),
    ]
    main_logger.info(
        f"Defined {len(workflow_defs)} workflows for optimization.")

    # 3. Create Initial FunctionalAssignment
    # This typically comes from the default `name_to_ids_map` in PathFinder,
    # which reflects the "as-is" layout from the original drawings.
    initial_assignment_map = path_finder.name_to_ids_map
    if not initial_assignment_map:
        main_logger.error(
            "PathFinder's name_to_ids_map is empty. Cannot create initial assignment.")
        return
    initial_functional_assignment = FunctionalAssignment(
        initial_assignment_map)
    main_logger.info(
        "Initial functional assignment created based on PathFinder's default map.")

    # 4. Initialize and Run Optimizer
    optimizer = LayoutOptimizer(
        path_finder=path_finder,
        workflow_definitions=workflow_defs,
        config=app_config,
        area_tolerance_ratio=0.3  # TODO: Allow up to 30% area difference for swaps
    )

    best_assignment, best_objective, best_outcomes = optimizer.run_optimization(
        initial_assignment=initial_functional_assignment,
        max_iterations=50  # TODO: Adjust as needed
    )

    # 5. Report Results
    main_logger.info("\n--- Optimization Results ---")
    main_logger.info(f"Final Optimized Objective Value: {best_objective:.2f}")

    main_logger.info(
        "\nFinal Functional Assignment (Functional Type -> [Physical Name_IDs]):")
    for func_type, phys_ids in sorted(best_assignment.assignment_map.items()):
        original_ids_for_type = sorted(
            initial_assignment_map.get(func_type, []))
        # Should already be sorted if FunctionalAssignment does it
        current_ids_for_type = sorted(phys_ids)

        changed_marker = ""
        if set(original_ids_for_type) != set(current_ids_for_type):  # Compare as sets for content
            changed_marker = " << MODIFIED"
            main_logger.info(f"  Function: {func_type}{changed_marker}")
            main_logger.info(
                f"    Original Locations: {original_ids_for_type}")
            main_logger.info(f"    New Locations     : {current_ids_for_type}")
        else:
            # Use debug for unchanged to reduce log noise, or info if you want to see all
            main_logger.debug(
                f"  Function: {func_type} -> Locations: {current_ids_for_type} (Unchanged)")

    main_logger.info("\nDetails of Optimized Workflows:")
    for outcome in best_outcomes:
        flow_details = "N/A"
        if outcome.shortest_flow:
            seq = outcome.shortest_flow.actual_node_id_sequence
            if seq:
                flow_details = f"[{seq[0]} ... {seq[-1]}] (len: {len(seq)})"
        main_logger.info(f"  Workflow: {outcome.workflow_definition.workflow_id} "
                         f"(Weight: {outcome.workflow_definition.weight:.1f}) - "
                         f"Optimized Time: {outcome.average_time:.2f} - Path: {flow_details}")


def initialize_setup():
    log_dir = pathlib.Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=log_dir / "application.log")

    main_logger = logging.getLogger(__name__)  # Logger for this main script
    main_logger.info("Application started.")

    app_config = NetworkConfig(color_map_data=COLOR_MAP)
    # Pass explicitly if needed, or rely on config's internal copy
    app_color_map_data = COLOR_MAP

    main_logger.info(f"Results will be saved in: {app_config.RESULT_PATH}")
    main_logger.info(
        f"Debug images (if any) will be saved in: {app_config.DEBUG_PATH}")

    return main_logger, app_config, app_color_map_data


if __name__ == "__main__":
    main_logger, app_config, app_color_map_data = initialize_setup()

    # ---- Example 1: Single-floor network (optional) ----
    # main_logger.info("Attempting to run single-floor example...")
    # run_single_floor_example(app_config, app_color_map_data)

    # ---- Example 2: Multi-floor SuperNetwork (primary use case) ----
    # main_logger.info("Attempting to run multi-floor SuperNetwork example...")
    # run_multi_floor_example(app_config, app_color_map_data)

    # ---- Example 3: Process Flow ----
    # main_logger.info("Attempting to run process flow example...")
    # workflow_list = ['大门', '妇科', '采血处', '超声科', '妇科', '门诊药房', '入口']
    # finder = PathFinder(config=app_config)
    # flows = finder.generate_flows(workflow_list)
    # for flow in flows:
    #     total_time = finder.calculate_flow_total_time(flow)

    # ---- Example 4: Layout Optimization ----
    travel_times_csv_path = app_config.RESULT_PATH / 'super_network_travel_times.csv'
    if not travel_times_csv_path.exists():
        main_logger.warning(f"{travel_times_csv_path} not found.")
        main_logger.warning("Please ensure `super_network_travel_times.csv` is generated first "
                            "(e.g., by running `run_multi_floor_example`).")
        main_logger.warning("Skipping layout optimization example.")
    else:
        profiler = cProfile.Profile()
        profiler.enable()
        run_layout_optimization_example(app_config)
        profiler.disable()
        profiler.dump_stats("layout_opt.prof")

    main_logger.info("Application finished.")
