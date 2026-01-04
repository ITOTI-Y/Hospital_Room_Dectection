"""
Manages loading and accessing configuration from graph_config.yaml.

This module provides a centralized, read-only interface to the graph
generation configuration, ensuring that the YAML file is loaded only once.
It also builds convenient reverse mappings and helper functions to query
node definitions by various attributes like category or RGB color.
"""

from pathlib import Path
from typing import Any

import yaml

_config_cache: dict[str, Any] | None = None
_rgb_to_name_map_cache: dict[tuple[int, int, int], str] | None = None


def get_config() -> dict[str, Any] | None:
    """
    Loads graph configuration from YAML file and caches it.

    This function ensures the YAML configuration is read only once and
    provides a consistent, read-only dictionary of settings for the
    entire application.

    Returns:
        A dictionary containing the entire graph configuration.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    global _config_cache
    if _config_cache is None:
        config_path = (
            Path(__file__).parent.parent.parent / 'configs' / 'graph_config.yaml'
        )
        if not config_path.exists():
            raise FileNotFoundError(f'Configuration file not found at: {config_path}')

        with open(config_path, encoding='utf-8') as f:
            try:
                _config_cache = yaml.safe_load(f)
            except yaml.YAMLError as e:
                # Log or handle the error appropriately
                raise e
    return _config_cache


def get_node_definitions() -> dict[str, dict[str, Any]]:
    """
    Retrieves the node definitions section from the configuration.

    Returns:
        A dictionary where keys are node names (e.g., 'Wall') and
        values are dictionaries of their properties.
    """
    config = get_config()
    if config is None:
        return {}
    return config.get('node_definitions', {})


def get_rgb_to_name_map() -> dict[tuple[int, int, int], str]:
    """
    Creates and caches a mapping from RGB color tuples to node names.

    This is essential for the image processing step, allowing for quick
    identification of node types based on pixel color.

    Returns:
        A dictionary mapping RGB tuples to their corresponding node names.
    """
    global _rgb_to_name_map_cache
    if _rgb_to_name_map_cache is None:
        node_defs = get_node_definitions()
        _rgb_to_name_map_cache = {
            tuple(properties['rgb']): node_name
            for node_name, properties in node_defs.items()
            if 'rgb' in properties
        }
    return _rgb_to_name_map_cache


def get_nodes_by_category(category: str) -> list[str]:
    """
    Finds all node names belonging to a specific category.

    Args:
        category: The category to filter by (e.g., 'SLOT', 'PATH').

    Returns:
        A list of node names that match the given category.
    """
    node_defs = get_node_definitions()
    return [
        node_name
        for node_name, properties in node_defs.items()
        if properties.get('category') == category
    ]


def get_geometry_config() -> dict[str, float]:
    """
    Retrieves the geometry settings from the configuration.

    Returns:
        A dictionary with geometry-related parameters like scale and speed.
    """
    config = get_config()
    if config is None:
        return {}
    return config.get('geometry', {})


def get_super_network_config() -> dict[str, Any]:
    """
    Retrieves the super_network settings from the configuration.

    Returns:
        A dictionary with super_network-related parameters.
    """
    config = get_config()
    if config is None:
        return {}
    return config.get('super_network', {})


def get_special_ids() -> dict[str, int]:
    """
    Retrieves the special ID mappings from the configuration.

    Returns:
        A dictionary mapping special area names to their integer IDs.
    """
    config = get_config()
    if config is None:
        return {}
    return config.get('special_ids', {})


_plotter_config_cache: dict[str, Any] = {}


def get_plotter_config() -> dict[str, Any]:
    """
    Loads plotter configuration from plotter.yaml and caches it.

    Returns:
        A dictionary containing the plotter configuration.
    """
    global _plotter_config_cache
    if not _plotter_config_cache:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'plotter.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f'Plotter configuration file not found at: {config_path}'
            )

        with open(config_path, encoding='utf-8') as f:
            try:
                _plotter_config_cache = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise e
    return _plotter_config_cache
