"""
Manages project paths based on a centralized YAML configuration.

This module loads path definitions from `configs/paths.yaml`, resolves them
relative to the project root, and provides a simple interface to access
them as `pathlib.Path` objects. It supports variable interpolation to
avoid path repetition.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

_paths_cache: Optional[Dict[str, Path]] = None
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """
    Determines and returns the project root directory.

    The project root is assumed to be the parent directory of the 'src' folder.

    Returns:
        A Path object representing the project's root directory.
    """
    global _project_root
    if _project_root is None:
        # Assumes this file is in src/config/path_manager.py
        _project_root = Path(__file__).parent.parent.parent
    return _project_root


def get_path(name: str, create_if_not_exist: bool = False) -> Path:
    """
    Retrieves a resolved, absolute path by its name from the configuration.

    Args:
        name: The key of the path in the `paths.yaml` file (e.g., 'network_dir').
        create_if_not_exist: If True, ensures the directory exists.

    Returns:
        A Path object for the requested path.

    Raises:
        KeyError: If the path name is not found in the configuration.
    """
    global _paths_cache
    if _paths_cache is None:
        _load_paths()

    if _paths_cache and name in _paths_cache:
        path = _paths_cache[name]
        if create_if_not_exist and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    else:
        raise KeyError(f"Path '{name}' not found in paths configuration.")


def _load_paths():
    """
    Loads and resolves all paths from the YAML configuration file.

    This internal function reads `configs/paths.yaml`, resolves paths relative
    to the project root, and performs variable substitution.
    """
    global _paths_cache
    _paths_cache = {}
    root = get_project_root()
    config_path = root / "configs" / "paths.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Paths configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_paths = yaml.safe_load(f)

    # Simple interpolation for variables like {data_dir}
    resolved_paths = {}
    for key, value in raw_paths.items():
        # Format the string with already resolved paths
        formatted_value = value.format(**resolved_paths)
        resolved_paths[key] = formatted_value

    # Convert to absolute Path objects
    for key, value in resolved_paths.items():
        _paths_cache[key] = root / value
