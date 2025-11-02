# src/utils/logger.py

import json
import pathlib
import pickle
import sys
from typing import Any

import numpy as np
from loguru import logger


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Numpy data types and path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return super().default(obj)


def setup_logger(log_file: pathlib.Path | None = None, level: str = "INFO") -> None:
    """Configure loguru logger.

    Args:
        log_file (pathlib.Path | None): Optional log file path.
        level (str): Log level, default is "INFO".
    """

    logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]!s}</cyan> | "
        "<level>{message}</level>"
    )

    logger.configure(extra={"module": "unknown"})

    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[module]} | {message}\n"
            "{exception}"
        )

        logger.add(
            str(log_file),
            format=file_format,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )


def save_json(data: dict, path: pathlib.Path):
    """Save dictionary to JSON file using custom encoder."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def load_json(path: pathlib.Path) -> dict:
    """Load dictionary from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: pathlib.Path):
    """Serialize any Python object to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: pathlib.Path) -> Any:
    """Load Python object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
