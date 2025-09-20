# src/utils/logger.py

import sys
from typing import Any, Optional
import json
import pathlib
import pickle
import numpy as np
import multiprocessing
from loguru import logger


class NpEncoder(json.JSONEncoder):
    """自定义JSON编码器，以处理Numpy数据类型和路径对象。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath)):
            return str(obj)
        return super(NpEncoder, self).default(obj)


def setup_logger(
    name: str, log_file: Optional[pathlib.Path] = None, level: int = 20
) -> Any:
    """配置并返回loguru日志记录器。

    Args:
        name (str): 日志记录器的名称（用于上下文标识）。
        log_file (Optional[pathlib.Path]): 可选的日志文件路径。
        level (int): 日志级别，默认为20（INFO）。

    Returns:
        logger: 配置好的loguru日志记录器实例。
    """
    # 转换标准logging级别到loguru级别
    level_map = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
    log_level = level_map.get(level, "INFO")

    # 检查是否为多进程环境中的子进程
    is_main_process = True
    try:
        current_process = multiprocessing.current_process()
        if current_process.name != "MainProcess":
            is_main_process = False
    except Exception as e:
        logger.error(f"无法确定进程类型，假设为主进程: {e}")


    # 移除现有的handlers以避免重复配置
    logger.remove()

    # 配置控制台输出（彩色日志）
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "<level>{message}</level>"
    )

    # 为主进程和子进程设置不同的日志级别
    console_level = log_level if is_main_process else "WARNING"

    logger.add(
        sys.stdout,
        format=console_format,
        level=console_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 如果提供了日志文件路径且在主进程中，添加文件处理器
    if log_file and is_main_process:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]} | {message}"
        )

        logger.add(
            str(log_file),
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
        )

    # 绑定模块名称到logger上下文
    return logger.bind(module=name)


def save_json(data: dict, path: pathlib.Path):
    """使用自定义编码器将字典保存为JSON文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def load_json(path: pathlib.Path) -> dict:
    """从JSON文件加载字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: pathlib.Path):
    """将任何Python对象序列化为pickle文件。"""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: pathlib.Path) -> Any:
    """从pickle文件加载Python对象。"""
    with open(path, "rb") as f:
        return pickle.load(f)
