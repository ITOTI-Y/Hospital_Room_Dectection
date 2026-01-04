# src/utils/logger.py

import pathlib
import sys

from loguru import logger


def setup_logger(log_file: pathlib.Path | None = None, level: str = 'INFO') -> None:
    """Configure loguru logger.

    Args:
        log_file (pathlib.Path | None): Optional log file path.
        level (str): Log level, default is "INFO".
    """

    logger.remove()

    console_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{extra[module]!s}</cyan> | '
        '<level>{message}</level>'
    )

    logger.configure(extra={'module': 'unknown'})

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
            '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | '
            '{extra[module]} | {message}\n'
            '{exception}'
        )

        logger.add(
            str(log_file),
            format=file_format,
            level=level,
            rotation='10 MB',
            retention='30 days',
            compression='zip',
            encoding='utf-8',
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
