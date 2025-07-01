# src/rl_optimizer/utils/setup.py

import logging
import sys
from typing import Any, Optional
import json
import pathlib
import pickle
import numpy as np

class NpEncoder(json.JSONEncoder):
    """自定义JSON编码器，以处理Numpy数据类型。"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def setup_logger(name: str, log_file: Optional[pathlib.Path] = None, level: int = logging.INFO) -> logging.Logger:
    """配置并返回一个标准化的日志记录器。

    Args:
        name (str): 日志记录器的名称。
        log_file (Optional[pathlib.Path]): 可选的日志文件路径。
        level (int): 日志级别，默认为INFO。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定了文件路径）
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
    
    return logger

def save_json(data: dict, path: pathlib.Path):
    """使用自定义编码器将字典保存为JSON文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NpEncoder)

def load_json(path: pathlib.Path) -> dict:
    """从JSON文件加载字典。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_pickle(data: Any, path: pathlib.Path):
    """将任何Python对象序列化为pickle文件。"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: pathlib.Path) -> Any:
    """从pickle文件加载Python对象。"""
    with open(path, 'rb') as f:
        return pickle.load(f)