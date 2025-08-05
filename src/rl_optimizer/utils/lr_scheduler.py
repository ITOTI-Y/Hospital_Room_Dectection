# src/rl_optimizer/utils/lr_scheduler.py

from typing import Callable


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    创建线性学习率调度器。
    
    该调度器会从初始学习率线性衰减到最终学习率。在训练开始时使用初始值，
    在训练结束时使用最终值，中间过程线性插值。
    
    Args:
        initial_value (float): 初始学习率值
        final_value (float): 最终学习率值，默认为0.0
    
    Returns:
        Callable[[float], float]: 学习率调度函数，接收progress_remaining参数
                                  (从1.0衰减到0.0表示训练进度)
    
    Example:
        >>> # 创建从3e-4线性衰减到1e-5的调度器
        >>> lr_scheduler = linear_schedule(3e-4, 1e-5)
        >>> # 训练开始时 (progress_remaining=1.0)
        >>> lr_scheduler(1.0)  # 返回 3e-4
        >>> # 训练中期 (progress_remaining=0.5)
        >>> lr_scheduler(0.5)  # 返回约 1.5e-4
        >>> # 训练结束时 (progress_remaining=0.0)
        >>> lr_scheduler(0.0)  # 返回 1e-5
    """
    def schedule_func(progress_remaining: float) -> float:
        """
        根据训练进度计算当前学习率。
        
        Args:
            progress_remaining (float): 剩余训练进度，从1.0（开始）递减到0.0（结束）
        
        Returns:
            float: 当前应使用的学习率值
        """
        # 线性插值公式：current_lr = final_value + progress_remaining * (initial_value - final_value)
        return final_value + progress_remaining * (initial_value - final_value)
    
    return schedule_func


def constant_schedule(value: float) -> Callable[[float], float]:
    """
    创建常数学习率调度器。
    
    该调度器在整个训练过程中保持固定的学习率不变。
    
    Args:
        value (float): 固定的学习率值
    
    Returns:
        Callable[[float], float]: 学习率调度函数，始终返回固定值
    
    Example:
        >>> # 创建固定3e-4学习率的调度器
        >>> lr_scheduler = constant_schedule(3e-4)
        >>> lr_scheduler(1.0)  # 返回 3e-4
        >>> lr_scheduler(0.5)  # 返回 3e-4
        >>> lr_scheduler(0.0)  # 返回 3e-4
    """
    def schedule_func(progress_remaining: float) -> float:
        """
        返回固定的学习率值，不受训练进度影响。
        
        Args:
            progress_remaining (float): 剩余训练进度（此参数被忽略）
        
        Returns:
            float: 固定的学习率值
        """
        return value
    
    return schedule_func


def get_lr_scheduler(schedule_type: str, initial_lr: float, final_lr: float = 0.0) -> Callable[[float], float]:
    """
    根据配置创建相应的学习率调度器。
    
    Args:
        schedule_type (str): 调度器类型，支持 "linear" 和 "constant"
        initial_lr (float): 初始学习率
        final_lr (float): 最终学习率，仅在线性调度器中使用
    
    Returns:
        Callable[[float], float]: 相应的学习率调度函数
    
    Raises:
        ValueError: 当调度器类型不支持时抛出异常
    
    Example:
        >>> # 创建线性衰减调度器
        >>> scheduler = get_lr_scheduler("linear", 3e-4, 1e-5)
        >>> # 创建常数调度器
        >>> scheduler = get_lr_scheduler("constant", 3e-4)
    """
    if schedule_type == "linear":
        return linear_schedule(initial_lr, final_lr)
    elif schedule_type == "constant":
        return constant_schedule(initial_lr)
    else:
        raise ValueError(
            f"不支持的学习率调度器类型: '{schedule_type}'. "
            f"支持的类型: 'linear', 'constant'"
        )