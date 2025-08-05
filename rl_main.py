# rl_main.py

from src.rl_optimizer.utils.setup import setup_logger
from src.rl_optimizer.agent.ppo_agent import PPOAgent
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.data.cache_manager import CacheManager
from src.config import RLConfig
import argparse
import sys
from pathlib import Path

# 将src目录添加到Python路径中，以便能够导入我们自己的模块
# 这是一个常见的做法，以避免复杂的相对导入问题
sys.path.append(str(Path(__file__).parent / 'src'))


logger = setup_logger(__name__)


def run_optimization(mode: str, model_path: str = None, resume_training: bool = False, 
                    checkpoint_freq: int = None):
    """
    执行强化学习布局优化的主函数。

    该函数会初始化所有必要的组件，并根据指定的模式（训练或评估）
    来启动相应的流程。

    Args:
        mode (str): 操作模式，必须是 'train' 或 'evaluate'。
        model_path (str, optional): 在 'evaluate' 模式下，需要提供的已训练模型的路径。
                                    在启用断点续训时，可指定特定的checkpoint路径。
                                    Defaults to None.
        resume_training (bool): 是否启用断点续训功能。Defaults to False.
        checkpoint_freq (int, optional): checkpoint保存频率（训练步数）。Defaults to None.
    """
    logger.info(f"===== 医院布局强化学习优化器 =====")
    logger.info(f"当前操作模式: {mode.upper()}")

    # --- 1. 初始化配置 ---
    try:
        config = RLConfig()
        
        # 根据命令行参数更新配置
        if resume_training:
            config.RESUME_TRAINING = True
            if model_path:
                config.PRETRAINED_MODEL_PATH = model_path
            logger.info("启用断点续训功能")
            
        if checkpoint_freq is not None:
            config.CHECKPOINT_FREQUENCY = checkpoint_freq
            logger.info(f"设置checkpoint频率: 每{checkpoint_freq}步")
            
        logger.info("配置模块初始化成功。")
    except Exception as e:
        logger.error(f"初始化配置时发生严重错误: {e}", exc_info=True)
        return

    # --- 2. 数据预处理与缓存加载 ---
    try:
        logger.info("正在初始化数据缓存管理器...")
        cache_manager = CacheManager(config)
        logger.info("数据缓存管理器初始化成功，所有数据已加载或生成。")
    except Exception as e:
        logger.error(f"数据预处理阶段发生严重错误: {e}", exc_info=True)
        return

    # --- 3. 初始化成本计算器 ---
    try:
        logger.info("正在初始化成本计算器...")
        # 从cache_manager获取预处理好的数据
        cost_calculator = CostCalculator(
            config=config,
            resolved_pathways=cache_manager.resolved_pathways,
            travel_times=cache_manager.travel_times_matrix,
            placeable_slots=cache_manager.placeable_slots,
            placeable_departments=cache_manager.placeable_departments
        )
        logger.info("成本计算器初始化成功。")
    except Exception as e:
        logger.error(f"初始化成本计算器时发生严重错误: {e}", exc_info=True)
        return

    # --- 4. 初始化PPO智能体 ---
    try:
        agent = PPOAgent(
            config=config,
            cache_manager=cache_manager,
            cost_calculator=cost_calculator
        )
        logger.info("PPO智能体初始化成功。")
    except Exception as e:
        logger.error(f"初始化PPO智能体时发生严重错误: {e}", exc_info=True)
        return

    # --- 5. 根据模式执行操作 ---
    if mode == 'train':
        try:
            agent.train()
        except Exception as e:
            logger.error(f"训练过程中发生未捕获的异常: {e}", exc_info=True)
    elif mode == 'evaluate':
        if not model_path:
            logger.error("评估模式需要通过 '--model-path' 参数指定模型文件路径。")
            return

        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"指定的模型文件未找到: {model_file}")
            return

        try:
            agent.evaluate(model_path)
        except Exception as e:
            logger.error(f"评估过程中发生未捕获的异常: {e}", exc_info=True)
    else:
        logger.error(f"无效的操作模式: '{mode}'. 请选择 'train' 或 'evaluate'。")

    logger.info("===== 流程执行完毕 =====")


if __name__ == "__main__":
    # --- 配置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="基于强化学习的医院布局优化器。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate'],
        default='train',
        help="选择运行模式:\n"
             "  train     - 启动一个全新的训练流程。\n"
             "  evaluate  - 加载一个已有的模型进行评估并输出最优布局。"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="在 'evaluate' 模式下，指定已训练好的模型文件路径 (.zip)。\n"
             "在 'train' 模式下结合 --resume 使用时，指定用于断点续训的模型路径。"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="启用断点续训功能。自动查找最新的checkpoint继续训练，\n"
             "或结合 --model-path 指定特定的checkpoint。"
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=None,
        help="设置checkpoint保存频率（训练步数）。默认使用配置文件中的设置。"
    )

    args = parser.parse_args()

    run_optimization(args.mode, args.model_path, args.resume, args.checkpoint_freq)
