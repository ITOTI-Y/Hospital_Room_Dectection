#!/usr/bin/env python3
"""
测试势函数奖励日志显示效果
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.algorithms.constraint_manager import ConstraintManager
import numpy as np

def test_potential_logging():
    """测试势函数奖励的日志显示"""
    
    # 设置日志级别为INFO，确保能看到INFO级别的日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("测试势函数奖励日志显示效果")
    print("=" * 80)
    
    # 初始化配置
    config = RLConfig()
    config.ENABLE_POTENTIAL_REWARD = True  # 确保启用势函数奖励
    
    # 初始化缓存管理器（初始化时会自动加载所有数据）
    cache_manager = CacheManager(config)
    
    # 初始化成本计算器
    cost_calculator = CostCalculator(
        config=config,
        resolved_pathways=cache_manager.resolved_pathways,
        travel_times=cache_manager.travel_times_matrix,
        placeable_slots=cache_manager.placeable_slots,
        placeable_departments=cache_manager.placeable_departments
    )
    
    # 初始化约束管理器
    constraint_manager = ConstraintManager(config, cache_manager)
    
    # 创建环境
    env = LayoutEnv(config, cache_manager, cost_calculator, constraint_manager)
    
    print("\n开始测试一个episode...")
    print("-" * 40)
    
    # 重置环境
    obs, info = env.reset()
    done = False
    step_count = 0
    max_steps = 100  # 增加到100步，确保能完成整个episode
    
    print(f"\n环境已重置，共有 {env.num_slots} 个槽位需要填充")
    print("注意：现在势函数奖励日志应该只在episode结束时显示\n")
    
    while not done and step_count < max_steps:
        # 获取动作掩码
        action_mask = info.get("action_mask", None)
        
        if action_mask is not None:
            # 选择一个合法的动作
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                print("警告：没有合法动作可选")
                break
        else:
            # 随机选择动作
            action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 只显示前5步和最后几步的详细信息
        if step_count <= 5 or done:
            print(f"步骤 {step_count}: 动作={action}, 奖励={reward:.4f}, 完成={done}")
        elif step_count == 6:
            print("... （省略中间步骤） ...")
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print("-" * 40)
    
    if done:
        print("✓ Episode正常结束")
        print("✓ 势函数奖励汇总应该在上面的日志中显示")
    else:
        print(f"✓ 运行了 {step_count} 步（限制为 {max_steps} 步）")
        print("✓ 在非结束步骤中，势函数奖励不应显示在INFO级别日志中")
    
    print("\n说明：")
    print("1. 每步执行时，势函数奖励计算被记录为DEBUG级别（不显示）")
    print("2. 只有在episode结束时，才会显示INFO级别的势函数奖励汇总")
    print("=" * 80)

if __name__ == "__main__":
    test_potential_logging()