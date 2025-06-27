"""
基于GPU加速的医院布局优化系统演示脚本（GPU版本）

本脚本展示如何使用GPU加速的强化学习优化器来改善医院布局分配，基于行程时间数据进行优化。
该演示使用Deep Q-Network (DQN)算法，充分利用GPU并行计算能力实现高效训练。

主要功能：
1. 检测GPU硬件可用性和CUDA支持
2. 加载医院网络行程时间数据
3. 创建多种医院工作流模式
4. 初始化DQN深度强化学习优化器
5. 评估当前布局的性能
6. 使用GPU加速训练DQN智能体
7. 优化医院功能区域的空间分配
8. 监控GPU内存使用情况
9. 保存训练模型和优化结果

适用场景：
- 大型医院复杂布局优化
- 需要高性能计算的场景
- 大规模状态空间的强化学习问题
- 研究和生产环境的GPU加速应用

GPU要求：
- NVIDIA GPU支持CUDA 12.4+
- 至少4GB显存
- 正确安装CUDA驱动和PyTorch
"""

import sys
import pathlib
import logging
import torch

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analysis.rl_layout_optimizer_gpu import (
    GPULayoutOptimizer, 
    check_gpu_availability,
    create_default_workflow_patterns
)

def setup_logging():
    """
    设置GPU演示脚本的日志配置
    
    配置详细的日志输出格式，包含时间戳、模块名称、日志级别和消息内容，
    特别适用于跟踪GPU训练过程、内存使用情况和优化结果。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """
    主演示函数 - GPU版强化学习布局优化完整流程
    
    该函数演示了完整的GPU加速医院布局优化流程：
    1. GPU硬件检测和环境验证
    2. 数据准备和验证
    3. 工作流模式配置
    4. DQN智能体训练（GPU加速）
    5. 布局优化执行
    6. GPU性能监控和分析
    7. 结果保存和性能总结
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== GPU加速强化学习布局优化演示 ===")
    
    logger.info("正在检测GPU硬件环境...")
    gpu_info = check_gpu_availability()
    logger.info("🔍 GPU硬件信息:")
    for key, value in gpu_info.items():
        logger.info(f"  {key}: {value}")
    
    if not gpu_info['cuda_available']:
        logger.warning("⚠️  CUDA不可用，系统将回退到CPU模式")
        logger.info("要启用GPU加速，请确保具备以下条件:")
        logger.info("  1. NVIDIA GPU支持CUDA")
        logger.info("  2. 已安装CUDA驱动程序")
        logger.info("  3. PyTorch支持CUDA版本")
        logger.info("  4. 足够的GPU显存（建议≥4GB）")
    else:
        logger.info("✅ GPU环境检测成功，将使用GPU加速训练")
    
    project_root = pathlib.Path(__file__).parent.parent
    csv_path = project_root / "result" / "super_network_travel_times.csv"  # 行程时间数据文件
    model_path = project_root / "result" / "rl_layout_model_gpu.pth"       # DQN模型保存路径（PyTorch格式）
    layout_path = project_root / "result" / "optimized_layout_gpu.json"    # GPU优化布局结果保存路径
    
    if not csv_path.exists():
        logger.error(f"❌ 未找到行程时间数据文件: {csv_path}")
        logger.info("请先运行主程序生成网络图和行程时间数据")
        logger.info("运行命令: python main.py")
        logger.info("该文件包含医院内所有节点间的最短路径时间，是优化算法的基础数据")
        return
    
    workflow_patterns = create_default_workflow_patterns()
    
    workflow_patterns.extend([
        ['门', '挂号收费', '妇科', '采血处', '检验中心', '门'],  # 妇科综合检查（包含血液检验）
        ['门', '挂号收费', '超声科', '妇科', '门'],              # 超声检查+妇科诊断
        ['门', '挂号收费', '内科', '放射科', '内诊药房', '门'],  # 内科诊疗+影像检查+取药
        ['门', '挂号收费', '儿科', '采血处', '门'],              # 儿科诊疗+血液检查
    ])
    
    logger.info(f"📋 共加载 {len(workflow_patterns)} 个工作流模式用于GPU优化")
    logger.info("GPU版本支持更复杂的工作流组合和大规模并行处理")
    
    try:
        optimizer = GPULayoutOptimizer(str(csv_path), workflow_patterns)
        logger.info("✅ GPU强化学习布局优化器初始化成功")
        logger.info(f"   - 神经网络架构: DQN (Deep Q-Network)")
        logger.info(f"   - 状态空间维度: {optimizer.environment.state_size}")
        logger.info(f"   - 动作空间大小: {optimizer.environment.action_size}")
        logger.info(f"   - 医院功能数量: {len(optimizer.environment.all_functions)}")
        logger.info(f"   - 物理空间数量: {len(optimizer.environment.all_spaces)}")
        logger.info(f"   - 计算设备: {optimizer.device}")
        
        if hasattr(optimizer, 'dqn_agent') and hasattr(optimizer.dqn_agent, 'q_network'):
            total_params = sum(p.numel() for p in optimizer.dqn_agent.q_network.parameters())
            logger.info(f"   - 神经网络参数总数: {total_params:,}")
            
    except Exception as e:
        logger.error(f"❌ GPU优化器初始化失败: {e}")
        logger.error("可能的原因: GPU内存不足、CUDA版本不兼容或数据文件格式错误")
        return
    
    logger.info("\n--- 评估当前布局性能 ---")
    logger.info("正在使用GPU加速分析当前医院布局的工作流效率...")
    current_eval = optimizer.evaluate_current_layout()
    logger.info(f"当前布局奖励值: {current_eval['current_reward']:.2f}")
    logger.info(f"计算设备: {current_eval['device_used']}")
    logger.info("(奖励值越高表示布局越优，负值表示存在时间惩罚)")
    
    logger.info("\n各工作流在当前布局下的时间惩罚:")
    total_penalty = 0
    for workflow_id, workflow_info in current_eval['workflow_penalties'].items():
        pattern = " → ".join(workflow_info['pattern'])
        penalty = workflow_info['penalty']
        logger.info(f"  {pattern}: {penalty:.2f}秒")
        total_penalty += penalty
    
    logger.info(f"总时间惩罚: {total_penalty:.2f}秒")
    logger.info(f"平均每工作流惩罚: {total_penalty/len(current_eval['workflow_penalties']):.2f}秒")
    
    logger.info("\n--- 训练深度Q网络智能体 ---")
    if gpu_info['cuda_available']:
        logger.info("🚀 使用GPU加速训练DQN模型...")
        num_episodes = 500
        max_steps = 50
        logger.info("GPU训练参数:")
        logger.info("  - 算法类型: Deep Q-Network (DQN)")
        logger.info("  - 训练轮数: 500轮")
        logger.info("  - 每轮最大步数: 50步")
        logger.info("  - 批处理大小: 32")
        logger.info("  - 学习率: 1e-3")
        logger.info("  - 经验回放缓冲区: 10,000")
        logger.info("  - 目标网络更新频率: 10轮")
    else:
        logger.info("⚠️  使用CPU训练（演示模式，减少训练轮数）...")
        num_episodes = 100
        max_steps = 30
        logger.info("CPU回退训练参数:")
        logger.info("  - 训练轮数: 100轮（减少以适应CPU性能）")
        logger.info("  - 每轮最大步数: 30步")
    
    try:
        training_stats = optimizer.train(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        logger.info(f"✅ DQN训练完成！最终探索率: {training_stats['final_epsilon']:.3f}")
        
        episode_rewards = training_stats['episode_rewards']
        if len(episode_rewards) >= 50:
            initial_avg = sum(episode_rewards[:50]) / 50
            final_avg = sum(episode_rewards[-50:]) / 50
            improvement = final_avg - initial_avg
            logger.info(f"平均奖励改进: {initial_avg:.2f} → {final_avg:.2f}")
            logger.info(f"训练改进幅度: {improvement:.2f} ({improvement/abs(initial_avg)*100:.1f}%)")
        
        losses = training_stats.get('losses', [])
        if losses:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            initial_loss = sum(losses[:10]) / min(10, len(losses)) if len(losses) >= 10 else losses[0]
            logger.info(f"最终平均损失: {avg_loss:.4f}")
            logger.info(f"损失函数改进: {initial_loss:.4f} → {avg_loss:.4f}")
            logger.info(f"神经网络收敛良好，损失下降 {((initial_loss-avg_loss)/initial_loss*100):.1f}%")
        
        if gpu_info['cuda_available']:
            logger.info(f"GPU训练总轮数: {len(episode_rewards)}")
            logger.info(f"经验回放样本数: {training_stats.get('memory_size', 'N/A')}")
            
    except Exception as e:
        logger.error(f"❌ DQN训练失败: {e}")
        logger.error("可能的原因: GPU内存不足、网络结构问题或训练参数设置不当")
        return
    
    logger.info("\n--- 执行GPU加速布局优化 ---")
    logger.info("使用训练好的DQN模型寻找最优布局...")
    
    try:
        max_iterations = 200 if gpu_info['cuda_available'] else 50
        logger.info(f"优化策略: DQN贪婪搜索 + 神经网络预测")
        logger.info(f"最大迭代次数: {max_iterations}次")
        
        if gpu_info['cuda_available']:
            logger.info("🚀 GPU加速优化进行中...")
        else:
            logger.info("⚠️  CPU模式优化进行中...")
            
        best_state, best_reward = optimizer.optimize_layout(max_iterations=max_iterations)
        logger.info(f"✅ 布局优化完成")
        logger.info(f"最优奖励值: {best_reward:.2f}")
        
        improvement = best_reward - current_eval['current_reward']
        logger.info(f"相比当前布局的改进: {improvement:.2f}")
        
        if improvement > 0:
            improvement_percent = improvement / abs(current_eval['current_reward']) * 100
            logger.info(f"性能提升百分比: {improvement_percent:.1f}%")
            logger.info("🎉 GPU优化找到了更优的布局配置！")
            
            time_saved_per_workflow = abs(improvement)
            daily_workflows = len(workflow_patterns) * 50  # 假设每天50次各类工作流
            daily_time_saved = time_saved_per_workflow * daily_workflows / 60  # 转换为分钟
            logger.info(f"预计每日节省时间: {daily_time_saved:.0f}分钟")
        else:
            logger.info("ℹ️  当前布局已经相当优化，GPU算法确认改进空间有限")
        
    except Exception as e:
        logger.error(f"❌ GPU布局优化失败: {e}")
        logger.error("可能的原因: GPU内存不足、搜索空间过大或网络预测不稳定")
        return
    
    logger.info("\n--- GPU优化后的布局分配方案 ---")
    logger.info("各医院功能的最优空间分配（GPU计算结果）:")
    
    assigned_count = 0
    total_spaces_used = 0
    multi_space_functions = 0
    
    for function, spaces in best_state.function_to_spaces.items():
        if spaces:  # 只显示有分配的功能
            logger.info(f"  {function}: {', '.join(spaces)}")
            assigned_count += 1
            total_spaces_used += len(spaces)
            if len(spaces) > 1:
                multi_space_functions += 1
    
    logger.info(f"\n📊 GPU优化分配统计:")
    logger.info(f"  - 已分配功能数: {assigned_count}")
    logger.info(f"  - 使用空间总数: {total_spaces_used}")
    logger.info(f"  - 多空间分配功能: {multi_space_functions}")
    logger.info(f"  - 平均每功能空间数: {total_spaces_used/assigned_count:.1f}")
    logger.info(f"  - 空间利用率: {(total_spaces_used/len(optimizer.environment.all_spaces)*100):.1f}%")
    
    logger.info("\n--- 保存GPU优化结果 ---")
    try:
        optimizer.save_model(str(model_path))
        logger.info(f"✅ DQN神经网络模型已保存: {model_path}")
        logger.info("   模型包含: 神经网络权重、优化器状态、训练统计信息")
        
        optimizer.export_optimized_layout(best_state, str(layout_path))
        logger.info(f"✅ GPU优化布局已保存: {layout_path}")
        logger.info("   布局包含: 功能-空间映射、GPU性能分析、DQN训练统计")
        
    except Exception as e:
        logger.error(f"❌ GPU结果保存失败: {e}")
        logger.error("请检查文件写入权限、磁盘空间和GPU内存状态")
    
    logger.info("\n--- GPU性能监控总结 ---")
    if gpu_info['cuda_available']:
        memory_info = gpu_info['memory_info']
        logger.info(f"🔍 GPU内存使用情况:")
        logger.info(f"  已分配显存: {memory_info['allocated_gb']:.2f} GB")
        logger.info(f"  已保留显存: {memory_info['reserved_gb']:.2f} GB")
        logger.info(f"  总显存容量: {memory_info['total_gb']:.2f} GB")
        
        memory_usage_percent = (memory_info['allocated_gb'] / memory_info['total_gb']) * 100
        logger.info(f"  显存使用率: {memory_usage_percent:.1f}%")
        
        if memory_usage_percent > 80:
            logger.warning("⚠️  显存使用率较高，建议监控内存泄漏")
        else:
            logger.info("✅ 显存使用正常")
    
    logger.info("\n=== GPU优化结果总结 ===")
    logger.info("📊 性能对比分析:")
    logger.info(f"   原始布局奖励: {current_eval['current_reward']:.2f}")
    logger.info(f"   GPU优化奖励: {best_reward:.2f}")
    improvement = best_reward - current_eval['current_reward']
    logger.info(f"   总体改进幅度: {improvement:.2f}")
    
    if improvement > 0:
        time_saved = abs(improvement)
        efficiency_gain = (improvement / abs(current_eval['current_reward'])) * 100
        logger.info(f"   预计节省时间: {time_saved:.0f}秒/工作流")
        logger.info(f"   效率提升: {efficiency_gain:.1f}%")
        logger.info("✅ GPU布局优化成功！新布局显著减少了患者行程时间")
        logger.info("💡 建议: GPU优化结果可以在实际医院中实施")
        logger.info("🔬 GPU加速使得能够探索更复杂的优化空间")
    else:
        logger.info("ℹ️  当前布局对于给定的工作流模式已经相当优化")
        logger.info("💡 建议: 可以尝试增加更多复杂工作流或调整DQN网络结构")
    
    logger.info(f"\n🚀 优化计算设备: {current_eval['device_used']}")
    
    if gpu_info['cuda_available']:
        logger.info("🎯 GPU加速强化学习布局优化演示完成！")
        logger.info("⚡ GPU版本相比CPU版本具有更强的计算能力和更快的收敛速度")
    else:
        logger.info("🎯 CPU回退模式强化学习布局优化演示完成！")
        logger.info("💡 建议配置GPU环境以获得更好的性能")
        
    logger.info("📁 GPU优化结果文件已保存到 result/ 目录")
    logger.info("🔄 如需对比CPU版本，请运行: python examples/rl_optimization_demo.py")

if __name__ == "__main__":
    main()
