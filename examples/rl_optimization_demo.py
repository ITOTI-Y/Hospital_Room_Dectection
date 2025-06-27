"""
基于强化学习的医院布局优化系统演示脚本（CPU版本）

本脚本展示如何使用强化学习优化器来改善医院布局分配，基于行程时间数据进行优化。
该演示使用Q-Learning算法，适合在没有GPU的环境中运行。

主要功能：
1. 加载医院网络行程时间数据
2. 创建多种医院工作流模式
3. 初始化Q-Learning强化学习优化器
4. 评估当前布局的性能
5. 训练Q-Learning智能体
6. 优化医院功能区域的空间分配
7. 保存训练模型和优化结果

适用场景：
- 中小型医院布局优化
- CPU环境下的快速原型验证
- 教学和研究用途
- 不需要大规模并行计算的场景
"""

import sys
import pathlib
import logging

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.analysis.rl_layout_optimizer import (
    LayoutOptimizer, 
    create_default_workflow_patterns
)

def setup_logging():
    """
    设置演示脚本的日志配置
    
    配置日志输出格式，包含时间戳、模块名称、日志级别和消息内容，
    便于跟踪强化学习训练过程和优化结果。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """
    主演示函数 - CPU版强化学习布局优化完整流程
    
    该函数演示了完整的医院布局优化流程：
    1. 数据准备和验证
    2. 工作流模式配置
    3. Q-Learning智能体训练
    4. 布局优化执行
    5. 结果保存和分析
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== 强化学习布局优化演示 (CPU版本) ===")
    
    project_root = pathlib.Path(__file__).parent.parent
    csv_path = project_root / "result" / "super_network_travel_times.csv"  # 行程时间数据文件
    model_path = project_root / "result" / "rl_layout_model.json"          # Q-Learning模型保存路径
    layout_path = project_root / "result" / "optimized_layout.json"        # 优化布局结果保存路径
    
    if not csv_path.exists():
        logger.error(f"未找到行程时间数据文件: {csv_path}")
        logger.info("请先运行主程序生成网络图和行程时间数据")
        logger.info("运行命令: python main.py")
        return
    
    workflow_patterns = create_default_workflow_patterns()
    
    workflow_patterns.extend([
        ['门', '挂号收费', '妇科', '采血处', '检验中心', '门'],  # 妇科综合检查（包含血液检验）
        ['门', '挂号收费', '超声科', '妇科', '门'],              # 超声检查+妇科诊断
        ['门', '挂号收费', '内科', '放射科', '内诊药房', '门'],  # 内科诊疗+影像检查+取药
        ['门', '挂号收费', '儿科', '采血处', '门'],              # 儿科诊疗+血液检查
    ])
    
    logger.info(f"共加载 {len(workflow_patterns)} 个工作流模式用于优化")
    logger.info("工作流模式包括：基础就诊、检查检验、综合诊疗等多种场景")
    
    try:
        optimizer = LayoutOptimizer(str(csv_path), workflow_patterns)
        logger.info("✅ 强化学习布局优化器初始化成功")
        logger.info(f"   - 状态空间维度: {optimizer.environment.state_size}")
        logger.info(f"   - 动作空间大小: {optimizer.environment.action_size}")
        logger.info(f"   - 医院功能数量: {len(optimizer.environment.all_functions)}")
        logger.info(f"   - 物理空间数量: {len(optimizer.environment.all_spaces)}")
    except Exception as e:
        logger.error(f"❌ 优化器初始化失败: {e}")
        logger.error("请检查数据文件格式和内容是否正确")
        return
    
    logger.info("\n--- 评估当前布局性能 ---")
    logger.info("正在分析当前医院布局的工作流效率...")
    current_eval = optimizer.evaluate_current_layout()
    logger.info(f"当前布局奖励值: {current_eval['current_reward']:.2f}")
    logger.info("(奖励值越高表示布局越优，负值表示存在惩罚)")
    
    logger.info("\n各工作流在当前布局下的时间惩罚:")
    for workflow_id, workflow_info in current_eval['workflow_penalties'].items():
        pattern = " → ".join(workflow_info['pattern'])
        penalty = workflow_info['penalty']
        logger.info(f"  {pattern}: {penalty:.2f}秒")
    
    total_penalty = sum(info['penalty'] for info in current_eval['workflow_penalties'].values())
    logger.info(f"总时间惩罚: {total_penalty:.2f}秒")
    
    logger.info("\n--- 训练强化学习智能体 ---")
    logger.info("开始Q-Learning算法训练，这可能需要几分钟时间...")
    logger.info("训练参数:")
    logger.info("  - 算法类型: Q-Learning (表格型强化学习)")
    logger.info("  - 训练轮数: 500轮")
    logger.info("  - 每轮最大步数: 50步")
    logger.info("  - 学习率: 0.1")
    logger.info("  - 折扣因子: 0.95")
    logger.info("  - 探索策略: ε-贪婪 (ε=0.1)")
    
    try:
        training_stats = optimizer.train(num_episodes=500, max_steps_per_episode=50)
        logger.info(f"✅ 训练完成！最终探索率: {training_stats['final_epsilon']:.3f}")
        
        episode_rewards = training_stats['episode_rewards']
        if len(episode_rewards) >= 100:
            initial_avg = sum(episode_rewards[:100]) / 100
            final_avg = sum(episode_rewards[-100:]) / 100
            improvement = final_avg - initial_avg
            logger.info(f"平均奖励改进: {initial_avg:.2f} → {final_avg:.2f}")
            logger.info(f"训练改进幅度: {improvement:.2f} ({improvement/abs(initial_avg)*100:.1f}%)")
        
        logger.info(f"Q表大小: {len(training_stats.get('q_table_size', 0))} 个状态-动作对")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        logger.error("可能的原因: 内存不足、数据格式错误或算法参数设置问题")
        return
    
    logger.info("\n--- 执行布局优化 ---")
    logger.info("使用训练好的Q-Learning模型寻找最优布局...")
    logger.info("优化策略: 贪婪搜索 + 训练经验")
    logger.info("最大迭代次数: 200次")
    
    try:
        best_state, best_reward = optimizer.optimize_layout(max_iterations=200)
        logger.info(f"✅ 布局优化完成")
        logger.info(f"最优奖励值: {best_reward:.2f}")
        
        improvement = best_reward - current_eval['current_reward']
        logger.info(f"相比当前布局的改进: {improvement:.2f}")
        
        if improvement > 0:
            improvement_percent = improvement / abs(current_eval['current_reward']) * 100
            logger.info(f"性能提升百分比: {improvement_percent:.1f}%")
            logger.info("🎉 找到了更优的布局配置！")
        else:
            logger.info("ℹ️  当前布局已经相当优化，改进空间有限")
        
    except Exception as e:
        logger.error(f"❌ 布局优化失败: {e}")
        logger.error("可能的原因: 搜索空间过大、收敛困难或算法参数需要调整")
        return
    
    logger.info("\n--- 优化后的布局分配方案 ---")
    logger.info("各医院功能的最优空间分配:")
    
    assigned_count = 0
    total_spaces_used = 0
    for function, spaces in best_state.function_to_spaces.items():
        if spaces:  # 只显示有分配的功能
            logger.info(f"  {function}: {', '.join(spaces)}")
            assigned_count += 1
            total_spaces_used += len(spaces)
    
    logger.info(f"\n分配统计:")
    logger.info(f"  - 已分配功能数: {assigned_count}")
    logger.info(f"  - 使用空间总数: {total_spaces_used}")
    logger.info(f"  - 平均每功能空间数: {total_spaces_used/assigned_count:.1f}")
    
    logger.info("\n--- 保存优化结果 ---")
    try:
        optimizer.save_model(str(model_path))
        logger.info(f"✅ Q-Learning模型已保存: {model_path}")
        logger.info("   模型包含: Q表、超参数、训练统计信息")
        
        optimizer.export_optimized_layout(best_state, str(layout_path))
        logger.info(f"✅ 优化布局已保存: {layout_path}")
        logger.info("   布局包含: 功能-空间映射、性能分析、改进统计")
        
    except Exception as e:
        logger.error(f"❌ 结果保存失败: {e}")
        logger.error("请检查文件写入权限和磁盘空间")
    
    logger.info("\n=== 优化结果总结 ===")
    logger.info("📊 性能对比:")
    logger.info(f"   原始布局奖励: {current_eval['current_reward']:.2f}")
    logger.info(f"   优化布局奖励: {best_reward:.2f}")
    improvement = best_reward - current_eval['current_reward']
    logger.info(f"   总体改进幅度: {improvement:.2f}")
    
    if improvement > 0:
        time_saved = abs(improvement)  # 改进的奖励值对应节省的时间
        logger.info(f"   预计节省时间: {time_saved:.0f}秒/工作流")
        logger.info("✅ 布局优化成功！新布局显著减少了患者行程时间")
        logger.info("💡 建议: 可以考虑在实际医院中实施这个优化方案")
    else:
        logger.info("ℹ️  当前布局对于给定的工作流模式已经相当优化")
        logger.info("💡 建议: 可以尝试添加更多工作流模式或调整优化参数")
    
    logger.info("\n🎯 CPU版强化学习布局优化演示完成！")
    logger.info("📁 结果文件已保存到 result/ 目录")
    logger.info("🔄 如需GPU加速版本，请运行: python examples/gpu_optimization_demo.py")

if __name__ == "__main__":
    main()
