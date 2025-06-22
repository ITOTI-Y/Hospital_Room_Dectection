#!/usr/bin/env python3
"""
强化学习布局优化演示脚本

此脚本展示如何使用新的RLLayoutOptimizer替代传统优化器
"""

import logging
import sys
import pathlib
from typing import Dict, List

# 导入必要模块
from src.config import NetworkConfig, COLOR_MAP
from src.analysis.process_flow import PathFinder
from src.analysis.word_detect import WordDetect
from src.optimization.optimizer import (
    FunctionalAssignment,
    WorkflowDefinition
)
from src.optimization.rl_optimizer import RLLayoutOptimizer, RLConfig

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/rl_optimization.log', mode='a', encoding='utf-8')
        ]
    )

def run_rl_optimization_demo():
    """运行强化学习优化演示"""
    logger = logging.getLogger(__name__)
    logger.info("=== 开始强化学习布局优化演示 ===")
    
    # 1. 初始化配置
    app_config = NetworkConfig(color_map_data=COLOR_MAP)
    
    # 2. 检查必要文件是否存在
    travel_times_csv_path = app_config.RESULT_PATH / 'super_network_travel_times.csv'
    if not travel_times_csv_path.exists():
        logger.error(f"Travel times CSV file not found: {travel_times_csv_path}")
        logger.error("Please run the main.py first to generate the travel times data.")
        return
    
    # 3. 初始化PathFinder
    try:
        path_finder = PathFinder(config=app_config)
        if path_finder.travel_times_df is None:
            logger.error("Failed to load travel times data.")
            return
        logger.info(f"Loaded travel times data with {len(path_finder.all_name_ids)} locations")
    except Exception as e:
        logger.error(f"Error initializing PathFinder: {e}")
        return
    
    # 4. 定义工作流程
    word_detect = WordDetect(config=app_config)
    workflow_definitions = [
        WorkflowDefinition(
            workflow_id='WF_Gynecology_A',
            functional_sequence=word_detect.detect_nearest_word(
                ['入口', '妇科', '采血处', '超声科', '妇科', '门诊药房', '入口']),
            weight=1.0
        ),
        WorkflowDefinition(
            workflow_id='WF_Cardiology_A',
            functional_sequence=word_detect.detect_nearest_word(
                ['入口', '心血管内科', '采血处', '超声科', '放射科', '心血管内科', '门诊药房', '入口']),
            weight=1.2
        ),
        WorkflowDefinition(
            workflow_id='WF_Pulmonology_A',
            functional_sequence=word_detect.detect_nearest_word(
                ['入口', '呼吸内科', '采血处', '放射科', '呼吸内科', '门诊药房', '入口']),
            weight=0.8
        )
    ]
    logger.info(f"Defined {len(workflow_definitions)} workflows for optimization")
    
    # 5. 创建初始功能分配
    initial_assignment = FunctionalAssignment(path_finder.name_to_ids_map)
    logger.info("Initial assignment created from PathFinder's default mapping")
    
    # 6. 配置强化学习参数
    rl_config = RLConfig(
        learning_rate=1e-4,
        batch_size=32,
        replay_buffer_size=10000,  # 减小缓冲区大小以适应演示
        target_update_frequency=500,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        gamma=0.99,
        hidden_dim1=256,  # 减小隐藏层大小以适应演示
        hidden_dim2=128,
        hidden_dim3=64,
    )
    
    # 7. 初始化强化学习优化器
    try:
        rl_optimizer = RLLayoutOptimizer(
            path_finder=path_finder,
            workflow_definitions=workflow_definitions,
            config=app_config,
            rl_config=rl_config,
            area_tolerance_ratio=0.3
        )
        logger.info("RLLayoutOptimizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RLLayoutOptimizer: {e}")
        return
    
    # 8. 运行优化
    model_save_path = app_config.RESULT_PATH / "rl_layout_model.pth"
    try:
        logger.info("Starting reinforcement learning optimization...")
        best_assignment, best_objective, best_outcomes = rl_optimizer.run_optimization(
            initial_assignment=initial_assignment,
            max_iterations=200,  # 减少迭代次数以适应演示
            save_model_path=str(model_save_path)
        )
        
        logger.info(f"Optimization completed! Best objective: {best_objective:.2f}")
        
        # 9. 分析结果
        logger.info("=== 优化结果分析 ===")
        logger.info(f"最终目标值: {best_objective:.2f}")
        
        # 比较初始和最终分配
        initial_objective = float('inf')
        try:
            initial_objective, _ = rl_optimizer.objective_calculator.evaluate(initial_assignment)
            improvement = initial_objective - best_objective
            improvement_pct = (improvement / initial_objective) * 100 if initial_objective != float('inf') else 0
            
            logger.info(f"初始目标值: {initial_objective:.2f}")
            logger.info(f"改进幅度: {improvement:.2f} ({improvement_pct:.1f}%)")
        except Exception as e:
            logger.warning(f"Error calculating initial objective: {e}")
        
        # 显示工作流程结果
        logger.info("\n各工作流程的优化结果:")
        for outcome in best_outcomes:
            logger.info(f"  {outcome.workflow_definition.workflow_id}: "
                       f"平均时间 = {outcome.average_time:.2f}, "
                       f"权重 = {outcome.workflow_definition.weight}")
        
        # 保存结果
        results_summary = {
            'initial_objective': initial_objective,
            'final_objective': best_objective,
            'improvement': improvement if initial_objective != float('inf') else 0,
            'improvement_percentage': improvement_pct if initial_objective != float('inf') else 0,
            'workflow_results': [
                {
                    'workflow_id': outcome.workflow_definition.workflow_id,
                    'average_time': outcome.average_time,
                    'weight': outcome.workflow_definition.weight,
                    'weighted_time': outcome.average_time * outcome.workflow_definition.weight
                }
                for outcome in best_outcomes
            ]
        }
        
        import json
        results_file = app_config.RESULT_PATH / "rl_optimization_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        return
    
    logger.info("=== 强化学习布局优化演示完成 ===")

if __name__ == "__main__":
    # 确保日志目录存在
    pathlib.Path("logs").mkdir(exist_ok=True)
    
    setup_logging()
    run_rl_optimization_demo()