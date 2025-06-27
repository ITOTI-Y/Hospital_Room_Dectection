"""
GPU-accelerated demonstration script for the RL-based layout optimization system.

This script shows how to use the GPU-accelerated reinforcement learning optimizer
to improve hospital layout assignments based on travel time data.
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
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main demonstration function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== GPU RL Layout Optimization Demo ===")
    
    gpu_info = check_gpu_availability()
    logger.info("GPU Information:")
    for key, value in gpu_info.items():
        logger.info(f"  {key}: {value}")
    
    if not gpu_info['cuda_available']:
        logger.warning("CUDA not available. The system will fall back to CPU.")
        logger.info("For GPU acceleration, ensure you have:")
        logger.info("  1. NVIDIA GPU with CUDA support")
        logger.info("  2. CUDA drivers installed")
        logger.info("  3. PyTorch with CUDA support")
    
    project_root = pathlib.Path(__file__).parent.parent
    csv_path = project_root / "result" / "super_network_travel_times.csv"
    model_path = project_root / "result" / "rl_layout_model_gpu.pth"
    layout_path = project_root / "result" / "optimized_layout_gpu.json"
    
    if not csv_path.exists():
        logger.error(f"Travel times CSV not found at {csv_path}")
        logger.info("Please run the main network generation first to create the travel times data.")
        return
    
    workflow_patterns = create_default_workflow_patterns()
    
    workflow_patterns.extend([
        ['门', '挂号收费', '妇科', '采血处', '检验中心', '门'],  # Complex gynecology visit with tests
        ['门', '挂号收费', '超声科', '妇科', '门'],  # Ultrasound + gynecology
        ['门', '挂号收费', '内科', '放射科', '内诊药房', '门'],  # Internal medicine with X-ray
        ['门', '挂号收费', '儿科', '采血处', '门'],  # Pediatrics with blood test
    ])
    
    logger.info(f"Using {len(workflow_patterns)} workflow patterns for optimization")
    
    try:
        optimizer = GPULayoutOptimizer(str(csv_path), workflow_patterns)
        logger.info("GPU RL Layout Optimizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GPU optimizer: {e}")
        return
    
    logger.info("\n--- Evaluating Current Layout ---")
    current_eval = optimizer.evaluate_current_layout()
    logger.info(f"Current layout reward: {current_eval['current_reward']:.2f}")
    logger.info(f"Device used: {current_eval['device_used']}")
    
    logger.info("Workflow penalties in current layout:")
    for workflow_id, workflow_info in current_eval['workflow_penalties'].items():
        pattern = " -> ".join(workflow_info['pattern'])
        penalty = workflow_info['penalty']
        logger.info(f"  {pattern}: {penalty:.2f}")
    
    logger.info("\n--- Training DQN Agent ---")
    if gpu_info['cuda_available']:
        logger.info("Training with GPU acceleration...")
        num_episodes = 500
        max_steps = 50
    else:
        logger.info("Training with CPU (reduced episodes for demo)...")
        num_episodes = 100
        max_steps = 30
    
    try:
        training_stats = optimizer.train(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        logger.info(f"Training completed. Final epsilon: {training_stats['final_epsilon']:.3f}")
        
        episode_rewards = training_stats['episode_rewards']
        if len(episode_rewards) >= 50:
            initial_avg = sum(episode_rewards[:50]) / 50
            final_avg = sum(episode_rewards[-50:]) / 50
            logger.info(f"Average reward improvement: {initial_avg:.2f} -> {final_avg:.2f}")
        
        losses = training_stats.get('losses', [])
        if losses:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            logger.info(f"Final average loss: {avg_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    logger.info("\n--- Optimizing Layout ---")
    try:
        max_iterations = 200 if gpu_info['cuda_available'] else 50
        best_state, best_reward = optimizer.optimize_layout(max_iterations=max_iterations)
        logger.info(f"Optimization completed")
        logger.info(f"Best reward found: {best_reward:.2f}")
        logger.info(f"Improvement over current: {best_reward - current_eval['current_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return
    
    logger.info("\n--- Optimized Layout Assignments ---")
    for function, spaces in best_state.function_to_spaces.items():
        if spaces:  # Only show functions that have assignments
            logger.info(f"{function}: {', '.join(spaces)}")
    
    logger.info("\n--- Saving Results ---")
    try:
        optimizer.save_model(str(model_path))
        optimizer.export_optimized_layout(best_state, str(layout_path))
        logger.info(f"GPU model saved to: {model_path}")
        logger.info(f"GPU optimized layout saved to: {layout_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info("\n--- Performance Summary ---")
    if gpu_info['cuda_available']:
        memory_info = gpu_info['memory_info']
        logger.info(f"GPU Memory Usage:")
        logger.info(f"  Allocated: {memory_info['allocated_gb']:.2f} GB")
        logger.info(f"  Reserved: {memory_info['reserved_gb']:.2f} GB")
        logger.info(f"  Total: {memory_info['total_gb']:.2f} GB")
    
    logger.info("\n=== Optimization Summary ===")
    logger.info(f"Original reward: {current_eval['current_reward']:.2f}")
    logger.info(f"Optimized reward: {best_reward:.2f}")
    improvement = best_reward - current_eval['current_reward']
    logger.info(f"Total improvement: {improvement:.2f}")
    
    if improvement > 0:
        logger.info("✅ GPU layout optimization successful! The new layout reduces travel times.")
    else:
        logger.info("ℹ️  Current layout is already quite optimal for the given workflows.")
    
    logger.info(f"🚀 Optimization completed using: {current_eval['device_used']}")
    logger.info("\nGPU demo completed successfully!")

if __name__ == "__main__":
    main()
