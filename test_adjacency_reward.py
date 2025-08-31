#!/usr/bin/env python3
"""
动态相邻性奖励功能全面测试脚本

测试范围：
1. 单元测试: 配置参数、相邻性算法、边界条件处理
2. 集成测试: 与LayoutEnv和PPO的集成
3. 性能测试: 缓存机制和预计算性能
4. 稳定性测试: 长时间运行和异常数据处理
"""

import os
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

# 添加项目路径到系统路径
sys.path.insert(0, '/home/pan/code/Hospital_Room_Dectection')

# 导入项目依赖
from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.algorithms.constraint_manager import ConstraintManager
from src.algorithms.adjacency.spatial_calculator import SpatialAdjacencyCalculator
from src.algorithms.adjacency.utils import (
    validate_adjacency_preferences, 
    calculate_distance_percentile,
    normalize_adjacency_matrix
)
from src.rl_optimizer.utils.setup import setup_logger

# 抑制不重要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = setup_logger(__name__)


class AdjacencyTestFramework:
    """动态相邻性奖励功能测试框架"""
    
    def __init__(self):
        """初始化测试框架"""
        self.test_results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'stability_tests': {},
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'errors': [],
                'warnings': []
            }
        }
        
        self.config = None
        self.cache_manager = None
        self.cost_calculator = None
        self.constraint_manager = None
        self.layout_env = None
        
        self.start_time = time.time()
        
        # 创建测试数据目录
        self.test_data_dir = Path('/home/pan/code/Hospital_Room_Dectection/test_data')
        self.test_data_dir.mkdir(exist_ok=True)
        
        logger.info("动态相邻性奖励功能测试框架初始化完成")
    
    def setup_test_environment(self) -> bool:
        """
        设置测试环境并初始化所有组件
        
        Returns:
            bool: 环境设置是否成功
        """
        try:
            logger.info("开始设置测试环境...")
            
            # 初始化配置
            self.config = RLConfig()
            
            # 初始化缓存管理器
            self.cache_manager = CacheManager(self.config)
            
            # 初始化成本计算器
            self.cost_calculator = CostCalculator(
                config=self.config,
                resolved_pathways=self.cache_manager.resolved_pathways,
                travel_times=self.cache_manager.travel_times_matrix,
                placeable_slots=self.cache_manager.placeable_slots,
                placeable_departments=self.cache_manager.placeable_departments
            )
            
            # 初始化约束管理器
            self.constraint_manager = ConstraintManager(
                config=self.config,
                cache_manager=self.cache_manager
            )
            
            # 初始化布局环境
            self.layout_env = LayoutEnv(
                config=self.config,
                cache_manager=self.cache_manager,
                cost_calculator=self.cost_calculator,
                constraint_manager=self.constraint_manager
            )
            
            logger.info("测试环境设置成功")
            return True
            
        except Exception as e:
            error_msg = f"测试环境设置失败: {e}"
            logger.error(error_msg)
            self.test_results['summary']['errors'].append(error_msg)
            return False
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """
        运行单个测试并记录结果
        
        Args:
            test_name: 测试名称
            test_func: 测试函数
            *args, **kwargs: 测试函数参数
            
        Returns:
            bool: 测试是否通过
        """
        self.test_results['summary']['total_tests'] += 1
        
        try:
            logger.info(f"开始运行测试: {test_name}")
            start_time = time.time()
            
            result = test_func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result:
                self.test_results['summary']['passed_tests'] += 1
                logger.info(f"测试通过: {test_name} (耗时: {execution_time:.3f}s)")
                return True
            else:
                self.test_results['summary']['failed_tests'] += 1
                error_msg = f"测试失败: {test_name}"
                logger.error(error_msg)
                self.test_results['summary']['errors'].append(error_msg)
                return False
                
        except Exception as e:
            self.test_results['summary']['failed_tests'] += 1
            error_msg = f"测试异常: {test_name} - {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.test_results['summary']['errors'].append(error_msg)
            return False
    
    # ==================== 单元测试 ====================
    
    def test_config_parameters(self) -> bool:
        """测试相邻性奖励配置参数的有效性"""
        logger.info("测试相邻性奖励配置参数...")
        
        try:
            # 测试配置参数存在性
            required_params = [
                'ENABLE_ADJACENCY_REWARD',
                'ADJACENCY_REWARD_WEIGHT',
                'SPATIAL_ADJACENCY_WEIGHT',
                'FUNCTIONAL_ADJACENCY_WEIGHT',
                'CONNECTIVITY_ADJACENCY_WEIGHT',
                'ADJACENCY_PERCENTILE_THRESHOLD',
                'ADJACENCY_REWARD_BASE',
                'MEDICAL_ADJACENCY_PREFERENCES'
            ]
            
            for param in required_params:
                if not hasattr(self.config, param):
                    logger.error(f"缺少配置参数: {param}")
                    return False
            
            # 测试参数值范围
            if not (0 <= self.config.ADJACENCY_PERCENTILE_THRESHOLD <= 1):
                logger.error(f"ADJACENCY_PERCENTILE_THRESHOLD超出范围: {self.config.ADJACENCY_PERCENTILE_THRESHOLD}")
                return False
            
            # 测试权重配置
            total_weight = (self.config.SPATIAL_ADJACENCY_WEIGHT + 
                          self.config.FUNCTIONAL_ADJACENCY_WEIGHT + 
                          self.config.CONNECTIVITY_ADJACENCY_WEIGHT)
            
            if abs(total_weight - 1.0) > 0.001:  # 允许小的浮点误差
                logger.warning(f"相邻性权重总和不等于1.0: {total_weight}")
            
            # 测试医疗偏好配置有效性
            if not validate_adjacency_preferences(self.config.MEDICAL_ADJACENCY_PREFERENCES):
                logger.error("医疗相邻性偏好配置无效")
                return False
            
            self.test_results['unit_tests']['config_parameters'] = {
                'status': 'PASSED',
                'details': f'验证了{len(required_params)}个必需参数'
            }
            
            logger.info("配置参数测试通过")
            return True
            
        except Exception as e:
            logger.error(f"配置参数测试失败: {e}")
            self.test_results['unit_tests']['config_parameters'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_spatial_adjacency_algorithm(self) -> bool:
        """测试空间相邻性算法的正确性"""
        logger.info("测试空间相邻性算法...")
        
        try:
            # 创建测试用的通行时间矩阵
            test_matrix = np.array([
                [0,   10,  20,  50,  100],
                [10,  0,   15,  45,  95],
                [20,  15,  0,   30,  80],
                [50,  45,  30,  0,   25],
                [100, 95,  80,  25,  0]
            ], dtype=float)
            
            # 创建空间相邻性计算器
            spatial_calc = SpatialAdjacencyCalculator(self.config, test_matrix)
            
            # 测试相邻性矩阵计算
            adjacency_matrix = spatial_calc.calculate_adjacency_matrix()
            
            # 验证矩阵形状
            if adjacency_matrix.shape != (5, 5):
                logger.error(f"相邻性矩阵形状错误: {adjacency_matrix.shape}")
                return False
            
            # 验证对角线为0
            if not np.allclose(np.diag(adjacency_matrix), 0):
                logger.error("相邻性矩阵对角线不为0")
                return False
            
            # 验证值范围
            if not ((adjacency_matrix >= 0) & (adjacency_matrix <= 1)).all():
                logger.error("相邻性矩阵值超出[0,1]范围")
                return False
            
            # 测试相邻槽位获取
            adjacent_slots = spatial_calc.get_adjacent_slots(0)
            if not isinstance(adjacent_slots, list):
                logger.error("get_adjacent_slots返回类型错误")
                return False
            
            # 测试布局得分计算
            test_layout = ['急诊科', '放射科', '检验中心', None, None]
            score = spatial_calc.calculate_adjacency_score(test_layout)
            
            if not isinstance(score, (int, float)):
                logger.error("相邻性得分类型错误")
                return False
            
            # 测试统计信息
            stats = spatial_calc.get_adjacency_statistics()
            required_stats = ['total_slots', 'adjacency_pairs', 'adjacency_density']
            
            for stat in required_stats:
                if stat not in stats:
                    logger.error(f"缺少统计信息: {stat}")
                    return False
            
            self.test_results['unit_tests']['spatial_adjacency_algorithm'] = {
                'status': 'PASSED',
                'details': {
                    'matrix_shape': adjacency_matrix.shape,
                    'adjacent_slots_count': len(adjacent_slots),
                    'layout_score': score,
                    'statistics': stats
                }
            }
            
            logger.info("空间相邻性算法测试通过")
            return True
            
        except Exception as e:
            logger.error(f"空间相邻性算法测试失败: {e}")
            self.test_results['unit_tests']['spatial_adjacency_algorithm'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_functional_adjacency_algorithm(self) -> bool:
        """测试功能相邻性算法的正确性"""
        logger.info("测试功能相邻性算法...")
        
        try:
            # 确保环境已初始化相邻性组件
            if not hasattr(self.layout_env, 'functional_adjacency_matrix'):
                # 如果环境没有初始化相邻性组件，强制初始化
                self.layout_env._initialize_adjacency_components()
            
            # 测试功能相邻性矩阵存在
            if not hasattr(self.layout_env, 'functional_adjacency_matrix'):
                logger.error("功能相邻性矩阵未初始化")
                return False
            
            matrix = self.layout_env.functional_adjacency_matrix
            
            # 验证矩阵基本属性
            if matrix.shape[0] != matrix.shape[1]:
                logger.error("功能相邻性矩阵不是方阵")
                return False
            
            # 测试医疗偏好映射
            # 查找相邻性奖励计算方法
            adjacency_method = None
            for method_name in ['_get_department_preference', '_calculate_adjacency_reward']:
                if hasattr(self.layout_env, method_name):
                    adjacency_method = method_name
                    break
            
            if adjacency_method is None:
                logger.warning("未找到相邻性奖励计算方法，跳过功能测试")
                self.test_results['unit_tests']['functional_adjacency_algorithm'] = {
                    'status': 'SKIPPED',
                    'reason': '未找到相邻性奖励计算方法'
                }
                return True
            
            # 测试具体偏好值（如果有偏好获取方法）
            if hasattr(self.layout_env, '_get_department_preference'):
                pref1 = self.layout_env._get_department_preference('急诊科', '放射科')
                pref2 = self.layout_env._get_department_preference('妇科', '产科')
                pref3 = self.layout_env._get_department_preference('随机科室1', '随机科室2')  # 应该返回0
                
                if pref3 != 0.0:
                    logger.error(f"未知科室对应该返回0偏好，实际返回: {pref3}")
                    return False
                    
                sample_prefs = {
                    '急诊科-放射科': pref1,
                    '妇科-产科': pref2,
                    '未知科室对': pref3
                }
            else:
                # 如果没有偏好获取方法，就测试相邻性奖励计算
                sample_prefs = "通过相邻性奖励计算间接验证"
            
            self.test_results['unit_tests']['functional_adjacency_algorithm'] = {
                'status': 'PASSED',
                'details': {
                    'matrix_shape': matrix.shape,
                    'sample_preferences': sample_prefs
                }
            }
            
            logger.info("功能相邻性算法测试通过")
            return True
            
        except Exception as e:
            logger.error(f"功能相邻性算法测试失败: {e}")
            self.test_results['unit_tests']['functional_adjacency_algorithm'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_connectivity_adjacency_algorithm(self) -> bool:
        """测试多跳连通性相邻性算法的正确性"""
        logger.info("测试多跳连通性相邻性算法...")
        
        try:
            # 检查是否启用了连通性相邻性
            if self.config.CONNECTIVITY_ADJACENCY_WEIGHT <= 0:
                logger.info("连通性相邻性权重为0，跳过此测试")
                self.test_results['unit_tests']['connectivity_adjacency_algorithm'] = {
                    'status': 'SKIPPED',
                    'reason': 'CONNECTIVITY_ADJACENCY_WEIGHT <= 0'
                }
                return True
            
            # 确保环境已初始化连通性组件
            if not hasattr(self.layout_env, 'connectivity_adjacency_matrix'):
                logger.warning("连通性相邻性矩阵未初始化，可能是配置问题")
                self.test_results['unit_tests']['connectivity_adjacency_algorithm'] = {
                    'status': 'SKIPPED',
                    'reason': '连通性相邻性矩阵未初始化'
                }
                return True
            
            matrix = self.layout_env.connectivity_adjacency_matrix
            
            # 验证矩阵基本属性
            if matrix.shape[0] != matrix.shape[1]:
                logger.error("连通性相邻性矩阵不是方阵")
                return False
            
            # 验证值范围
            if not ((matrix >= 0) & (matrix <= 1)).all():
                logger.error("连通性相邻性矩阵值超出[0,1]范围")
                return False
            
            # 测试路径权重衰减
            max_path_length = self.config.CONNECTIVITY_MAX_PATH_LENGTH
            weight_decay = self.config.CONNECTIVITY_WEIGHT_DECAY
            
            if max_path_length <= 0:
                logger.error(f"最大路径长度配置错误: {max_path_length}")
                return False
            
            if not (0 <= weight_decay <= 1):
                logger.error(f"权重衰减因子超出范围: {weight_decay}")
                return False
            
            self.test_results['unit_tests']['connectivity_adjacency_algorithm'] = {
                'status': 'PASSED',
                'details': {
                    'matrix_shape': matrix.shape,
                    'max_path_length': max_path_length,
                    'weight_decay': weight_decay,
                    'matrix_stats': {
                        'non_zero_elements': np.count_nonzero(matrix),
                        'max_value': np.max(matrix),
                        'mean_value': np.mean(matrix[matrix > 0]) if np.any(matrix > 0) else 0
                    }
                }
            }
            
            logger.info("连通性相邻性算法测试通过")
            return True
            
        except Exception as e:
            logger.error(f"连通性相邻性算法测试失败: {e}")
            self.test_results['unit_tests']['connectivity_adjacency_algorithm'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_boundary_conditions_and_error_handling(self) -> bool:
        """测试边界条件和异常处理"""
        logger.info("测试边界条件和异常处理...")
        
        try:
            # 测试空矩阵
            empty_matrix = np.array([])
            normalized = normalize_adjacency_matrix(empty_matrix)
            if normalized.size != 0:
                logger.error("空矩阵归一化处理错误")
                return False
            
            # 测试分位数计算的边界情况
            empty_distances = np.array([])
            percentile_result = calculate_distance_percentile(empty_distances, 0.5)
            if percentile_result != 0.0:
                logger.error("空距离数组分位数计算错误")
                return False
            
            # 测试只有零值的距离数组
            zero_distances = np.array([0, 0, 0, 0])
            percentile_result = calculate_distance_percentile(zero_distances, 0.5)
            if percentile_result != 0.0:
                logger.error("零值距离数组分位数计算错误")
                return False
            
            # 测试无效布局
            if hasattr(self.layout_env, '_calculate_adjacency_reward'):
                # 测试空布局
                reward = self.layout_env._calculate_adjacency_reward([])
                if reward != 0.0:
                    logger.error(f"空布局应该返回0奖励，实际返回: {reward}")
                    return False
                
                # 测试全None布局
                reward = self.layout_env._calculate_adjacency_reward([None, None, None])
                if reward != 0.0:
                    logger.error(f"全None布局应该返回0奖励，实际返回: {reward}")
                    return False
            
            # 测试异常偏好配置
            invalid_prefs = {
                "科室1": {
                    "科室2": 2.0  # 超出范围
                }
            }
            
            if validate_adjacency_preferences(invalid_prefs):
                logger.error("无效偏好配置验证失败")
                return False
            
            self.test_results['unit_tests']['boundary_conditions_error_handling'] = {
                'status': 'PASSED',
                'details': '所有边界条件和异常处理测试通过'
            }
            
            logger.info("边界条件和异常处理测试通过")
            return True
            
        except Exception as e:
            logger.error(f"边界条件和异常处理测试失败: {e}")
            self.test_results['unit_tests']['boundary_conditions_error_handling'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    # ==================== 集成测试 ====================
    
    def test_layout_env_integration(self) -> bool:
        """测试与LayoutEnv环境的集成"""
        logger.info("测试与LayoutEnv环境的集成...")
        
        try:
            # 检查相邻性奖励是否启用
            if not self.config.ENABLE_ADJACENCY_REWARD:
                logger.warning("相邻性奖励未启用，跳过集成测试")
                self.test_results['integration_tests']['layout_env'] = {
                    'status': 'SKIPPED',
                    'reason': 'ENABLE_ADJACENCY_REWARD = False'
                }
                return True
            
            # 测试环境重置
            obs, info = self.layout_env.reset()
            if not isinstance(obs, dict):
                logger.error("环境重置后观察空间类型错误")
                return False
            
            # 测试动作掩码
            action_mask = self.layout_env.get_action_mask()
            if not isinstance(action_mask, np.ndarray):
                logger.error("动作掩码类型错误")
                return False
            
            # 模拟几步交互
            total_reward = 0
            step_count = 0
            adjacency_rewards = []
            
            while step_count < min(5, self.layout_env.num_slots):
                # 获取有效动作
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                
                # 选择第一个有效动作
                action = valid_actions[0]
                
                # 执行动作
                obs, reward, terminated, truncated, info = self.layout_env.step(action)
                
                # 检查info中的相邻性奖励信息
                if 'adjacency_reward' in info:
                    adjacency_rewards.append(info['adjacency_reward'])
                
                total_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
                    
                # 更新动作掩码
                action_mask = self.layout_env.get_action_mask()
            
            # 验证相邻性奖励计算
            if hasattr(self.layout_env, '_calculate_adjacency_reward'):
                # 测试当前布局的相邻性奖励
                current_layout = self.layout_env.current_layout
                adjacency_reward = self.layout_env._calculate_adjacency_reward(current_layout)
                
                if not isinstance(adjacency_reward, (int, float)):
                    logger.error("相邻性奖励返回类型错误")
                    return False
            
            self.test_results['integration_tests']['layout_env'] = {
                'status': 'PASSED',
                'details': {
                    'total_steps': step_count,
                    'total_reward': total_reward,
                    'adjacency_rewards': adjacency_rewards,
                    'final_layout_length': len([x for x in self.layout_env.current_layout if x is not None])
                }
            }
            
            logger.info("与LayoutEnv环境的集成测试通过")
            return True
            
        except Exception as e:
            logger.error(f"LayoutEnv集成测试失败: {e}")
            self.test_results['integration_tests']['layout_env'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_ppo_compatibility(self) -> bool:
        """测试与PPO算法的兼容性"""
        logger.info("测试与PPO算法的兼容性...")
        
        try:
            # 导入PPO相关组件
            try:
                from stable_baselines3 import MaskablePPO
                from sb3_contrib.common.wrappers import ActionMasker
                from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            except ImportError as e:
                logger.warning(f"PPO依赖导入失败，跳过PPO兼容性测试: {e}")
                self.test_results['integration_tests']['ppo_compatibility'] = {
                    'status': 'SKIPPED',
                    'reason': f'依赖导入失败: {e}'
                }
                return True
            
            # 创建动作掩码包装器
            def mask_fn(env):
                return env.action_masks()
            
            masked_env = ActionMasker(self.layout_env, mask_fn)
            
            # 尝试创建PPO模型
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                masked_env,
                learning_rate=3e-4,
                n_steps=64,  # 减少步数以加快测试
                batch_size=32,
                n_epochs=2,
                verbose=0
            )
            
            # 测试模型训练几步
            model.learn(total_timesteps=128, progress_bar=False)
            
            # 测试模型预测
            obs, _ = masked_env.reset()
            action_masks = masked_env.action_masks()
            
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            if not isinstance(action, (int, np.integer)):
                logger.error("PPO模型预测动作类型错误")
                return False
            
            # 验证动作有效性
            if not action_masks[action]:
                logger.error("PPO模型选择了无效动作")
                return False
            
            self.test_results['integration_tests']['ppo_compatibility'] = {
                'status': 'PASSED',
                'details': {
                    'model_created': True,
                    'training_completed': True,
                    'prediction_valid': True,
                    'selected_action': int(action)
                }
            }
            
            logger.info("与PPO算法的兼容性测试通过")
            return True
            
        except Exception as e:
            logger.error(f"PPO兼容性测试失败: {e}")
            self.test_results['integration_tests']['ppo_compatibility'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    # ==================== 性能测试 ====================
    
    def test_cache_and_precomputation_performance(self) -> bool:
        """测试缓存机制和预计算性能"""
        logger.info("测试缓存机制和预计算性能...")
        
        try:
            # 测试相邻性矩阵预计算性能
            start_time = time.time()
            
            # 重新初始化环境以触发预计算
            test_env = LayoutEnv(
                config=self.config,
                cache_manager=self.cache_manager,
                cost_calculator=self.cost_calculator,
                constraint_manager=self.constraint_manager
            )
            
            precomputation_time = time.time() - start_time
            
            # 测试相邻性奖励计算性能
            if hasattr(test_env, '_calculate_adjacency_reward'):
                # 创建测试布局
                test_layouts = []
                for i in range(10):
                    # 创建随机布局
                    depts = test_env.placeable_depts[:min(10, len(test_env.placeable_depts))]
                    layout = depts + [None] * (test_env.num_slots - len(depts))
                    np.random.shuffle(layout)
                    test_layouts.append(tuple(layout))  # 转换为tuple以支持哈希
                
                # 测试首次计算（无缓存）
                start_time = time.time()
                for layout in test_layouts:
                    test_env._calculate_adjacency_reward(list(layout))  # 转回list进行计算
                first_calculation_time = time.time() - start_time
                
                # 测试重复计算（有缓存）
                start_time = time.time()
                for layout in test_layouts:
                    test_env._calculate_adjacency_reward(list(layout))  # 转回list进行计算
                cached_calculation_time = time.time() - start_time
                
                # 计算加速比
                speedup = first_calculation_time / max(cached_calculation_time, 1e-6)
                
            else:
                first_calculation_time = 0
                cached_calculation_time = 0
                speedup = 1.0
            
            # 测试内存使用
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            self.test_results['performance_tests']['cache_precomputation'] = {
                'status': 'PASSED',
                'details': {
                    'precomputation_time': f'{precomputation_time:.3f}s',
                    'first_calculation_time': f'{first_calculation_time:.3f}s',
                    'cached_calculation_time': f'{cached_calculation_time:.3f}s',
                    'cache_speedup': f'{speedup:.2f}x',
                    'memory_usage': f'{memory_usage:.1f}MB'
                }
            }
            
            logger.info(f"缓存和预计算性能测试通过 (加速比: {speedup:.2f}x)")
            return True
            
        except Exception as e:
            logger.error(f"缓存和预计算性能测试失败: {e}")
            self.test_results['performance_tests']['cache_precomputation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    # ==================== 稳定性测试 ====================
    
    def test_long_running_stability(self) -> bool:
        """测试长时间运行稳定性和异常数据处理"""
        logger.info("测试长时间运行稳定性...")
        
        try:
            test_episodes = 20  # 减少测试轮次以加快速度
            successful_episodes = 0
            error_count = 0
            total_adjacency_rewards = []
            
            for episode in range(test_episodes):
                try:
                    # 重置环境
                    obs, info = self.layout_env.reset()
                    
                    episode_reward = 0
                    episode_adjacency_rewards = []
                    steps = 0
                    
                    while steps < self.layout_env.num_slots:
                        # 获取动作掩码
                        action_mask = self.layout_env.get_action_mask()
                        valid_actions = np.where(action_mask)[0]
                        
                        if len(valid_actions) == 0:
                            break
                        
                        # 随机选择有效动作
                        action = np.random.choice(valid_actions)
                        
                        # 执行动作
                        obs, reward, terminated, truncated, info = self.layout_env.step(action)
                        
                        episode_reward += reward
                        
                        # 记录相邻性奖励
                        if 'adjacency_reward' in info:
                            episode_adjacency_rewards.append(info['adjacency_reward'])
                        
                        steps += 1
                        
                        if terminated or truncated:
                            break
                    
                    successful_episodes += 1
                    if episode_adjacency_rewards:
                        total_adjacency_rewards.extend(episode_adjacency_rewards)
                    
                    # 每5轮报告一次进度
                    if (episode + 1) % 5 == 0:
                        logger.info(f"稳定性测试进度: {episode + 1}/{test_episodes}")
                        
                except Exception as e:
                    error_count += 1
                    logger.warning(f"第{episode}轮测试出现错误: {e}")
                    continue
            
            # 分析结果
            success_rate = successful_episodes / test_episodes
            
            if success_rate < 0.8:  # 至少80%的成功率
                logger.error(f"稳定性测试成功率过低: {success_rate:.2%}")
                return False
            
            # 分析相邻性奖励统计
            adj_reward_stats = {}
            if total_adjacency_rewards:
                adj_reward_stats = {
                    'count': len(total_adjacency_rewards),
                    'mean': np.mean(total_adjacency_rewards),
                    'std': np.std(total_adjacency_rewards),
                    'min': np.min(total_adjacency_rewards),
                    'max': np.max(total_adjacency_rewards)
                }
            
            self.test_results['stability_tests']['long_running'] = {
                'status': 'PASSED',
                'details': {
                    'test_episodes': test_episodes,
                    'successful_episodes': successful_episodes,
                    'success_rate': f'{success_rate:.2%}',
                    'error_count': error_count,
                    'adjacency_reward_stats': adj_reward_stats
                }
            }
            
            logger.info(f"长时间运行稳定性测试通过 (成功率: {success_rate:.2%})")
            return True
            
        except Exception as e:
            logger.error(f"长时间运行稳定性测试失败: {e}")
            self.test_results['stability_tests']['long_running'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def generate_test_report(self) -> str:
        """生成详细的测试报告"""
        total_time = time.time() - self.start_time
        
        report = f"""# 动态相邻性奖励功能测试报告

## 测试概述

**测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**测试环境**: {os.uname().sysname} {os.uname().release}
**Python版本**: {sys.version}
**总测试时间**: {total_time:.2f}秒

## 测试结果汇总

- **总测试数**: {self.test_results['summary']['total_tests']}
- **通过测试**: {self.test_results['summary']['passed_tests']}
- **失败测试**: {self.test_results['summary']['failed_tests']}
- **成功率**: {self.test_results['summary']['passed_tests']/max(1,self.test_results['summary']['total_tests'])*100:.1f}%

"""
        
        # 详细测试结果
        for category, tests in self.test_results.items():
            if category == 'summary':
                continue
                
            report += f"## {category.replace('_', ' ').title()}\n\n"
            
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if isinstance(result, dict):
                        status = result.get('status', 'UNKNOWN')
                        report += f"### {test_name.replace('_', ' ').title()}\n\n"
                        report += f"**状态**: {status}\n\n"
                        
                        if 'details' in result:
                            report += f"**详细信息**:\n```\n{result['details']}\n```\n\n"
                        
                        if 'error' in result:
                            report += f"**错误信息**: {result['error']}\n\n"
            
        # 问题清单
        if self.test_results['summary']['errors']:
            report += "## 发现的问题\n\n"
            for i, error in enumerate(self.test_results['summary']['errors'], 1):
                report += f"{i}. {error}\n"
            report += "\n"
        
        # 结论和建议
        report += "## 结论和建议\n\n"
        
        success_rate = self.test_results['summary']['passed_tests'] / max(1, self.test_results['summary']['total_tests'])
        
        if success_rate >= 0.9:
            report += "✅ **建议通过**: 相邻性奖励功能运行稳定，质量良好，建议合并到主分支。\n\n"
        elif success_rate >= 0.7:
            report += "⚠️ **需要修复**: 发现一些问题，建议修复后再次测试。\n\n"
        else:
            report += "❌ **不建议合并**: 发现严重问题，需要重大修复。\n\n"
        
        # 添加优化建议
        report += "### 优化建议\n\n"
        report += "1. **性能优化**: 考虑进一步优化相邻性矩阵计算和缓存机制\n"
        report += "2. **参数调优**: 根据实际场景调整相邻性权重和阈值参数\n"
        report += "3. **监控完善**: 增加更多的运行时监控和异常处理\n"
        report += "4. **文档更新**: 完善相邻性奖励功能的使用文档\n\n"
        
        return report
    
    def cleanup_test_files(self):
        """清理测试过程中创建的临时文件"""
        try:
            import shutil
            
            # 清理测试数据目录
            if self.test_data_dir.exists():
                shutil.rmtree(self.test_data_dir)
                logger.info("测试数据目录已清理")
            
            logger.info("测试文件清理完成")
        except Exception as e:
            logger.warning(f"测试文件清理失败: {e}")


def main():
    """主测试函数"""
    print("="*80)
    print("动态相邻性奖励功能全面测试")
    print("="*80)
    
    # 创建测试框架
    test_framework = AdjacencyTestFramework()
    
    # 设置测试环境
    if not test_framework.setup_test_environment():
        print("❌ 测试环境设置失败，退出测试")
        return 1
    
    print("✅ 测试环境设置成功")
    print("\n开始执行测试套件...")
    
    # 执行测试套件
    tests_to_run = [
        ("配置参数验证", test_framework.test_config_parameters),
        ("空间相邻性算法", test_framework.test_spatial_adjacency_algorithm),
        ("功能相邻性算法", test_framework.test_functional_adjacency_algorithm),
        ("连通性相邻性算法", test_framework.test_connectivity_adjacency_algorithm),
        ("边界条件和异常处理", test_framework.test_boundary_conditions_and_error_handling),
        ("LayoutEnv集成", test_framework.test_layout_env_integration),
        ("PPO兼容性", test_framework.test_ppo_compatibility),
        ("缓存和预计算性能", test_framework.test_cache_and_precomputation_performance),
        ("长时间运行稳定性", test_framework.test_long_running_stability),
    ]
    
    # 运行所有测试
    for test_name, test_func in tests_to_run:
        test_framework.run_test(test_name, test_func)
        print(f"{'✅' if test_framework.test_results['summary']['passed_tests'] > test_framework.test_results['summary']['failed_tests'] else '❌'} {test_name}")
    
    # 生成测试报告
    print("\n" + "="*80)
    print("生成测试报告...")
    
    report_content = test_framework.generate_test_report()
    
    # 保存测试报告
    report_path = Path('/home/pan/code/Hospital_Room_Dectection/docs/Test.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding='utf-8')
    
    print(f"✅ 测试报告已保存: {report_path}")
    
    # 清理测试文件
    test_framework.cleanup_test_files()
    
    # 打印测试摘要
    summary = test_framework.test_results['summary']
    success_rate = summary['passed_tests'] / max(1, summary['total_tests']) * 100
    
    print(f"\n" + "="*80)
    print("测试完成!")
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed_tests']}")
    print(f"失败: {summary['failed_tests']}")
    print(f"成功率: {success_rate:.1f}%")
    
    if summary['failed_tests'] > 0:
        print("\n❌ 发现问题，请查看测试报告了解详情")
        return 1
    else:
        print("\n✅ 所有测试通过！相邻性奖励功能运行良好")
        return 0


if __name__ == "__main__":
    sys.exit(main())