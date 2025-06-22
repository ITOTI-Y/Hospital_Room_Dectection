#!/usr/bin/env python3
"""
强化学习布局优化器端到端测试

此脚本测试RLLayoutOptimizer的核心功能
"""

import unittest
import tempfile
import pathlib
from unittest.mock import Mock, patch

# 条件导入
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 导入被测试的模块
from src.config import NetworkConfig, COLOR_MAP
from src.optimization.rl_optimizer import (
    RLConfig, StateEncoder, ReplayBuffer, DuelingDQN, 
    DQNAgent, RLEnvironment, RLLayoutOptimizer
)
from src.optimization.optimizer import (
    PhysicalLocation, FunctionalAssignment, WorkflowDefinition,
    LayoutObjectiveCalculator
)


class TestStateEncoder(unittest.TestCase):
    """测试状态编码器"""
    
    def setUp(self):
        self.physical_locations = [
            PhysicalLocation("妇科_101", "妇科", 100.0, True),
            PhysicalLocation("门_201", "门", 50.0, False),
            PhysicalLocation("采血处_301", "采血处", 150.0, True)
        ]
        
        self.workflow_definitions = [
            WorkflowDefinition("WF1", ["妇科", "采血处"], 1.0),
            WorkflowDefinition("WF2", ["采血处", "妇科"], 0.5)
        ]
        
        self.encoder = StateEncoder(self.physical_locations, self.workflow_definitions)
    
    def test_encoder_initialization(self):
        """测试编码器初始化"""
        self.assertEqual(self.encoder.n_locations, 3)
        self.assertEqual(self.encoder.n_functions, 3)  # 妇科, 门, 采血处
        self.assertGreater(self.encoder.state_dim, 0)
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_encode_assignment(self):
        """测试功能分配编码"""
        assignment_map = {
            "妇科": ["妇科_101"],
            "门": ["门_201"],
            "采血处": ["采血处_301"]
        }
        assignment = FunctionalAssignment(assignment_map)
        
        state = self.encoder.encode(assignment)
        
        self.assertEqual(len(state), self.encoder.state_dim)
        self.assertTrue(np.all(np.isfinite(state)))  # 所有值都是有限的


class TestReplayBuffer(unittest.TestCase):
    """测试经验回放缓冲区"""
    
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100, state_dim=10)
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_buffer_operations(self):
        """测试缓冲区基本操作"""
        state = np.random.random(10)
        next_state = np.random.random(10)
        
        # 添加经验
        self.buffer.push(state, 0, 1.0, next_state, False)
        self.assertEqual(len(self.buffer), 1)
        
        # 添加更多经验
        for i in range(50):
            self.buffer.push(
                np.random.random(10), i % 5, 
                np.random.random(), np.random.random(10), 
                False
            )
        
        self.assertEqual(len(self.buffer), 51)
        
        # 测试采样
        if len(self.buffer) >= 32:
            batch = self.buffer.sample(32)
            self.assertEqual(len(batch), 5)  # states, actions, rewards, next_states, dones


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDuelingDQN(unittest.TestCase):
    """测试DuelingDQN网络"""
    
    def setUp(self):
        self.config = RLConfig()
        self.network = DuelingDQN(state_dim=20, action_dim=10, config=self.config)
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 4
        state_dim = 20
        
        x = torch.randn(batch_size, state_dim)
        output = self.network(x)
        
        self.assertEqual(output.shape, (batch_size, 10))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestRLEnvironment(unittest.TestCase):
    """测试强化学习环境"""
    
    def setUp(self):
        # 创建mock对象
        self.mock_objective_calculator = Mock(spec=LayoutObjectiveCalculator)
        self.mock_objective_calculator.evaluate.return_value = (100.0, [])
        
        self.physical_locations = [
            PhysicalLocation("妇科_101", "妇科", 100.0, True),
            PhysicalLocation("采血处_301", "采血处", 150.0, True),
            PhysicalLocation("超声科_401", "超声科", 120.0, True)
        ]
        
        initial_assignment = FunctionalAssignment({
            "妇科": ["妇科_101"],
            "采血处": ["采血处_301"],
            "超声科": ["超声科_401"]
        })
        
        self.environment = RLEnvironment(
            objective_calculator=self.mock_objective_calculator,
            physical_locations=self.physical_locations,
            initial_assignment=initial_assignment
        )
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        self.assertGreater(len(self.environment.valid_swap_pairs), 0)
        self.assertGreater(self.environment.get_action_dim(), 0)
    
    def test_reset_and_step(self):
        """测试环境重置和步进"""
        # 重置环境
        assignment = self.environment.reset()
        self.assertIsNotNone(assignment)
        
        # 执行一步
        if len(self.environment.get_valid_actions()) > 0:
            action = 0
            new_assignment, reward, done = self.environment.step(action)
            
            self.assertIsNotNone(new_assignment)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)


class TestRLLayoutOptimizer(unittest.TestCase):
    """测试强化学习布局优化器"""
    
    def setUp(self):
        # 创建临时文件用于测试
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.write("Name_ID,妇科_101,采血处_301\n")
        self.temp_file.write("妇科_101,0,10\n")
        self.temp_file.write("采血处_301,10,0\n")
        self.temp_file.close()
        
        self.config = NetworkConfig()
        
        # 创建mock PathFinder
        self.mock_path_finder = Mock()
        self.mock_path_finder.all_name_ids = {"妇科_101", "采血处_301"}
        self.mock_path_finder.name_to_ids_map = {
            "妇科": ["妇科_101"],
            "采血处": ["采血处_301"]
        }
        self.mock_path_finder._parse_name_id = lambda x: x.split('_')
        
        self.workflow_definitions = [
            WorkflowDefinition("WF1", ["妇科", "采血处"], 1.0)
        ]
    
    def tearDown(self):
        # 清理临时文件
        pathlib.Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        rl_config = RLConfig()
        
        optimizer = RLLayoutOptimizer(
            path_finder=self.mock_path_finder,
            workflow_definitions=self.workflow_definitions,
            config=self.config,
            rl_config=rl_config
        )
        
        self.assertIsNotNone(optimizer.state_encoder)
        self.assertIsNotNone(optimizer.objective_calculator)
        self.assertGreater(len(optimizer.physical_locations), 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [TestStateEncoder, TestReplayBuffer, TestRLEnvironment, TestRLLayoutOptimizer]
    if TORCH_AVAILABLE:
        test_classes.append(TestDuelingDQN)
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)