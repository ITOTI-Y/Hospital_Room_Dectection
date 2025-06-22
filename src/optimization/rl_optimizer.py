"""
基于深度强化学习的医院布局优化器实现

此模块提供基于Deep Q-Network (DQN)的布局优化解决方案，
替代传统的贪心+随机搜索方法。
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle
from collections import deque
from typing import List, Dict, Optional, Tuple, Set, Any, Sequence
from dataclasses import dataclass
import json

from src.analysis.process_flow import PathFinder
from src.config import NetworkConfig
from src.optimization.optimizer import (
    PhysicalLocation,
    FunctionalAssignment,
    WorkflowDefinition,
    EvaluatedWorkflowOutcome,
    LayoutObjectiveCalculator
)

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class RLConfig:
    """强化学习配置参数"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_buffer_size: int = 50000
    target_update_frequency: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99  # 折扣因子
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    hidden_dim3: int = 128
    gradient_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class StateEncoder:
    """状态编码器，将FunctionalAssignment转换为神经网络输入"""
    
    def __init__(self, physical_locations: List[PhysicalLocation], 
                 workflow_definitions: Sequence[WorkflowDefinition]):
        self.physical_locations = physical_locations
        self.workflow_definitions = workflow_definitions
        
        # 构建索引映射
        self.location_to_idx = {loc.name_id: idx for idx, loc in enumerate(physical_locations)}
        self.functional_types = list(set(loc.original_functional_type for loc in physical_locations))
        self.function_to_idx = {func: idx for idx, func in enumerate(self.functional_types)}
        
        # 计算状态维度
        self.n_locations = len(physical_locations)
        self.n_functions = len(self.functional_types)
        self.assignment_dim = self.n_locations * self.n_functions
        self.workflow_feature_dim = len(workflow_definitions) * 2  # weight + sequence_length
        self.location_feature_dim = self.n_locations * 2  # area + is_swappable
        
        self.state_dim = self.assignment_dim + self.workflow_feature_dim + self.location_feature_dim
        
        logger.info(f"StateEncoder initialized: state_dim={self.state_dim}, "
                   f"n_locations={self.n_locations}, n_functions={self.n_functions}")
    
    def encode(self, assignment: FunctionalAssignment) -> np.ndarray:
        """将FunctionalAssignment编码为状态向量"""
        state = np.zeros(self.state_dim, dtype=np.float32)
        offset = 0
        
        # 1. Assignment matrix (flattened)
        assignment_matrix = np.zeros((self.n_locations, self.n_functions), dtype=np.float32)
        for func_type, location_ids in assignment.assignment_map.items():
            if func_type in self.function_to_idx:
                func_idx = self.function_to_idx[func_type]
                for location_id in location_ids:
                    if location_id in self.location_to_idx:
                        loc_idx = self.location_to_idx[location_id]
                        assignment_matrix[loc_idx, func_idx] = 1.0
        
        state[offset:offset + self.assignment_dim] = assignment_matrix.flatten()
        offset += self.assignment_dim
        
        # 2. Workflow features
        workflow_features = np.zeros(self.workflow_feature_dim, dtype=np.float32)
        for i, wf in enumerate(self.workflow_definitions):
            workflow_features[i * 2] = wf.weight
            workflow_features[i * 2 + 1] = len(wf.functional_sequence)
        
        state[offset:offset + self.workflow_feature_dim] = workflow_features
        offset += self.workflow_feature_dim
        
        # 3. Location features
        location_features = np.zeros(self.location_feature_dim, dtype=np.float32)
        for i, loc in enumerate(self.physical_locations):
            location_features[i * 2] = loc.area / 1000.0  # 归一化面积
            location_features[i * 2 + 1] = float(loc.is_swappable)
        
        state[offset:offset + self.location_feature_dim] = location_features
        
        return state


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor, torch.Tensor]:
        """从缓冲区采样batch"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """Dueling DQN网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        super(DuelingDQN, self).__init__()
        
        # 共享层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim1, config.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim2, config.hidden_dim3),
            nn.ReLU()
        )
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_dim3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_dim3, 64),
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling架构公式: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化网络
        self.q_network = DuelingDQN(state_dim, action_dim, config).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, config).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # 初始化目标网络权重
        self._update_target_network()
        
        # 探索参数
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        logger.info(f"DQNAgent initialized on device: {self.device}")
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """选择动作 (epsilon-greedy策略)"""
        if random.random() > self.epsilon:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                if valid_actions is not None:
                    # 屏蔽无效动作
                    masked_q_values = q_values.clone()
                    mask = torch.ones(q_values.size(1), dtype=torch.bool)
                    mask[valid_actions] = False
                    masked_q_values[0, mask] = float('-inf')
                    action = masked_q_values.argmax().item()
                else:
                    action = q_values.argmax().item()
        else:
            # 随机探索
            if valid_actions is not None:
                action = random.choice(valid_actions)
            else:
                action = random.randint(0, self.q_network.advantage_stream[-1].out_features - 1)
        
        return action
    
    def learn(self, replay_buffer: ReplayBuffer):
        """从经验回放中学习"""
        if len(replay_buffer) < self.config.batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.config.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: 使用主网络选择动作，目标网络计算Q值
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        self.steps_done += 1
        
        # 定期更新目标网络
        if self.steps_done % self.config.target_update_frequency == 0:
            self._update_target_network()
            logger.info(f"Target network updated at step {self.steps_done}")
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        logger.info(f"Model loaded from {filepath}")


class RLEnvironment:
    """强化学习环境包装器"""
    
    def __init__(self, 
                 objective_calculator: LayoutObjectiveCalculator,
                 physical_locations: List[PhysicalLocation],
                 initial_assignment: FunctionalAssignment):
        self.objective_calculator = objective_calculator
        self.physical_locations = physical_locations
        self.initial_assignment = initial_assignment
        self.current_assignment: Optional[FunctionalAssignment] = None
        self.current_objective: Optional[float] = None
        self.best_objective = float('inf')
        
        # 构建有效交换对
        self._build_valid_swap_pairs()
        
        logger.info(f"RLEnvironment initialized with {len(self.valid_swap_pairs)} valid actions")
    
    def _build_valid_swap_pairs(self):
        """构建有效的交换对列表"""
        swappable_locations = [loc for loc in self.physical_locations if loc.is_swappable]
        self.valid_swap_pairs = []
        
        for i in range(len(swappable_locations)):
            for j in range(i + 1, len(swappable_locations)):
                loc_a = swappable_locations[i]
                loc_b = swappable_locations[j]
                
                # 检查面积兼容性 (简化版本，可以根据需要调整)
                area_ratio = abs(loc_a.area - loc_b.area) / max(loc_a.area, loc_b.area)
                if area_ratio <= 0.5:  # 面积差异不超过50%
                    self.valid_swap_pairs.append((loc_a.name_id, loc_b.name_id))
    
    def reset(self) -> FunctionalAssignment:
        """重置环境到初始状态"""
        self.current_assignment = FunctionalAssignment(self.initial_assignment.get_map_copy())
        self.current_objective, _ = self.objective_calculator.evaluate(self.current_assignment)
        self.best_objective = self.current_objective
        return self.current_assignment
    
    def step(self, action: int) -> Tuple[FunctionalAssignment, float, bool]:
        """执行动作并返回新状态、奖励、完成标志"""
        if action >= len(self.valid_swap_pairs):
            raise ValueError(f"Invalid action {action}, max action is {len(self.valid_swap_pairs) - 1}")
        
        if self.current_assignment is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # 执行交换
        loc_a_id, loc_b_id = self.valid_swap_pairs[action]
        new_assignment = self.current_assignment.apply_functional_swap(loc_a_id, loc_b_id)
        
        # 评估新状态
        new_objective, _ = self.objective_calculator.evaluate(
            new_assignment, 
            base_assignment_for_cache=self.current_assignment,
            changed_locations=(loc_a_id, loc_b_id)
        )
        
        # 计算奖励
        if self.current_objective is not None:
            reward = self._calculate_reward(self.current_objective, new_objective)
        else:
            reward = -1.0  # 如果当前目标值为None，给予惩罚
        
        # 更新状态
        self.current_assignment = new_assignment
        self.current_objective = new_objective
        
        # 更新最佳解
        if new_objective < self.best_objective:
            self.best_objective = new_objective
        
        # 简单的完成条件 (可以根据需要调整)
        done = False
        
        return new_assignment, reward, done
    
    def _calculate_reward(self, old_objective: float, new_objective: float) -> float:
        """计算奖励"""
        if old_objective == float('inf') and new_objective == float('inf'):
            return -1.0  # 两个都是无效状态
        elif old_objective == float('inf'):
            return 10.0  # 从无效状态转到有效状态
        elif new_objective == float('inf'):
            return -10.0  # 从有效状态转到无效状态
        else:
            # 基于改进的奖励
            improvement = old_objective - new_objective
            if improvement > 0:
                return improvement / old_objective * 100  # 正向改进奖励
            else:
                return improvement / old_objective * 10   # 负向改进惩罚(较小)
    
    def get_valid_actions(self) -> List[int]:
        """获取当前状态下的有效动作"""
        return list(range(len(self.valid_swap_pairs)))
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        return len(self.valid_swap_pairs)


class RLLayoutOptimizer:
    """基于强化学习的布局优化器"""
    
    def __init__(self,
                 path_finder: PathFinder,
                 workflow_definitions: Sequence[WorkflowDefinition],
                 config: NetworkConfig,
                 rl_config: Optional[RLConfig] = None,
                 area_tolerance_ratio: float = 0.3):
        
        self.path_finder = path_finder
        self.workflow_definitions = workflow_definitions
        self.config = config
        self.rl_config = rl_config or RLConfig()
        self.area_tolerance_ratio = area_tolerance_ratio
        
        # 初始化物理位置
        self.physical_locations = self._initialize_physical_locations()
        
        # 初始化目标函数计算器
        self.objective_calculator = LayoutObjectiveCalculator(
            workflow_definitions=workflow_definitions,
            path_finder=path_finder,
            n_jobs_for_flows=1  # RL训练时使用单进程避免并发问题
        )
        
        # 初始化状态编码器
        self.state_encoder = StateEncoder(self.physical_locations, workflow_definitions)
        
        logger.info(f"RLLayoutOptimizer initialized with {len(self.physical_locations)} locations")
    
    def _initialize_physical_locations(self) -> List[PhysicalLocation]:
        """初始化物理位置列表 (从现有优化器复制逻辑)"""
        physical_locations = []
        
        # 遍历所有Name_ID构建PhysicalLocation对象
        for name_id in self.path_finder.all_name_ids:
            name, node_id = self.path_finder._parse_name_id(name_id)
            
            # 估算面积 (这里使用简化逻辑，实际项目中可能需要从其他数据源获取)
            area = hash(name_id) % 1000 + 100  # 100-1099的随机面积
            
            # 判断是否可交换
            is_swappable = name not in self.config.CONNECTION_TYPES and name not in self.config.BAN_TYPES
            
            physical_location = PhysicalLocation(
                name_id=name_id,
                original_functional_type=name,
                area=float(area),
                is_swappable=is_swappable
            )
            physical_locations.append(physical_location)
        
        return physical_locations
    
    def run_optimization(self,
                        initial_assignment: FunctionalAssignment,
                        max_iterations: int = 1000,
                        save_model_path: Optional[str] = None) -> Tuple[FunctionalAssignment, float, List[EvaluatedWorkflowOutcome]]:
        """运行强化学习优化"""
        
        logger.info("开始强化学习布局优化...")
        
        # 初始化环境
        environment = RLEnvironment(
            objective_calculator=self.objective_calculator,
            physical_locations=self.physical_locations,
            initial_assignment=initial_assignment
        )
        
        # 初始化智能体
        agent = DQNAgent(
            state_dim=self.state_encoder.state_dim,
            action_dim=environment.get_action_dim(),
            config=self.rl_config
        )
        
        # 初始化经验回放缓冲区
        replay_buffer = ReplayBuffer(
            capacity=self.rl_config.replay_buffer_size,
            state_dim=self.state_encoder.state_dim
        )
        
        # 训练统计
        episode_rewards = []
        best_assignment = initial_assignment
        best_objective = float('inf')
        
        # 训练循环
        for episode in range(max_iterations):
            current_assignment = environment.reset()
            current_state = self.state_encoder.encode(current_assignment)
            episode_reward = 0
            
            # 每个episode最多100步
            for step in range(100):
                # 选择动作
                valid_actions = environment.get_valid_actions()
                action = agent.act(current_state, valid_actions)
                
                # 执行动作
                next_assignment, reward, done = environment.step(action)
                next_state = self.state_encoder.encode(next_assignment)
                
                # 存储经验
                replay_buffer.push(current_state, action, reward, next_state, done)
                
                # 学习
                agent.learn(replay_buffer)
                
                # 更新状态
                current_state = next_state
                current_assignment = next_assignment
                episode_reward += reward
                
                # 更新最佳解
                if environment.current_objective is not None and environment.current_objective < best_objective:
                    best_objective = environment.current_objective
                    best_assignment = FunctionalAssignment(current_assignment.get_map_copy())
                    logger.info(f"Episode {episode}, Step {step}: New best objective = {best_objective:.2f}")
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 记录进度
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                           f"Epsilon = {agent.epsilon:.3f}, Best Objective = {best_objective:.2f}")
        
        # 保存模型
        if save_model_path:
            agent.save(save_model_path)
        
        # 评估最终结果
        final_objective, final_outcomes = self.objective_calculator.evaluate(best_assignment)
        
        logger.info(f"强化学习优化完成！最终目标值: {final_objective:.2f}")
        
        return best_assignment, final_objective, final_outcomes