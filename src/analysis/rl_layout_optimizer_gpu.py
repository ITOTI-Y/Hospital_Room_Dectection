"""
GPU-accelerated reinforcement learning-based layout optimization system.

This module provides a PyTorch-based implementation of the RL layout optimizer
that can leverage GPU acceleration for faster training and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import logging
import random
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from .rl_layout_optimizer import LayoutState, LayoutAction, LayoutEnvironment, create_default_workflow_patterns

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Deep Q-Network for layout optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class GPULayoutEnvironment(LayoutEnvironment):
    """GPU-accelerated layout environment using PyTorch tensors"""
    
    def __init__(self, csv_filepath: str, workflow_patterns: List[List[str]] = None, device: str = None):
        super().__init__(csv_filepath, workflow_patterns)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self._prepare_gpu_tensors()
        
    def _prepare_gpu_tensors(self):
        """Convert travel times data to GPU tensors for faster computation"""
        self.space_to_idx = {space: idx for idx, space in enumerate(self.spaces)}
        self.idx_to_space = {idx: space for space, idx in self.space_to_idx.items()}
        
        travel_matrix = np.full((len(self.spaces), len(self.spaces)), np.inf)
        
        for i, from_space in enumerate(self.spaces):
            for j, to_space in enumerate(self.spaces):
                if from_space in self.travel_times_df.index and to_space in self.travel_times_df.columns:
                    time_val = self.travel_times_df.loc[from_space, to_space]
                    if isinstance(time_val, (int, float)):
                        travel_matrix[i, j] = float(time_val)
                    elif isinstance(time_val, str) and time_val != '∞':
                        try:
                            travel_matrix[i, j] = float(time_val)
                        except ValueError:
                            pass
        
        self.travel_times_tensor = torch.tensor(travel_matrix, dtype=torch.float32, device=self.device)
        logger.info(f"Travel times tensor shape: {self.travel_times_tensor.shape}")
    
    def _get_travel_time_gpu(self, from_space: str, to_space: str) -> float:
        """Get travel time using GPU tensor lookup"""
        try:
            from_idx = self.space_to_idx.get(from_space)
            to_idx = self.space_to_idx.get(to_space)
            
            if from_idx is not None and to_idx is not None:
                time_val = self.travel_times_tensor[from_idx, to_idx].item()
                return time_val if not torch.isinf(torch.tensor(time_val)) else float('inf')
            
            return float('inf')
        except Exception:
            return float('inf')
    
    def _calculate_workflow_penalty_gpu(self, workflow: List[str], state: LayoutState) -> float:
        """GPU-accelerated workflow penalty calculation"""
        total_time = 0.0
        
        for i in range(len(workflow) - 1):
            from_function = workflow[i]
            to_function = workflow[i + 1]
            
            from_spaces = state.function_to_spaces.get(from_function, [])
            to_spaces = state.function_to_spaces.get(to_function, [])
            
            if not from_spaces or not to_spaces:
                total_time += 10000
                continue
            
            min_time = float('inf')
            for from_space in from_spaces:
                for to_space in to_spaces:
                    travel_time = self._get_travel_time_gpu(from_space, to_space)
                    if travel_time < min_time:
                        min_time = travel_time
            
            if min_time == float('inf'):
                min_time = 10000
            
            total_time += min_time
        
        return total_time


class DQNAgent:
    """Deep Q-Network agent with GPU acceleration"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 32, device: str = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = []
        self.memory_idx = 0
        
        self.update_target_network()
        
        logger.info(f"DQN Agent initialized on device: {self.device}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.memory_idx] = experience
            self.memory_idx = (self.memory_idx + 1) % self.memory_size
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy policy"""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[0, i] = float('-inf')
        
        return masked_q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate
        }
        
        torch.save(model_data, filepath)
        logger.info(f"GPU model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(model_data['q_network_state_dict'])
            self.target_network.load_state_dict(model_data['target_network_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            logger.info(f"GPU model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load GPU model: {e}")


class GPULayoutOptimizer:
    """
    GPU-accelerated layout optimizer using Deep Q-Network.
    """
    
    def __init__(self, csv_filepath: str, workflow_patterns: List[List[str]] = None, device: str = None):
        """
        Initialize the GPU layout optimizer.
        
        Args:
            csv_filepath: Path to super_network_travel_times.csv
            workflow_patterns: Common workflow patterns to optimize for
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.environment = GPULayoutEnvironment(csv_filepath, workflow_patterns, device)
        
        self.state_size = len(self.environment.functions) + len(self.environment.spaces)
        self.action_size = len(self.environment.functions) * len(self.environment.spaces) * 2  # assign/unassign
        
        self.agent = DQNAgent(self.state_size, self.action_size, device=device)
        self.training_history = []
        
        self._create_encodings()
        
        logger.info(f"GPU Layout Optimizer initialized with state_size={self.state_size}, action_size={self.action_size}")
    
    def _create_encodings(self):
        """Create encodings for states and actions"""
        self.function_to_idx = {func: idx for idx, func in enumerate(self.environment.functions)}
        self.space_to_idx = {space: idx for idx, space in enumerate(self.environment.spaces)}
        
        self.action_to_tuple = []
        for func_idx, function in enumerate(self.environment.functions):
            for space_idx, space in enumerate(self.environment.spaces):
                self.action_to_tuple.append((function, space, 'assign'))
                self.action_to_tuple.append((function, space, 'unassign'))
    
    def _encode_state(self, state: LayoutState) -> np.ndarray:
        """Encode layout state as a vector"""
        state_vector = np.zeros(self.state_size)
        
        for function, spaces in state.function_to_spaces.items():
            func_idx = self.function_to_idx.get(function)
            if func_idx is not None:
                state_vector[func_idx] = len(spaces)  # Number of spaces assigned
        
        offset = len(self.environment.functions)
        for space, function in state.space_to_function.items():
            space_idx = self.space_to_idx.get(space)
            if space_idx is not None:
                state_vector[offset + space_idx] = 1.0  # Space is occupied
        
        return state_vector
    
    def _get_valid_action_indices(self, state: LayoutState) -> List[int]:
        """Get valid action indices for the current state"""
        valid_actions = []
        
        for action_idx, (function, space, action_type) in enumerate(self.action_to_tuple):
            if action_type == 'assign':
                if space in state.unassigned_spaces:
                    valid_actions.append(action_idx)
            else:  # unassign
                if space in state.function_to_spaces.get(function, []):
                    valid_actions.append(action_idx)
        
        return valid_actions
    
    def _action_index_to_layout_action(self, action_idx: int) -> LayoutAction:
        """Convert action index to LayoutAction"""
        function, space, action_type = self.action_to_tuple[action_idx]
        return LayoutAction(function, space, action_type)
    
    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 100, 
              target_update_freq: int = 100) -> Dict[str, Any]:
        """
        Train the DQN agent to optimize layout.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            target_update_freq: Frequency to update target network
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting GPU training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            state_vector = self._encode_state(state)
            episode_reward = 0.0
            step_count = 0
            
            for step in range(max_steps_per_episode):
                valid_action_indices = self._get_valid_action_indices(state)
                if not valid_action_indices:
                    break
                
                action_idx = self.agent.select_action(state_vector, valid_action_indices)
                layout_action = self._action_index_to_layout_action(action_idx)
                
                next_state, reward, done = self.environment.step(layout_action)
                next_state_vector = self._encode_state(next_state)
                
                self.agent.remember(state_vector, action_idx, reward, next_state_vector, done)
                
                loss = self.agent.replay()
                if loss is not None:
                    losses.append(loss)
                
                episode_reward += reward
                state = next_state
                state_vector = next_state_vector
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                          f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.agent.epsilon:.3f}")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'losses': losses,
            'final_epsilon': self.agent.epsilon
        }
        
        logger.info("GPU training completed")
        return self.training_history
    
    def optimize_layout(self, max_iterations: int = 1000) -> Tuple[LayoutState, float]:
        """
        Find the optimal layout using the trained DQN agent.
        
        Args:
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (best_state, best_reward)
        """
        logger.info("Starting GPU layout optimization")
        
        best_state = None
        best_reward = float('-inf')
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # No exploration during optimization
        
        try:
            for iteration in range(max_iterations):
                state = self.environment.reset()
                state_vector = self._encode_state(state)
                current_reward = self.environment._calculate_reward(state)
                
                for step in range(100):  # Max steps per optimization run
                    valid_action_indices = self._get_valid_action_indices(state)
                    if not valid_action_indices:
                        break
                    
                    action_idx = self.agent.select_action(state_vector, valid_action_indices)
                    layout_action = self._action_index_to_layout_action(action_idx)
                    
                    next_state, reward, done = self.environment.step(layout_action)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_state = next_state.copy()
                    
                    state = next_state
                    state_vector = self._encode_state(next_state)
                    
                    if done:
                        break
                
                if iteration % 100 == 0:
                    logger.info(f"GPU optimization iteration {iteration}, Best reward: {best_reward:.2f}")
        
        finally:
            self.agent.epsilon = original_epsilon
        
        logger.info(f"GPU optimization completed. Best reward: {best_reward:.2f}")
        return best_state, best_reward
    
    def evaluate_current_layout(self) -> Dict[str, Any]:
        """Evaluate the current layout from the CSV data"""
        current_state = self.environment.reset()
        current_reward = self.environment._calculate_reward(current_state)
        
        evaluation = {
            'current_reward': current_reward,
            'function_assignments': dict(current_state.function_to_spaces),
            'workflow_penalties': {},
            'device_used': str(self.environment.device)
        }
        
        for i, workflow in enumerate(self.environment.workflow_patterns):
            penalty = self.environment._calculate_workflow_penalty_gpu(workflow, current_state)
            evaluation['workflow_penalties'][f'workflow_{i}'] = {
                'pattern': workflow,
                'penalty': penalty
            }
        
        return evaluation
    
    def save_model(self, filepath: str):
        """Save the trained DQN model"""
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained DQN model"""
        self.agent.load_model(filepath)
    
    def export_optimized_layout(self, state: LayoutState, filepath: str):
        """Export optimized layout to JSON file"""
        layout_data = {
            'function_to_spaces': state.function_to_spaces,
            'space_to_function': state.space_to_function,
            'unassigned_spaces': list(state.unassigned_spaces),
            'optimization_timestamp': pd.Timestamp.now().isoformat(),
            'device_used': str(self.environment.device),
            'model_type': 'DQN_GPU'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GPU optimized layout exported to {filepath}")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return system information"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_info': None
    }
    
    if torch.cuda.is_available():
        gpu_info['current_device'] = torch.cuda.current_device()
        gpu_info['device_name'] = torch.cuda.get_device_name()
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        gpu_info['memory_info'] = {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'total_gb': memory_total
        }
    
    return gpu_info


if __name__ == "__main__":
    gpu_info = check_gpu_availability()
    print("GPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    if gpu_info['cuda_available']:
        csv_path = "result/super_network_travel_times.csv"
        workflow_patterns = create_default_workflow_patterns()
        
        optimizer = GPULayoutOptimizer(csv_path, workflow_patterns)
        
        current_eval = optimizer.evaluate_current_layout()
        print(f"\nCurrent layout reward: {current_eval['current_reward']:.2f}")
        print(f"Using device: {current_eval['device_used']}")
        
        training_stats = optimizer.train(num_episodes=100)
        
        best_state, best_reward = optimizer.optimize_layout(max_iterations=50)
        
        print(f"GPU optimized layout reward: {best_reward:.2f}")
        print(f"Improvement: {best_reward - current_eval['current_reward']:.2f}")
        
        optimizer.save_model("result/rl_layout_model_gpu.pth")
        optimizer.export_optimized_layout(best_state, "result/optimized_layout_gpu.json")
    else:
        print("\nCUDA not available. Please use the CPU version (rl_layout_optimizer.py)")
