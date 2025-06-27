"""
Reinforcement Learning-based Layout Optimization Module

This module provides a decoupled reinforcement learning approach to optimize
hospital layout assignments based on travel time data from super_network_travel_times.csv.

The RL agent learns to assign functions to physical spaces to minimize total travel time
while considering that single functions may have multiple physical spaces.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from collections import defaultdict, deque
import json
import pathlib

logger = logging.getLogger(__name__)


@dataclass
class LayoutState:
    """Represents the current state of function-to-space assignments"""
    function_to_spaces: Dict[str, List[str]]  # function -> list of assigned space IDs
    space_to_function: Dict[str, str]  # space ID -> assigned function
    unassigned_spaces: Set[str]  # available space IDs
    
    def copy(self) -> 'LayoutState':
        """Create a deep copy of the layout state"""
        return LayoutState(
            function_to_spaces={k: v.copy() for k, v in self.function_to_spaces.items()},
            space_to_function=self.space_to_function.copy(),
            unassigned_spaces=self.unassigned_spaces.copy()
        )


@dataclass
class LayoutAction:
    """Represents an action to assign/reassign a function to a space"""
    function: str
    space_id: str
    action_type: str  # 'assign', 'reassign', 'remove'


class LayoutEnvironment:
    """
    Environment for the layout optimization problem.
    
    Manages the state space, action space, and reward calculation based on
    travel time data from the CSV file.
    """
    
    def __init__(self, csv_filepath: str, workflow_patterns: List[List[str]] = None):
        """
        Initialize the layout environment.
        
        Args:
            csv_filepath: Path to the super_network_travel_times.csv file
            workflow_patterns: List of common workflow patterns (sequences of functions)
        """
        self.csv_filepath = csv_filepath
        self.workflow_patterns = workflow_patterns or []
        
        self.travel_times_df = None
        self.functions = set()
        self.spaces = set()
        self.function_to_possible_spaces = defaultdict(list)
        self.space_to_function_type = {}
        
        self._load_travel_data()
        self._parse_functions_and_spaces()
        
        self.current_state = None
        self.reset()
        
    def _load_travel_data(self):
        """Load travel time data from CSV"""
        try:
            self.travel_times_df = pd.read_csv(self.csv_filepath, index_col=0)
            logger.info(f"Loaded travel times data with shape: {self.travel_times_df.shape}")
        except Exception as e:
            logger.error(f"Failed to load travel times data: {e}")
            raise
    
    def _parse_functions_and_spaces(self):
        """Parse functions and spaces from the travel times data"""
        all_locations = list(self.travel_times_df.columns)
        
        for location in all_locations:
            if '_' in location:
                parts = location.split('_', 1)
                function_name = parts[0]
                space_id = parts[1]
                
                self.functions.add(function_name)
                self.spaces.add(location)
                self.function_to_possible_spaces[function_name].append(location)
                self.space_to_function_type[location] = function_name
        
        logger.info(f"Parsed {len(self.functions)} unique functions and {len(self.spaces)} spaces")
        logger.info(f"Functions: {sorted(self.functions)}")
    
    def reset(self) -> LayoutState:
        """Reset the environment to initial state"""
        function_to_spaces = {}
        space_to_function = {}
        
        for function in self.functions:
            function_to_spaces[function] = self.function_to_possible_spaces[function].copy()
            for space in self.function_to_possible_spaces[function]:
                space_to_function[space] = function
        
        self.current_state = LayoutState(
            function_to_spaces=function_to_spaces,
            space_to_function=space_to_function,
            unassigned_spaces=set()
        )
        
        return self.current_state
    
    def get_valid_actions(self, state: LayoutState) -> List[LayoutAction]:
        """Get all valid actions from the current state"""
        actions = []
        
        for function in self.functions:
            current_spaces = state.function_to_spaces.get(function, [])
            possible_spaces = self.function_to_possible_spaces[function]
            
            for space in possible_spaces:
                if space not in current_spaces:
                    actions.append(LayoutAction(function, space, 'assign'))
            
            if len(current_spaces) > 1:
                for space in current_spaces:
                    actions.append(LayoutAction(function, space, 'remove'))
        
        return actions
    
    def step(self, action: LayoutAction) -> Tuple[LayoutState, float, bool]:
        """
        Execute an action and return new state, reward, and done flag.
        
        Returns:
            new_state: The resulting state after action
            reward: Reward for this action
            done: Whether the episode is complete
        """
        new_state = self.current_state.copy()
        
        if action.action_type == 'assign':
            if action.space_id not in new_state.function_to_spaces[action.function]:
                new_state.function_to_spaces[action.function].append(action.space_id)
                new_state.space_to_function[action.space_id] = action.function
                new_state.unassigned_spaces.discard(action.space_id)
        
        elif action.action_type == 'remove':
            if action.space_id in new_state.function_to_spaces[action.function]:
                new_state.function_to_spaces[action.function].remove(action.space_id)
                if action.space_id in new_state.space_to_function:
                    del new_state.space_to_function[action.space_id]
                new_state.unassigned_spaces.add(action.space_id)
        
        reward = self._calculate_reward(new_state)
        
        self.current_state = new_state
        done = False  # Can define termination conditions if needed
        
        return new_state, reward, done
    
    def _calculate_reward(self, state: LayoutState) -> float:
        """
        Calculate reward based on the current layout state.
        
        Reward is based on minimizing total travel time for workflow patterns.
        """
        if not self.workflow_patterns:
            return 0.0
        
        total_penalty = 0.0
        
        for workflow in self.workflow_patterns:
            workflow_penalty = self._calculate_workflow_penalty(workflow, state)
            total_penalty += workflow_penalty
        
        return -total_penalty
    
    def _calculate_workflow_penalty(self, workflow: List[str], state: LayoutState) -> float:
        """Calculate penalty for a specific workflow pattern"""
        if len(workflow) < 2:
            return 0.0
        
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
                    travel_time = self._get_travel_time(from_space, to_space)
                    if travel_time < min_time:
                        min_time = travel_time
            
            if min_time == float('inf'):
                min_time = 10000  # Large penalty for unreachable
            
            total_time += min_time
        
        return total_time
    
    def _get_travel_time(self, from_space: str, to_space: str) -> float:
        """Get travel time between two spaces from the CSV data"""
        try:
            if from_space in self.travel_times_df.index and to_space in self.travel_times_df.columns:
                time_val = self.travel_times_df.loc[from_space, to_space]
                if isinstance(time_val, (int, float)):
                    return float(time_val)
                elif isinstance(time_val, str) and time_val != '∞':
                    return float(time_val)
            return float('inf')
        except (KeyError, ValueError):
            return float('inf')


class RLAgent(ABC):
    """Abstract base class for RL agents"""
    
    @abstractmethod
    def select_action(self, state: LayoutState, valid_actions: List[LayoutAction]) -> LayoutAction:
        """Select an action given the current state and valid actions"""
        pass
    
    @abstractmethod
    def update(self, state: LayoutState, action: LayoutAction, reward: float, next_state: LayoutState):
        """Update the agent's policy based on experience"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save the trained model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load a trained model"""
        pass


class QLearningAgent(RLAgent):
    """Q-Learning agent for layout optimization"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def _state_to_hash(self, state: LayoutState) -> str:
        """Convert state to a hashable representation"""
        assignments = []
        for function in sorted(state.function_to_spaces.keys()):
            spaces = sorted(state.function_to_spaces[function])
            assignments.append(f"{function}:{','.join(spaces)}")
        return "|".join(assignments)
    
    def _action_to_hash(self, action: LayoutAction) -> str:
        """Convert action to a hashable representation"""
        return f"{action.function}_{action.space_id}_{action.action_type}"
    
    def select_action(self, state: LayoutState, valid_actions: List[LayoutAction]) -> LayoutAction:
        """Select action using epsilon-greedy policy"""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_hash = self._state_to_hash(state)
            best_action = None
            best_q_value = float('-inf')
            
            for action in valid_actions:
                action_hash = self._action_to_hash(action)
                q_value = self.q_table[state_hash][action_hash]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action else random.choice(valid_actions)
    
    def update(self, state: LayoutState, action: LayoutAction, reward: float, next_state: LayoutState):
        """Update Q-table using Q-learning update rule"""
        state_hash = self._state_to_hash(state)
        action_hash = self._action_to_hash(action)
        next_state_hash = self._state_to_hash(next_state)
        
        max_next_q = 0.0
        if next_state_hash in self.q_table:
            max_next_q = max(self.q_table[next_state_hash].values()) if self.q_table[next_state_hash] else 0.0
        
        current_q = self.q_table[state_hash][action_hash]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_hash][action_hash] = new_q
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save Q-table to file"""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state_hash, actions in model_data['q_table'].items():
                for action_hash, q_value in actions.items():
                    self.q_table[state_hash][action_hash] = q_value
            
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class LayoutOptimizer:
    """
    Main class for running reinforcement learning-based layout optimization.
    """
    
    def __init__(self, csv_filepath: str, workflow_patterns: List[List[str]] = None):
        """
        Initialize the layout optimizer.
        
        Args:
            csv_filepath: Path to super_network_travel_times.csv
            workflow_patterns: Common workflow patterns to optimize for
        """
        self.environment = LayoutEnvironment(csv_filepath, workflow_patterns)
        self.agent = QLearningAgent()
        self.training_history = []
        
    def add_workflow_pattern(self, workflow: List[str]):
        """Add a workflow pattern to optimize for"""
        self.environment.workflow_patterns.append(workflow)
    
    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Train the RL agent to optimize layout.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            step_count = 0
            
            for step in range(max_steps_per_episode):
                valid_actions = self.environment.get_valid_actions(state)
                if not valid_actions:
                    break
                
                action = self.agent.select_action(state, valid_actions)
                next_state, reward, done = self.environment.step(action)
                
                self.agent.update(state, action, reward, next_state)
                
                episode_reward += reward
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_epsilon': self.agent.epsilon
        }
        
        logger.info("Training completed")
        return self.training_history
    
    def optimize_layout(self, max_iterations: int = 1000) -> Tuple[LayoutState, float]:
        """
        Find the optimal layout using the trained agent.
        
        Args:
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (best_state, best_reward)
        """
        logger.info("Starting layout optimization")
        
        best_state = None
        best_reward = float('-inf')
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        try:
            for iteration in range(max_iterations):
                state = self.environment.reset()
                current_reward = self.environment._calculate_reward(state)
                
                for step in range(100):  # Max steps per optimization run
                    valid_actions = self.environment.get_valid_actions(state)
                    if not valid_actions:
                        break
                    
                    action = self.agent.select_action(state, valid_actions)
                    next_state, reward, done = self.environment.step(action)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_state = next_state.copy()
                    
                    state = next_state
                    
                    if done:
                        break
                
                if iteration % 100 == 0:
                    logger.info(f"Optimization iteration {iteration}, Best reward: {best_reward:.2f}")
        
        finally:
            self.agent.epsilon = original_epsilon
        
        logger.info(f"Optimization completed. Best reward: {best_reward:.2f}")
        return best_state, best_reward
    
    def evaluate_current_layout(self) -> Dict[str, Any]:
        """Evaluate the current layout from the CSV data"""
        current_state = self.environment.reset()
        current_reward = self.environment._calculate_reward(current_state)
        
        evaluation = {
            'current_reward': current_reward,
            'function_assignments': dict(current_state.function_to_spaces),
            'workflow_penalties': {}
        }
        
        for i, workflow in enumerate(self.environment.workflow_patterns):
            penalty = self.environment._calculate_workflow_penalty(workflow, current_state)
            evaluation['workflow_penalties'][f'workflow_{i}'] = {
                'pattern': workflow,
                'penalty': penalty
            }
        
        return evaluation
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.agent.load_model(filepath)
    
    def export_optimized_layout(self, state: LayoutState, filepath: str):
        """Export optimized layout to JSON file"""
        layout_data = {
            'function_to_spaces': state.function_to_spaces,
            'space_to_function': state.space_to_function,
            'unassigned_spaces': list(state.unassigned_spaces),
            'optimization_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimized layout exported to {filepath}")


def create_default_workflow_patterns() -> List[List[str]]:
    """Create default workflow patterns for hospital operations"""
    return [
        ['门', '挂号收费', '妇科', '门'],  # Gynecology visit
        ['门', '挂号收费', '采血处', '检验中心', '门'],  # Blood test
        ['门', '挂号收费', '超声科', '门'],  # Ultrasound
        ['门', '挂号收费', '内科', '内诊药房', '门'],  # Internal medicine visit
        ['门', '挂号收费', '儿科', '门'],  # Pediatrics visit
        ['门', '急诊科', '门'],  # Emergency visit
        ['门', '挂号收费', '眼科', '门'],  # Ophthalmology visit
        ['门', '挂号收费', '放射科', '门'],  # Radiology
    ]


if __name__ == "__main__":
    csv_path = "result/super_network_travel_times.csv"
    workflow_patterns = create_default_workflow_patterns()
    
    optimizer = LayoutOptimizer(csv_path, workflow_patterns)
    
    current_eval = optimizer.evaluate_current_layout()
    print(f"Current layout reward: {current_eval['current_reward']:.2f}")
    
    training_stats = optimizer.train(num_episodes=500)
    
    best_state, best_reward = optimizer.optimize_layout()
    
    print(f"Optimized layout reward: {best_reward:.2f}")
    print(f"Improvement: {best_reward - current_eval['current_reward']:.2f}")
    
    optimizer.save_model("result/rl_layout_model.json")
    optimizer.export_optimized_layout(best_state, "result/optimized_layout.json")
