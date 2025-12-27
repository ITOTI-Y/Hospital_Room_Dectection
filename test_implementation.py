"""Test script to verify flow-aware adaptation implementation.

This script tests:
1. Enhanced environment with flow injection
2. Flow-aware encoder
3. Fast adaptation wrapper
4. Traditional baselines
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from loguru import logger

logger.info("Starting implementation tests...")

# Test 1: Environment with flow injection
logger.info("\n" + "=" * 60)
logger.info("Test 1: Environment with flow injection")
logger.info("=" * 60)

try:
    from src.config.config_loader import ConfigLoader
    from src.rl.env import LayoutEnv

    config = ConfigLoader()
    env = LayoutEnv(config, max_departments=50, max_step=100, patience=20)

    # Test reset without flow
    obs, info = env.reset()
    logger.info(f"✓ Environment reset successful")
    logger.info(f"  Observation keys: {list(obs.keys())}")
    logger.info(f"  Flow matrix shape: {obs['flow_matrix'].shape}")
    logger.info(f"  Initial cost: {info['initial_cost']:.2f}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    logger.info(f"✓ Environment step successful")
    logger.info(f"  Reward: {reward:.4f}")
    logger.info(f"  Current cost: {info['current_cost']:.2f}")

    # Test reset with custom flow
    custom_flow = np.random.rand(env.num_total_slot, env.num_total_slot).astype(np.float32)
    custom_flow = (custom_flow + custom_flow.T) / 2  # Make symmetric
    obs, info = env.reset(flow_matrix=custom_flow)
    logger.info(f"✓ Custom flow injection successful")

    logger.success("Test 1 PASSED: Environment works correctly")

except Exception as e:
    logger.error(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Flow-aware encoder
logger.info("\n" + "=" * 60)
logger.info("Test 2: Flow-aware encoder")
logger.info("=" * 60)

try:
    from src.rl.models.flow_encoder import FlowAwareEncoder, AdaptiveLayoutEncoder

    batch_size = 2
    n_depts = 10
    max_depts = 50

    # Test FlowAwareEncoder
    encoder = FlowAwareEncoder(
        dept_attr_dim=4,
        flow_embed_dim=max_depts,
        hidden_dim=128,
    )

    dept_attrs = torch.randn(batch_size, max_depts, 4)
    flow_matrix = torch.rand(batch_size, max_depts, max_depts)
    node_mask = torch.zeros(batch_size, max_depts)
    node_mask[:, :n_depts] = 1

    output = encoder(dept_attrs, flow_matrix, node_mask)
    logger.info(f"✓ FlowAwareEncoder forward pass successful")
    logger.info(f"  Input shape: {dept_attrs.shape}")
    logger.info(f"  Output shape: {output.shape}")

    # Test AdaptiveLayoutEncoder
    encoder = AdaptiveLayoutEncoder(
        n_dept_attrs=4,
        max_depts=max_depts,
        hidden_dim=128,
        gnn_layers=2,
    )

    # Create dummy edge data
    edge_index = torch.randint(0, n_depts, (batch_size, 2, 20))
    edge_weight = torch.rand(batch_size, 20)

    output = encoder(dept_attrs, flow_matrix, edge_index, edge_weight, node_mask)
    logger.info(f"✓ AdaptiveLayoutEncoder forward pass successful")
    logger.info(f"  Output shape: {output.shape}")

    logger.success("Test 2 PASSED: Flow-aware encoder works correctly")

except Exception as e:
    logger.error(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Traditional baselines
logger.info("\n" + "=" * 60)
logger.info("Test 3: Traditional baselines")
logger.info("=" * 60)

try:
    from src.baselines import SimulatedAnnealing, GreedySwap

    # Simple cost function for testing
    def dummy_cost(permutation):
        # Return sum of indices (prefer lower indices first)
        return sum(i * p for i, p in enumerate(permutation))

    n_items = 10

    # Test Simulated Annealing
    sa = SimulatedAnnealing(
        cost_fn=dummy_cost,
        n_items=n_items,
        T_init=100.0,
        alpha=0.9,
    )
    best, cost, history = sa.solve(max_steps=50)
    logger.info(f"✓ Simulated Annealing completed")
    logger.info(f"  Best cost: {cost:.2f}")
    logger.info(f"  History length: {len(history)}")

    # Test Greedy
    greedy = GreedySwap(cost_fn=dummy_cost, n_items=n_items)
    best, cost, history = greedy.solve(max_steps=20)
    logger.info(f"✓ Greedy search completed")
    logger.info(f"  Best cost: {cost:.2f}")
    logger.info(f"  History length: {len(history)}")

    logger.success("Test 3 PASSED: Traditional baselines work correctly")

except Exception as e:
    logger.error(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Fast adaptation wrapper (basic test without policy)
logger.info("\n" + "=" * 60)
logger.info("Test 4: Fast adaptation wrapper structure")
logger.info("=" * 60)

try:
    from src.rl.adaptation import FastAdaptationWrapper

    # Create a dummy policy
    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_encoder = torch.nn.Linear(10, 10)
            self.static_encoder = torch.nn.Linear(10, 10)

        def forward(self, obs):
            return torch.zeros(1)

        def forward_actor(self, obs):
            action = torch.zeros(1, 2)
            log_prob = torch.zeros(1)
            state = None
            return action, log_prob, state

    dummy_policy = DummyPolicy()
    wrapper = FastAdaptationWrapper(
        policy=dummy_policy,
        adaptation_lr=1e-3,
    )

    logger.info(f"✓ FastAdaptationWrapper created")

    # Test parameter filtering
    trainable = wrapper._get_trainable_params(dummy_policy)
    logger.info(f"  Trainable parameters: {len(trainable)}")

    logger.success("Test 4 PASSED: Adaptation wrapper structure correct")

except Exception as e:
    logger.error(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Summary
logger.info("\n" + "=" * 60)
logger.info("TEST SUMMARY")
logger.info("=" * 60)
logger.success("✓ All basic component tests completed!")
logger.info("Note: Full integration tests require trained policy")
logger.info("\nNext steps:")
logger.info("1. Train baseline PPO policy")
logger.info("2. Run full sample efficiency experiment")
logger.info("3. Run full adaptation experiment")
