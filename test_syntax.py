"""Quick syntax and import test without running code."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

# Test 1: Baseline imports
try:
    from src.baselines.traditional import SimulatedAnnealing, GreedySwap, RandomSearch
    print("✓ Baselines import successful")
except Exception as e:
    print(f"✗ Baselines import failed: {e}")

# Test 2: Flow encoder imports
try:
    from src.rl.models.flow_encoder import (
        FlowAwareEncoder,
        AdaptiveLayoutEncoder,
        FlowMatrixExtractor,
    )
    print("✓ Flow encoder import successful")
except Exception as e:
    print(f"✗ Flow encoder import failed: {e}")

# Test 3: Adaptation imports
try:
    from src.rl.adaptation import FastAdaptationWrapper
    print("✓ Adaptation wrapper import successful")
except Exception as e:
    print(f"✗ Adaptation wrapper import failed: {e}")

# Test 4: Experiment imports
try:
    from experiments.sample_efficiency import run_sample_efficiency_experiment
    from experiments.adaptation_experiment import run_adaptation_experiment
    print("✓ Experiment modules import successful")
except Exception as e:
    print(f"✗ Experiment modules import failed: {e}")

# Test 5: Check environment modifications
try:
    import inspect
    from src.rl.env import LayoutEnv

    # Check if new methods exist
    methods = [
        "_estimate_cost_scale",
        "_extract_flow_matrix",
        "_update_cost_engine_with_flow",
    ]

    for method in methods:
        if hasattr(LayoutEnv, method):
            print(f"✓ LayoutEnv.{method} exists")
        else:
            print(f"✗ LayoutEnv.{method} missing")

    # Check reset signature
    sig = inspect.signature(LayoutEnv.reset)
    params = list(sig.parameters.keys())
    if "flow_matrix" in params:
        print(f"✓ LayoutEnv.reset accepts flow_matrix parameter")
    else:
        print(f"✗ LayoutEnv.reset missing flow_matrix parameter")

except Exception as e:
    print(f"✗ Environment check failed: {e}")

print("\nAll import tests completed!")
