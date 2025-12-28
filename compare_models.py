"""Compare v2 and v3 model performance in detail."""
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tianshou.data import Batch

from src.config import config_loader
from src.rl.env import LayoutEnv
from src.rl.models.policy import LayoutA2CPolicy
from src.rl.models.ppo_model import LayoutOptimizationModel

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>", level="INFO")


def load_model(config, model_path: Path, device: torch.device):
    agent_cfg = config.agent
    model = LayoutOptimizationModel(
        num_categories=agent_cfg.max_departments,
        embedding_dim=agent_cfg.embedding_dim,
        numerical_feat_dim=agent_cfg.numerical_feat_dim,
        numerical_hidden_dim=agent_cfg.numerical_hidden_dim,
        gnn_hidden_dims=agent_cfg.gnn_hidden_dims,
        gnn_output_dim=agent_cfg.gnn_output_dim,
        gnn_num_layers=agent_cfg.gnn_num_layers,
        gnn_dropout=agent_cfg.gnn_dropout,
        actor_hidden_dim=agent_cfg.actor_hidden_dim,
        actor_dropout=agent_cfg.actor_dropout,
        value_hidden_dim=agent_cfg.value_hidden_dim,
        value_num_layers=agent_cfg.value_num_layers,
        value_pooling_type=agent_cfg.value_pooling_type,
        value_dropout=agent_cfg.value_dropout,
        device=device,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    env = LayoutEnv(config=config, max_departments=agent_cfg.max_departments,
                   max_step=agent_cfg.max_steps, is_training=False)
    policy = LayoutA2CPolicy(
        model=model, optim=optim, action_space=env.action_space,
        discount_factor=agent_cfg.discount_factor, gae_lambda=agent_cfg.gae_lambda,
        vf_coef=agent_cfg.vf_coef, ent_coef=agent_cfg.ent_coef,
        max_grad_norm=agent_cfg.max_grad_norm, value_clip=agent_cfg.value_clip,
        advantage_normalization=agent_cfg.advantage_normalization,
        recompute_advantage=agent_cfg.recompute_advantage,
        dual_clip=agent_cfg.dual_clip, reward_normalization=agent_cfg.reward_normalization,
        eps_clip=agent_cfg.eps_clip, max_batchsize=agent_cfg.max_batchsize,
        deterministic_eval=True,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_episode_detailed(config, policy, seed=None):
    """Run a single episode and return detailed metrics."""
    env = LayoutEnv(
        config=config,
        max_departments=config.agent.max_departments,
        max_step=config.agent.max_steps,
        is_training=False
    )
    obs, info = env.reset(seed=seed)
    initial_cost = info["initial_cost"]

    # Track metrics
    step_costs = [initial_cost]
    step_improvements = []
    actions_taken = []

    done = False
    while not done:
        obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        batch = Batch(obs=obs_batch)
        with torch.no_grad():
            result = policy(batch)
            action = result.act[0].cpu().numpy()

        actions_taken.append((int(action[0]), int(action[1])))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_costs.append(info["current_cost"])
        step_imp = (initial_cost - info["current_cost"]) / initial_cost * 100
        step_improvements.append(step_imp)

    final_cost = info["current_cost"]
    best_cost = info.get("best_cost", env.best_cost)

    return {
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "best_cost": best_cost,
        "improvement": (initial_cost - final_cost) / initial_cost * 100,
        "best_improvement": (initial_cost - best_cost) / initial_cost * 100,
        "total_steps": len(actions_taken),
        "best_step": info.get("best_step", env.best_step),
        "total_swaps": env.total_swaps,
        "invalid_swaps": env.invalid_swaps,
        "no_change_swaps": env.no_change_swaps,
        "unique_actions": len(set(env.action_history)),
        "step_costs": step_costs,
        "step_improvements": step_improvements,
        "actions": actions_taken,
    }


def analyze_action_patterns(results: dict):
    """Analyze action patterns from episode results."""
    actions = results["actions"]

    # Count repeated actions
    action_counts = {}
    for a in actions:
        key = frozenset(a)
        action_counts[key] = action_counts.get(key, 0) + 1

    repeated = sum(1 for c in action_counts.values() if c > 1)
    max_repeat = max(action_counts.values()) if action_counts else 0

    return {
        "unique_pairs": len(action_counts),
        "repeated_pairs": repeated,
        "max_repeat_count": max_repeat,
    }


def compare_models(config, v2_policy, v3_policy, num_cases=10):
    """Compare two models on the same test cases."""
    print("\n" + "=" * 80)
    print("DETAILED MODEL COMPARISON")
    print("=" * 80)

    v2_results = []
    v3_results = []

    # Use fixed seeds for fair comparison
    seeds = [42 + i for i in range(num_cases)]

    for i, seed in enumerate(seeds):
        print(f"\n--- Test Case #{i+1} (seed={seed}) ---")

        # Run v2 model
        r2 = run_episode_detailed(config, v2_policy, seed=seed)
        v2_results.append(r2)

        # Run v3 model with same seed
        r3 = run_episode_detailed(config, v3_policy, seed=seed)
        v3_results.append(r3)

        # Print comparison for this case
        print(f"  Initial cost: {r2['initial_cost']:.1f}")
        print(f"  V2: {r2['improvement']:6.2f}% (best: {r2['best_improvement']:.2f}% @ step {r2['best_step']}), "
              f"no-change: {r2['no_change_swaps']}, invalid: {r2['invalid_swaps']}")
        print(f"  V3: {r3['improvement']:6.2f}% (best: {r3['best_improvement']:.2f}% @ step {r3['best_step']}), "
              f"no-change: {r3['no_change_swaps']}, invalid: {r3['invalid_swaps']}")

        # Show improvement trajectory difference
        if r2['improvement'] != r3['improvement']:
            better = "V2" if r2['improvement'] > r3['improvement'] else "V3"
            diff = abs(r2['improvement'] - r3['improvement'])
            print(f"  Winner: {better} (+{diff:.2f}%)")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    metrics = ['improvement', 'best_improvement', 'no_change_swaps', 'invalid_swaps',
               'unique_actions', 'best_step', 'total_steps']

    print(f"\n{'Metric':<25} {'V2 (mean±std)':<20} {'V3 (mean±std)':<20} {'Diff':<10}")
    print("-" * 75)

    for m in metrics:
        v2_vals = [r[m] for r in v2_results]
        v3_vals = [r[m] for r in v3_results]
        v2_mean, v2_std = np.mean(v2_vals), np.std(v2_vals)
        v3_mean, v3_std = np.mean(v3_vals), np.std(v3_vals)
        diff = v3_mean - v2_mean

        print(f"{m:<25} {v2_mean:>7.2f} ± {v2_std:<7.2f}  {v3_mean:>7.2f} ± {v3_std:<7.2f}  {diff:>+7.2f}")

    # Win rate
    v2_wins = sum(1 for r2, r3 in zip(v2_results, v3_results) if r2['improvement'] > r3['improvement'])
    v3_wins = sum(1 for r2, r3 in zip(v2_results, v3_results) if r3['improvement'] > r2['improvement'])
    ties = num_cases - v2_wins - v3_wins

    print(f"\n{'Win Rate':<25} V2: {v2_wins}/{num_cases}  V3: {v3_wins}/{num_cases}  Ties: {ties}/{num_cases}")

    # Action pattern analysis
    print("\n" + "=" * 80)
    print("ACTION PATTERN ANALYSIS")
    print("=" * 80)

    for name, results in [("V2", v2_results), ("V3", v3_results)]:
        patterns = [analyze_action_patterns(r) for r in results]
        avg_unique = np.mean([p['unique_pairs'] for p in patterns])
        avg_repeated = np.mean([p['repeated_pairs'] for p in patterns])
        avg_max_repeat = np.mean([p['max_repeat_count'] for p in patterns])
        print(f"\n{name}:")
        print(f"  Avg unique action pairs: {avg_unique:.1f}")
        print(f"  Avg repeated pairs: {avg_repeated:.1f}")
        print(f"  Avg max repeat count: {avg_max_repeat:.1f}")

    # Improvement trajectory analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT TRAJECTORY (sample case)")
    print("=" * 80)

    # Take first case as sample
    r2, r3 = v2_results[0], v3_results[0]

    print(f"\nCase #1 improvement over steps:")
    print(f"{'Step':<6} {'V2 Imp%':<12} {'V3 Imp%':<12} {'V2 Cost':<12} {'V3 Cost':<12}")
    print("-" * 54)

    max_steps = max(len(r2['step_improvements']), len(r3['step_improvements']))
    for i in range(0, min(max_steps, 20)):  # Show first 20 steps
        v2_imp = r2['step_improvements'][i] if i < len(r2['step_improvements']) else r2['step_improvements'][-1]
        v3_imp = r3['step_improvements'][i] if i < len(r3['step_improvements']) else r3['step_improvements'][-1]
        v2_cost = r2['step_costs'][i+1] if i+1 < len(r2['step_costs']) else r2['step_costs'][-1]
        v3_cost = r3['step_costs'][i+1] if i+1 < len(r3['step_costs']) else r3['step_costs'][-1]
        print(f"{i+1:<6} {v2_imp:<12.2f} {v3_imp:<12.2f} {v2_cost:<12.1f} {v3_cost:<12.1f}")

    return v2_results, v3_results


def main():
    config = config_loader.ConfigLoader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model paths
    v2_model_path = Path("results/model/best_v3/best_before_v3_training.pth")  # V2 best
    v3_model_path = Path("results/model/best_ppo_layout_model.pth")  # V3 best

    print(f"\nV2 model: {v2_model_path}")
    print(f"V3 model: {v3_model_path}")

    if not v2_model_path.exists():
        print(f"ERROR: V2 model not found: {v2_model_path}")
        return
    if not v3_model_path.exists():
        print(f"ERROR: V3 model not found: {v3_model_path}")
        return

    # Load models
    print("\nLoading models...")
    v2_policy = load_model(config, v2_model_path, device)
    v3_policy = load_model(config, v3_model_path, device)

    # Compare
    v2_results, v3_results = compare_models(config, v2_policy, v3_policy, num_cases=10)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    v2_mean = np.mean([r['improvement'] for r in v2_results])
    v3_mean = np.mean([r['improvement'] for r in v3_results])

    if v2_mean > v3_mean:
        print(f"\nV2 performs better by {v2_mean - v3_mean:.2f}% on average.")
        print("Possible reasons:")
        print("- V3 early stopping cut training too short")
        print("- V3 evaluation metric (actual improvement) is noisy")
        print("- V2 had more epochs to explore better policies")
    else:
        print(f"\nV3 performs better by {v3_mean - v2_mean:.2f}% on average.")
        print("The new training approach is working well.")


if __name__ == "__main__":
    main()
