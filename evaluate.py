"""
evaluate.py — Baseline vs RL Comparison + Plotting
====================================================
Compares the trained RL policy against a fixed-policy baseline
(random resource deployment) on the same wildfire simulator.

Usage:
  python evaluate.py --config configs/qlearning_v1.yaml --policy models/policy_exp-qlearning-1_final.pkl

Outputs:
  - Comparison table printed to console
  - results/comparison_<experiment>.csv
  - results/reward_curve.png        (average reward over episodes)
  - results/burned_area_curve.png   (burned cells over episodes)
"""

import argparse
import os
import sys
import csv
import yaml
import numpy as np
import json
import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_baseline(env_cfg, n_episodes=100):
    """Random deployment baseline — deploy resources to random cells each step."""
    rewards = []
    burned_list = []

    for _ in range(n_episodes):
        env = WildfireEnv(**env_cfg)
        state = env.reset()
        total_reward = 0
        while True:
            action = np.random.randint(0, env.action_size)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        burned_list.append(env.total_burned)

    return np.mean(rewards), np.mean(burned_list), rewards, burned_list


def run_rl_policy(env_cfg, agent_cfg, policy_path, n_episodes=100):
    """Run the trained RL policy (greedy, epsilon=0)."""
    env = WildfireEnv(**env_cfg)
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        **agent_cfg
    )
    agent.load(policy_path)
    agent.epsilon = 0.0  # Pure exploitation

    rewards = []
    burned_list = []

    for _ in range(n_episodes):
        env_eval = WildfireEnv(**env_cfg)
        state = env_eval.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            state, reward, done, info = env_eval.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        burned_list.append(env_eval.total_burned)

    return np.mean(rewards), np.mean(burned_list), rewards, burned_list


def plot_training_curves(exp_name):
    """Plot reward and burned-area curves from training CSV."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots.")
        return

    csv_path = f"results/results_{exp_name}.csv"
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found — skipping plots.")
        return

    episodes, rewards, burned = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            burned.append(int(row["total_burned"]))

    # Smoothed curves (moving average, window=20)
    window = 20

    def smooth(data, w):
        return np.convolve(data, np.ones(w) / w, mode='valid')

    # --- Plot 1: Average reward over episodes ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue', label='Per-episode')
    if len(rewards) >= window:
        sm = smooth(rewards, window)
        ax.plot(episodes[window - 1:], sm, color='darkblue', linewidth=2,
                label=f'{window}-episode moving avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(f'Average Reward Over Episodes ({exp_name})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    reward_path = f'results/reward_curve_{exp_name}.png'
    fig.savefig(reward_path, dpi=150)
    print(f"[PLOT] Saved {reward_path}")
    plt.close()

    # --- Plot 2: Burned area over episodes ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, burned, alpha=0.3, color='orangered', label='Per-episode')
    if len(burned) >= window:
        sm = smooth(burned, window)
        ax.plot(episodes[window - 1:], sm, color='darkred', linewidth=2,
                label=f'{window}-episode moving avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Cells Burned', fontsize=12)
    ax.set_title(f'Burned Area Over Episodes ({exp_name})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    burned_path = f'results/burned_area_curve_{exp_name}.png'
    fig.savefig(burned_path, dpi=150)
    print(f"[PLOT] Saved {burned_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL policy vs random baseline")
    parser.add_argument("--config", type=str,
                        default="configs/qlearning_v1.yaml")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained policy .pkl file")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config["experiment_name"]
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    policy_dir = config["training"].get("policy_dir", "models/")

    # Auto-detect policy if not specified
    if args.policy is None:
        args.policy = os.path.join(policy_dir, f"policy_{exp_name}_final.pkl")

    if not os.path.exists(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        print("        Run train.py first to generate a trained policy.")
        sys.exit(1)

    print("=" * 60)
    print("  WILDFIRE CONTAINMENT — EVALUATION")
    print(f"  Experiment: {exp_name}")
    print("=" * 60)

    n = args.eval_episodes
    print(f"\nRunning {n} episodes each for Baseline and RL...\n")

    # --- Baseline ---
    base_reward, base_burned, base_r_list, base_b_list = run_baseline(env_cfg, n)

    # --- RL Policy ---
    rl_reward, rl_burned, rl_r_list, rl_b_list = run_rl_policy(
        env_cfg, agent_cfg, args.policy, n)

    # --- Comparison Table ---
    improvement_burned = ((base_burned - rl_burned) / base_burned) * 100 if base_burned > 0 else 0

    print("\n" + "=" * 60)
    print("  BASELINE vs RL COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'Random Baseline':>16s} {'RL Policy':>16s}")
    print(f"  {'-'*25} {'-'*16} {'-'*16}")
    print(f"  {'Avg Reward':<25s} {base_reward:>16.2f} {rl_reward:>16.2f}")
    print(f"  {'Avg Burned Cells':<25s} {base_burned:>16.1f} {rl_burned:>16.1f}")
    print(f"  {'Burned Reduction':<25s} {'—':>16s} {improvement_burned:>15.1f}%")
    print("=" * 60)

    # --- Save comparison CSV ---
    os.makedirs("results", exist_ok=True)
    comp_path = f"results/comparison_{exp_name}.csv"
    with open(comp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "random_baseline", "rl_policy"])
        writer.writerow(["avg_reward", round(base_reward, 2), round(rl_reward, 2)])
        writer.writerow(["avg_burned_cells", round(base_burned, 1), round(rl_burned, 1)])
        writer.writerow(["burned_reduction_pct", "—", f"{improvement_burned:.1f}%"])
    print(f"\n[RESULTS] Comparison saved to {comp_path}")

    # --- Plot training curves ---
    plot_training_curves(exp_name)

    # --- SDG Impact ---
    print("\n" + "=" * 60)
    print("  SDG IMPACT ANALYSIS")
    print("=" * 60)
    print(f"  Reducing average burned area by {improvement_burned:.1f}% supports:")
    print(f"    • SDG 13 (Climate Action) — fewer CO2 emissions from wildfires")
    print(f"    • SDG 15 (Life on Land)   — preserved forest ecosystems & biodiversity")
    print(f"  Optimised resource deployment saves firefighting costs and lives.")
    print("=" * 60)

    # --- Save evaluation summary JSON ---
    eval_summary = {
        "experiment": exp_name,
        "eval_episodes": n,
        "baseline_avg_reward": round(base_reward, 2),
        "baseline_avg_burned": round(base_burned, 1),
        "rl_avg_reward": round(rl_reward, 2),
        "rl_avg_burned": round(rl_burned, 1),
        "burned_reduction_pct": round(improvement_burned, 1),
        "sdg_alignment": ["SDG 13 (Climate Action)", "SDG 15 (Life on Land)"],
    }
    eval_path = f"results/evaluation_{exp_name}.json"
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[MLOPS] Evaluation summary saved to {eval_path}")

    # --- Log to MLflow ---
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"eval_{exp_name}"):
        mlflow.log_metrics({
            "baseline_avg_reward": float(base_reward),
            "baseline_avg_burned": float(base_burned),
            "rl_avg_reward": float(rl_reward),
            "rl_avg_burned": float(rl_burned),
            "burned_reduction_pct": float(improvement_burned)
        })
        mlflow.log_artifact(eval_path, artifact_path="evaluation")
        mlflow.log_artifact(comp_path, artifact_path="evaluation")
        # Log plot artifacts if they exist
        if os.path.exists(f'results/reward_curve_{exp_name}.png'):
            mlflow.log_artifact(f'results/reward_curve_{exp_name}.png', artifact_path="evaluation_plots")
        if os.path.exists(f'results/burned_area_curve_{exp_name}.png'):
            mlflow.log_artifact(f'results/burned_area_curve_{exp_name}.png', artifact_path="evaluation_plots")
    print(f"[MLOPS] Metrics and artifacts logged to MLflow Registry")


if __name__ == "__main__":
    main()
