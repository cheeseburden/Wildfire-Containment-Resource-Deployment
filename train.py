"""
train.py — Main Training Script
=================================
Trains the Q-learning agent on the Wildfire Containment simulator.

Usage:
  python train.py --config configs/qlearning_v1.yaml

Features:
  - Loads config from YAML
  - Trains for N episodes with epsilon-greedy exploration
  - Logs per-episode metrics (reward, burned cells, epsilon)
  - Saves policy checkpoints at specified episodes
  - Writes experiment results to results/results_<experiment>.csv
  - Writes a log.json summary for MLOps tracking

Reproducibility:
  "Run python train.py --config configs/qlearning_v1.yaml to get the same result."
"""

import argparse
import os
import sys
import json
import csv
import yaml
import numpy as np
import time
import mlflow
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config):
    """Run the training loop."""
    exp_name = config["experiment_name"]
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]

    print("=" * 60)
    print(f"  WILDFIRE CONTAINMENT — RL TRAINING")
    print(f"  Experiment: {exp_name}")
    print(f"  Algorithm : Q-Learning (tabular, epsilon-greedy)")
    print("=" * 60)

    # --- Start MLflow Run ---
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    mlflow.log_params(env_cfg)
    mlflow.log_params(agent_cfg)
    mlflow.log_params(train_cfg)

    # --- Initialise environment ---
    env = WildfireEnv(**env_cfg)

    # --- Initialise agent ---
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=agent_cfg["learning_rate"],
        discount_factor=agent_cfg["discount_factor"],
        epsilon=agent_cfg["epsilon"],
        epsilon_min=agent_cfg["epsilon_min"],
        epsilon_decay=agent_cfg["epsilon_decay"],
    )

    episodes = train_cfg["episodes"]
    log_interval = train_cfg["log_interval"]
    save_at = train_cfg.get("save_policy_at", [])
    policy_dir = train_cfg.get("policy_dir", "models/")

    os.makedirs(policy_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Training metrics storage ---
    all_rewards = []
    all_burned = []
    all_epsilons = []
    episode_logs = []

    print(f"\nTraining for {episodes} episodes...\n")
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        step = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if done:
                break

        agent.decay_epsilon()

        total_burned = env.total_burned
        all_rewards.append(total_reward)
        all_burned.append(total_burned)
        all_epsilons.append(agent.epsilon)

        mlflow.log_metric("reward", total_reward, step=ep)
        mlflow.log_metric("total_burned", total_burned, step=ep)
        mlflow.log_metric("epsilon", agent.epsilon, step=ep)

        episode_logs.append({
            "episode": ep,
            "reward": round(total_reward, 2),
            "total_burned": int(total_burned),
            "epsilon": round(agent.epsilon, 4),
            "steps": step,
            "q_table_size": agent.get_q_table_size(),
        })

        # --- Periodic logging ---
        if ep % log_interval == 0 or ep == 1:
            avg_reward = np.mean(all_rewards[-log_interval:])
            avg_burned = np.mean(all_burned[-log_interval:])
            print(f"  Episode {ep:>4d}/{episodes}  |  "
                  f"Avg Reward: {avg_reward:>7.2f}  |  "
                  f"Avg Burned: {avg_burned:>5.1f}  |  "
                  f"Epsilon: {agent.epsilon:.4f}  |  "
                  f"Q-table: {agent.get_q_table_size()} states")

        # --- Save policy checkpoints ---
        if ep in save_at:
            policy_path = os.path.join(policy_dir,
                                       f"policy_{exp_name}_ep{ep}.pkl")
            agent.save(policy_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")

    # --- Save final policy ---
    final_path = os.path.join(policy_dir, f"policy_{exp_name}_final.pkl")
    agent.save(final_path)
    mlflow.log_artifact(final_path, artifact_path="models")

    # --- Write results CSV ---
    csv_path = f"results/results_{exp_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "reward", "total_burned", "epsilon", "steps", "q_table_size"
        ])
        writer.writeheader()
        writer.writerows(episode_logs)
    print(f"[RESULTS] Episode log saved to {csv_path}")

    # --- Write log.json summary (MLOps tracking) ---
    log_summary = {
        "run_id": f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "average_reward": round(float(np.mean(all_rewards)), 2),
        "average_burned": round(float(np.mean(all_burned)), 2),
        "final_reward_avg_50": round(float(np.mean(all_rewards[-50:])), 2),
        "final_burned_avg_50": round(float(np.mean(all_burned[-50:])), 2),
        "parameters": {
            "learning_rate": agent_cfg["learning_rate"],
            "discount_factor": agent_cfg["discount_factor"],
            "epsilon_start": agent_cfg["epsilon"],
            "epsilon_min": agent_cfg["epsilon_min"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "grid_size": env_cfg["grid_size"],
            "wind_direction": env_cfg["wind_direction"],
            "base_spread_prob": env_cfg["base_spread_prob"],
        },
        "training_time_seconds": round(elapsed, 1),
        "q_table_states": agent.get_q_table_size(),
        "policy_files": [
            os.path.join(policy_dir, f"policy_{exp_name}_ep{ep}.pkl")
            for ep in save_at
        ] + [final_path],
    }
    log_path = f"results/log_{exp_name}.json"
    with open(log_path, "w") as f:
        json.dump(log_summary, f, indent=2)
    print(f"[MLOPS]   Run summary saved to {log_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Experiment       : {exp_name}")
    print(f"  Episodes         : {episodes}")
    print(f"  Avg Reward (all) : {np.mean(all_rewards):.2f}")
    print(f"  Avg Burned (all) : {np.mean(all_burned):.2f}")
    print(f"  Avg Reward (last50): {np.mean(all_rewards[-50:]):.2f}")
    print(f"  Avg Burned (last50): {np.mean(all_burned[-50:]):.2f}")
    print(f"  Q-table states   : {agent.get_q_table_size()}")
    print(f"  Final epsilon    : {agent.epsilon:.4f}")
    print("=" * 60)

    mlflow.end_run()

    return all_rewards, all_burned, agent


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-learning agent for Wildfire Containment")
    parser.add_argument("--config", type=str,
                        default="configs/qlearning_v1.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
