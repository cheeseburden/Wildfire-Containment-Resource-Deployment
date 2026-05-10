"""
MLflow Tracking Integration
==============================
Wraps the training pipeline with MLflow experiment tracking for
full reproducibility, model registry, and comparison.

Features:
  - Auto-log hyperparameters, metrics, and artifacts
  - Model versioning via MLflow Model Registry
  - Cross-experiment comparison
  - Artifact storage (policies, plots, configs)
"""

import os
import json
import time
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# MLflow import with graceful fallback
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] MLflow not installed. Tracking will use local JSON fallback.")

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent
from src.models.sarsa_agent import SARSAAgent
from src.models.dqn_agent import DQNAgent
from src.models.double_q_agent import DoubleQLearningAgent


AGENT_REGISTRY = {
    "qlearning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
    "double_q": DoubleQLearningAgent,
}


def train_with_tracking(config, experiment_name="wildfire-containment"):
    """Full training loop with MLflow tracking."""
    exp_name = config["experiment_name"]
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]
    algorithm = config.get("algorithm", "qlearning")

    agent_class = AGENT_REGISTRY.get(algorithm, QLearningAgent)

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=f"{exp_name}_{algorithm}")
        
        # Log parameters
        mlflow.log_params({
            "algorithm": algorithm,
            "learning_rate": agent_cfg["learning_rate"],
            "discount_factor": agent_cfg["discount_factor"],
            "epsilon": agent_cfg["epsilon"],
            "epsilon_min": agent_cfg["epsilon_min"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "episodes": train_cfg["episodes"],
            "grid_size": env_cfg["grid_size"],
            "wind_direction": env_cfg["wind_direction"],
            "base_spread_prob": env_cfg["base_spread_prob"],
            "num_initial_fires": env_cfg.get("num_initial_fires", 2),
        })
        mlflow.set_tag("team", "ml_engineering")
        mlflow.set_tag("sdg_alignment", "SDG13,SDG15")

    print(f"\n{'='*60}")
    print(f"  TRAINING WITH MLFLOW TRACKING")
    print(f"  Experiment: {exp_name}")
    print(f"  Algorithm: {algorithm}")
    print(f"  MLflow: {'Active' if MLFLOW_AVAILABLE else 'Fallback (local JSON)'}")
    print(f"{'='*60}\n")

    # Initialise
    env = WildfireEnv(**env_cfg)
    
    # Build agent kwargs (DQN needs different params)
    agent_kwargs = {
        "state_size": env.state_size,
        "action_size": env.action_size,
        "learning_rate": agent_cfg["learning_rate"],
        "discount_factor": agent_cfg["discount_factor"],
        "epsilon": agent_cfg["epsilon"],
        "epsilon_min": agent_cfg["epsilon_min"],
        "epsilon_decay": agent_cfg["epsilon_decay"],
    }
    if algorithm == "dqn":
        agent_kwargs["hidden_dim"] = agent_cfg.get("hidden_dim", 64)
        agent_kwargs["replay_size"] = agent_cfg.get("replay_size", 5000)
        agent_kwargs["batch_size"] = agent_cfg.get("batch_size", 32)

    agent = agent_class(**agent_kwargs)

    episodes = train_cfg["episodes"]
    log_interval = train_cfg["log_interval"]
    save_at = train_cfg.get("save_policy_at", [])
    policy_dir = train_cfg.get("policy_dir", "models/")

    os.makedirs(policy_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_rewards = []
    all_burned = []
    all_steps = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        step = 0

        if algorithm == "sarsa":
            action = agent.choose_action(state)
            while True:
                next_state, reward, done, info = env.step(action)
                next_action = agent.choose_action(next_state) if not done else 0
                agent.learn(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_reward += reward
                step += 1
                if done:
                    break
        else:
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
        all_rewards.append(total_reward)
        all_burned.append(env.total_burned)
        all_steps.append(step)

        # Log metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "episode_reward": float(total_reward),
                "episode_burned": float(env.total_burned),
                "episode_steps": step,
                "epsilon": float(agent.epsilon),
            }, step=ep)

        if ep % log_interval == 0 or ep == 1:
            avg_r = np.mean(all_rewards[-log_interval:])
            avg_b = np.mean(all_burned[-log_interval:])
            print(f"  Ep {ep:>4d}/{episodes} | "
                  f"Avg Reward: {avg_r:>7.2f} | "
                  f"Avg Burned: {avg_b:>5.1f} | "
                  f"ε: {agent.epsilon:.4f}")

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "rolling_avg_reward": float(avg_r),
                    "rolling_avg_burned": float(avg_b),
                }, step=ep)

        # Save checkpoints
        if ep in save_at:
            ckpt_path = os.path.join(policy_dir, f"policy_{exp_name}_{algorithm}_ep{ep}.pkl")
            agent.save(ckpt_path)
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(ckpt_path)

    elapsed = time.time() - start_time

    # Final policy
    final_path = os.path.join(policy_dir, f"policy_{exp_name}_{algorithm}_final.pkl")
    agent.save(final_path)

    # Final metrics
    final_metrics = {
        "total_training_time": round(elapsed, 1),
        "final_avg_reward_50": float(np.mean(all_rewards[-50:])),
        "final_avg_burned_50": float(np.mean(all_burned[-50:])),
        "overall_avg_reward": float(np.mean(all_rewards)),
        "overall_avg_burned": float(np.mean(all_burned)),
        "q_table_size": agent.get_q_table_size(),
        "final_epsilon": float(agent.epsilon),
    }

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(final_path)

        # Log config as artifact
        config_artifact = os.path.join("results", f"config_{exp_name}.json")
        with open(config_artifact, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_artifact)

        mlflow.end_run()

    # Local summary
    summary = {
        "run_id": f"{exp_name}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "algorithm": algorithm,
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": final_metrics,
        "policy_path": final_path,
    }
    summary_path = os.path.join("results", f"run_{exp_name}_{algorithm}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {algorithm.upper()}")
    print(f"  Time: {elapsed:.1f}s | Final Reward: {final_metrics['final_avg_reward_50']:.2f}")
    print(f"  Burned: {final_metrics['final_avg_burned_50']:.1f} | ε: {final_metrics['final_epsilon']:.4f}")
    print(f"{'='*60}")

    return agent, all_rewards, all_burned, final_metrics


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Train with MLflow tracking")
    parser.add_argument("--config", default="configs/qlearning_v1.yaml")
    parser.add_argument("--algorithm", default=None,
                        choices=["qlearning", "sarsa", "dqn", "double_q"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.algorithm:
        config["algorithm"] = args.algorithm

    train_with_tracking(config)
