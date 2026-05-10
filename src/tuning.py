"""
Hyperparameter Tuning Module
===============================
Systematic search over RL hyperparameters using grid search
and random search strategies with cross-validation.

Tunable parameters:
  - Learning rate (α)
  - Discount factor (γ)  
  - Epsilon decay rate
  - Epsilon minimum
  - Number of episodes
  - Grid configuration (sectors, fire count)
"""

import os
import json
import time
import itertools
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent
from src.models.sarsa_agent import SARSAAgent
from src.models.double_q_agent import DoubleQLearningAgent


AGENT_REGISTRY = {
    "qlearning": QLearningAgent,
    "sarsa": SARSAAgent,
    "double_q": DoubleQLearningAgent,
}


def evaluate_config(agent_class, agent_params, env_config,
                    train_episodes=300, eval_episodes=50, seed=None):
    """Train an agent with given params and return evaluation metrics."""
    if seed is not None:
        np.random.seed(seed)

    env = WildfireEnv(**env_config)
    agent = agent_class(
        state_size=env.state_size,
        action_size=env.action_size,
        **agent_params,
    )

    # Training phase
    train_rewards = []
    for ep in range(train_episodes):
        state = env.reset()
        total_reward = 0

        if agent_class == SARSAAgent:
            action = agent.choose_action(state)
            while True:
                next_state, reward, done, info = env.step(action)
                next_action = agent.choose_action(next_state) if not done else 0
                agent.learn(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_reward += reward
                if done:
                    break
        else:
            while True:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

        agent.decay_epsilon()
        train_rewards.append(total_reward)

    # Evaluation phase (greedy)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    eval_rewards = []
    eval_burned = []

    for _ in range(eval_episodes):
        env_eval = WildfireEnv(**env_config)
        state = env_eval.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            state, reward, done, info = env_eval.step(action)
            total_reward += reward
            if done:
                break
        eval_rewards.append(total_reward)
        eval_burned.append(env_eval.total_burned)

    agent.epsilon = original_epsilon

    return {
        "eval_avg_reward": float(np.mean(eval_rewards)),
        "eval_std_reward": float(np.std(eval_rewards)),
        "eval_avg_burned": float(np.mean(eval_burned)),
        "eval_std_burned": float(np.std(eval_burned)),
        "train_final_50_reward": float(np.mean(train_rewards[-50:])),
        "convergence_episode": _find_convergence(train_rewards),
    }


def _find_convergence(rewards, window=50, threshold=0.05):
    """Find the episode where rewards stabilise (change < threshold)."""
    if len(rewards) < window * 2:
        return len(rewards)

    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    for i in range(len(smoothed) - window):
        segment = smoothed[i:i + window]
        if np.std(segment) / (abs(np.mean(segment)) + 1e-8) < threshold:
            return i + window
    return len(rewards)


def cross_validate(agent_class, agent_params, env_config,
                   n_folds=5, train_episodes=300, eval_episodes=30):
    """K-fold cross-validation with different random seeds."""
    fold_results = []
    for fold in range(n_folds):
        seed = fold * 42 + 7
        result = evaluate_config(
            agent_class, agent_params, env_config,
            train_episodes, eval_episodes, seed=seed,
        )
        result["fold"] = fold
        result["seed"] = seed
        fold_results.append(result)

    # Aggregate
    avg_result = {
        "cv_avg_reward": float(np.mean([r["eval_avg_reward"] for r in fold_results])),
        "cv_std_reward": float(np.std([r["eval_avg_reward"] for r in fold_results])),
        "cv_avg_burned": float(np.mean([r["eval_avg_burned"] for r in fold_results])),
        "cv_std_burned": float(np.std([r["eval_avg_burned"] for r in fold_results])),
        "cv_avg_convergence": float(np.mean([r["convergence_episode"] for r in fold_results])),
        "folds": fold_results,
    }
    return avg_result


def grid_search(algorithm="qlearning", env_config=None, output_dir="results/tuning"):
    """Exhaustive grid search over hyperparameter space."""
    os.makedirs(output_dir, exist_ok=True)

    if env_config is None:
        env_config = {
            "grid_size": 10, "num_resources": 2, "base_spread_prob": 0.3,
            "wind_spread_bonus": 0.2, "wind_direction": "N", "max_steps": 50,
            "num_initial_fires": 2, "tree_density": 0.85, "num_sectors_per_side": 2,
        }

    agent_class = AGENT_REGISTRY[algorithm]

    param_grid = {
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "discount_factor": [0.90, 0.95, 0.99],
        "epsilon_decay": [0.990, 0.995, 0.998],
        "epsilon_min": [0.03, 0.05],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    print("\n" + "="*60)
    print("  HYPERPARAMETER GRID SEARCH")
    print(f"  Algorithm: {algorithm}")
    print(f"  Combinations: {len(combos)}")
    print("="*60 + "\n")

    results = []
    best_reward = -float("inf")
    best_config = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["epsilon"] = 1.0  # Always start full exploration

        print(f"  [{i+1}/{len(combos)}] "
              f"α={params['learning_rate']}, γ={params['discount_factor']}, "
              f"ε_decay={params['epsilon_decay']}, ε_min={params['epsilon_min']}")

        start = time.time()
        cv_result = cross_validate(
            agent_class, params, env_config,
            n_folds=3, train_episodes=200, eval_episodes=30,
        )
        elapsed = time.time() - start

        entry = {
            "rank": 0,
            "algorithm": algorithm,
            "params": params,
            "cv_avg_reward": cv_result["cv_avg_reward"],
            "cv_std_reward": cv_result["cv_std_reward"],
            "cv_avg_burned": cv_result["cv_avg_burned"],
            "cv_avg_convergence": cv_result["cv_avg_convergence"],
            "time_seconds": round(elapsed, 1),
        }
        results.append(entry)

        if cv_result["cv_avg_reward"] > best_reward:
            best_reward = cv_result["cv_avg_reward"]
            best_config = entry

        print(f"    → Avg Reward: {cv_result['cv_avg_reward']:.2f} "
              f"± {cv_result['cv_std_reward']:.2f}  "
              f"| Burned: {cv_result['cv_avg_burned']:.1f}  "
              f"| Time: {elapsed:.1f}s")

    # Rank results
    results.sort(key=lambda x: x["cv_avg_reward"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Save results
    results_path = os.path.join(output_dir, f"grid_search_{algorithm}.json")
    with open(results_path, "w") as f:
        json.dump({
            "search_type": "grid",
            "algorithm": algorithm,
            "timestamp": datetime.now().isoformat(),
            "total_combinations": len(combos),
            "best_config": best_config,
            "all_results": results,
        }, f, indent=2)

    print("\n" + "="*60)
    print("  BEST CONFIG (Rank 1)")
    print(f"  Avg Reward: {best_config['cv_avg_reward']:.2f}")
    print(f"  Params: {best_config['params']}")
    print("="*60)
    print(f"\n[TUNING] Results saved to {results_path}")

    return best_config, results


def random_search(algorithm="qlearning", n_trials=20, env_config=None,
                  output_dir="results/tuning"):
    """Random search over continuous hyperparameter ranges."""
    os.makedirs(output_dir, exist_ok=True)

    if env_config is None:
        env_config = {
            "grid_size": 10, "num_resources": 2, "base_spread_prob": 0.3,
            "wind_spread_bonus": 0.2, "wind_direction": "N", "max_steps": 50,
            "num_initial_fires": 2, "tree_density": 0.85, "num_sectors_per_side": 2,
        }

    agent_class = AGENT_REGISTRY[algorithm]

    print("\n" + "="*60)
    print("  HYPERPARAMETER RANDOM SEARCH")
    print(f"  Algorithm: {algorithm}")
    print(f"  Trials: {n_trials}")
    print("="*60 + "\n")

    results = []
    best_reward = -float("inf")
    best_config = None

    for trial in range(n_trials):
        params = {
            "learning_rate": float(np.random.uniform(0.01, 0.3)),
            "discount_factor": float(np.random.uniform(0.85, 0.99)),
            "epsilon": 1.0,
            "epsilon_min": float(np.random.uniform(0.01, 0.1)),
            "epsilon_decay": float(np.random.uniform(0.985, 0.999)),
        }

        print(f"  [Trial {trial+1}/{n_trials}] "
              f"α={params['learning_rate']:.3f}, γ={params['discount_factor']:.3f}")

        start = time.time()
        cv_result = cross_validate(
            agent_class, params, env_config,
            n_folds=3, train_episodes=200, eval_episodes=30,
        )
        elapsed = time.time() - start

        entry = {
            "trial": trial + 1,
            "algorithm": algorithm,
            "params": {k: round(v, 4) for k, v in params.items()},
            "cv_avg_reward": cv_result["cv_avg_reward"],
            "cv_std_reward": cv_result["cv_std_reward"],
            "cv_avg_burned": cv_result["cv_avg_burned"],
            "time_seconds": round(elapsed, 1),
        }
        results.append(entry)

        if cv_result["cv_avg_reward"] > best_reward:
            best_reward = cv_result["cv_avg_reward"]
            best_config = entry

        print(f"    → Avg Reward: {cv_result['cv_avg_reward']:.2f} "
              f"| Burned: {cv_result['cv_avg_burned']:.1f}")

    # Save
    results_path = os.path.join(output_dir, f"random_search_{algorithm}.json")
    with open(results_path, "w") as f:
        json.dump({
            "search_type": "random",
            "algorithm": algorithm,
            "n_trials": n_trials,
            "best_config": best_config,
            "all_results": results,
        }, f, indent=2)

    print(f"\n  BEST: Trial {best_config['trial']}, Reward={best_reward:.2f}")
    print(f"[TUNING] Saved to {results_path}")
    return best_config, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--algorithm", default="qlearning",
                        choices=["qlearning", "sarsa", "double_q"])
    parser.add_argument("--method", default="random", choices=["grid", "random"])
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    if args.method == "grid":
        grid_search(args.algorithm)
    else:
        random_search(args.algorithm, n_trials=args.trials)
