"""
Data Generation & Collection
==============================
Generates wildfire simulation episodes as structured datasets for
supervised pre-training and offline RL analysis.

Produces:
  - Episode trajectories (state, action, reward, next_state, done)
  - Aggregated episode statistics
  - Environmental condition snapshots
"""

import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sim.wildfire_env import WildfireEnv


def generate_random_episodes(env_config, n_episodes=500, output_dir="data/raw"):
    """Generate episode data using random policy for baseline analysis."""
    os.makedirs(output_dir, exist_ok=True)

    trajectories = []
    episode_stats = []

    for ep in range(n_episodes):
        env = WildfireEnv(**env_config)
        state = env.reset()
        ep_trajectory = []
        total_reward = 0
        step = 0

        while True:
            action = np.random.randint(0, env.action_size)
            next_state, reward, done, info = env.step(action)

            ep_trajectory.append(
                {
                    "episode": ep,
                    "step": step,
                    "state": list(state),
                    "action": int(action),
                    "reward": round(float(reward), 4),
                    "next_state": list(next_state),
                    "done": done,
                    "burning_cells": info.get("burning_cells", 0),
                    "cells_suppressed": info.get("cells_suppressed", 0),
                }
            )

            state = next_state
            total_reward += reward
            step += 1

            if done:
                break

        trajectories.extend(ep_trajectory)

        episode_stats.append(
            {
                "episode": ep,
                "total_reward": round(float(total_reward), 4),
                "total_burned": int(env.total_burned),
                "steps": step,
                "wind_direction": env_config.get("wind_direction", "N"),
                "num_fires": env_config.get("num_initial_fires", 2),
                "spread_prob": env_config.get("base_spread_prob", 0.3),
            }
        )

        if (ep + 1) % 100 == 0:
            print(f"  Generated {ep + 1}/{n_episodes} episodes")

    # Save trajectories
    traj_path = os.path.join(output_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    # Save episode stats CSV
    stats_path = os.path.join(output_dir, "episode_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_stats[0].keys())
        writer.writeheader()
        writer.writerows(episode_stats)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "n_episodes": n_episodes,
        "n_transitions": len(trajectories),
        "env_config": env_config,
        "files": [traj_path, stats_path],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[DATA] Generated {len(trajectories)} transitions from {n_episodes} episodes"
    )
    print(f"[DATA] Saved to {output_dir}")
    return trajectories, episode_stats


def generate_varied_conditions(output_dir="data/raw/varied"):
    """Generate data across multiple environmental conditions for robustness."""
    conditions = [
        {"wind_direction": "N", "base_spread_prob": 0.25, "num_initial_fires": 1},
        {"wind_direction": "N", "base_spread_prob": 0.30, "num_initial_fires": 2},
        {"wind_direction": "NE", "base_spread_prob": 0.35, "num_initial_fires": 3},
        {"wind_direction": "E", "base_spread_prob": 0.30, "num_initial_fires": 2},
        {"wind_direction": "S", "base_spread_prob": 0.40, "num_initial_fires": 4},
        {"wind_direction": "SW", "base_spread_prob": 0.30, "num_initial_fires": 2},
    ]

    all_stats = []
    for i, extra in enumerate(conditions):
        env_cfg = {
            "grid_size": 10,
            "num_resources": 2,
            "wind_spread_bonus": 0.2,
            "max_steps": 50,
            "tree_density": 0.85,
            "num_sectors_per_side": 2,
            **extra,
        }
        print(
            f"\n[CONDITION {i + 1}/{len(conditions)}] "
            f"wind={extra['wind_direction']}, fires={extra['num_initial_fires']}, "
            f"spread={extra['base_spread_prob']}"
        )

        cond_dir = os.path.join(output_dir, f"condition_{i + 1}")
        _, stats = generate_random_episodes(
            env_cfg, n_episodes=200, output_dir=cond_dir
        )
        all_stats.extend(stats)

    # Combined stats
    combined_path = os.path.join(output_dir, "all_conditions_stats.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)

    print(
        f"\n[DATA] Combined {len(all_stats)} episodes across {len(conditions)} conditions"
    )
    return all_stats


if __name__ == "__main__":
    default_cfg = {
        "grid_size": 10,
        "num_resources": 2,
        "base_spread_prob": 0.3,
        "wind_spread_bonus": 0.2,
        "wind_direction": "N",
        "max_steps": 50,
        "num_initial_fires": 2,
        "tree_density": 0.85,
        "num_sectors_per_side": 2,
    }
    generate_random_episodes(default_cfg, n_episodes=500)
    generate_varied_conditions()
