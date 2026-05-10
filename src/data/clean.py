"""
Data Cleaning Pipeline
========================
Validates, deduplicates, and normalises raw trajectory data.

Steps:
  1. Schema validation — ensure all required fields present
  2. Outlier detection — flag episodes with extreme reward/burned values
  3. Deduplication — remove exact duplicate transitions
  4. Normalisation — scale rewards and state features for model input
  5. Train/val/test split for offline evaluation
"""

import os
import json
import csv
import numpy as np
from datetime import datetime


def validate_trajectory(traj):
    """Validate a single trajectory record has all required fields."""
    required = ["episode", "step", "state", "action", "reward", "next_state", "done"]
    for field in required:
        if field not in traj:
            return False, f"Missing field: {field}"
    if not isinstance(traj["state"], list) or len(traj["state"]) != 8:
        return False, f"Invalid state dimensions: {len(traj.get('state', []))}"
    if not (0 <= traj["action"] < 4):
        return False, f"Invalid action: {traj['action']}"
    return True, "OK"


def detect_outliers(episode_stats, z_threshold=3.0):
    """Flag episodes with reward/burned values beyond z-score threshold."""
    rewards = np.array([s["total_reward"] for s in episode_stats])
    burned = np.array([s["total_burned"] for s in episode_stats])

    r_mean, r_std = rewards.mean(), rewards.std() + 1e-8
    b_mean, b_std = burned.mean(), burned.std() + 1e-8

    outlier_episodes = set()
    for i, stat in enumerate(episode_stats):
        r_z = abs((stat["total_reward"] - r_mean) / r_std)
        b_z = abs((stat["total_burned"] - b_mean) / b_std)
        if r_z > z_threshold or b_z > z_threshold:
            outlier_episodes.add(stat["episode"])

    return outlier_episodes


def normalise_rewards(trajectories, clip_range=(-500, 100)):
    """Clip and normalise rewards to a standard range."""
    for traj in trajectories:
        traj["reward_raw"] = traj["reward"]
        traj["reward"] = max(clip_range[0], min(clip_range[1], traj["reward"]))
    return trajectories


def deduplicate(trajectories):
    """Remove exact duplicate transitions based on (episode, step, state, action)."""
    seen = set()
    unique = []
    for traj in trajectories:
        key = (traj["episode"], traj["step"], tuple(traj["state"]), traj["action"])
        if key not in seen:
            seen.add(key)
            unique.append(traj)
    removed = len(trajectories) - len(unique)
    if removed > 0:
        print(f"[CLEAN] Removed {removed} duplicate transitions")
    return unique


def split_data(episode_stats, train_ratio=0.7, val_ratio=0.15):
    """Split episode IDs into train/val/test sets."""
    episodes = [s["episode"] for s in episode_stats]
    np.random.shuffle(episodes)

    n = len(episodes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": sorted(episodes[:n_train]),
        "val": sorted(episodes[n_train:n_train + n_val]),
        "test": sorted(episodes[n_train + n_val:]),
    }
    print(f"[SPLIT] Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits


def clean_pipeline(raw_dir="data/raw", output_dir="data/processed"):
    """Run the full cleaning pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    traj_path = os.path.join(raw_dir, "trajectories.json")
    stats_path = os.path.join(raw_dir, "episode_stats.csv")

    if not os.path.exists(traj_path):
        print(f"[ERROR] {traj_path} not found. Run data generation first.")
        return

    with open(traj_path) as f:
        trajectories = json.load(f)

    episode_stats = []
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["episode"] = int(row["episode"])
                row["total_reward"] = float(row["total_reward"])
                row["total_burned"] = int(row["total_burned"])
                row["steps"] = int(row["steps"])
                episode_stats.append(row)

    print(f"[CLEAN] Loaded {len(trajectories)} transitions, "
          f"{len(episode_stats)} episode stats")

    # Step 1: Validate
    valid = []
    invalid_count = 0
    for traj in trajectories:
        ok, msg = validate_trajectory(traj)
        if ok:
            valid.append(traj)
        else:
            invalid_count += 1
    if invalid_count:
        print(f"[CLEAN] Removed {invalid_count} invalid transitions")
    trajectories = valid

    # Step 2: Detect outliers
    outliers = detect_outliers(episode_stats)
    if outliers:
        print(f"[CLEAN] Flagged {len(outliers)} outlier episodes (not removed, flagged)")

    # Step 3: Deduplicate
    trajectories = deduplicate(trajectories)

    # Step 4: Normalise
    trajectories = normalise_rewards(trajectories)

    # Step 5: Split
    splits = split_data(episode_stats) if episode_stats else None

    # Save cleaned data
    clean_traj_path = os.path.join(output_dir, "trajectories_clean.json")
    with open(clean_traj_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    if splits:
        splits_path = os.path.join(output_dir, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)

    report = {
        "cleaned_at": datetime.now().isoformat(),
        "original_transitions": len(valid) + invalid_count,
        "cleaned_transitions": len(trajectories),
        "invalid_removed": invalid_count,
        "outlier_episodes_flagged": len(outliers) if outliers else 0,
        "splits": {k: len(v) for k, v in splits.items()} if splits else None,
    }
    report_path = os.path.join(output_dir, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[CLEAN] Pipeline complete. Output: {output_dir}")
    return trajectories, splits


if __name__ == "__main__":
    clean_pipeline()
