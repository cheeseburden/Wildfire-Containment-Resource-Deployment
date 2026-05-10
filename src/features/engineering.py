"""
Feature Engineering Module
============================
Transforms raw state representations into enriched feature vectors
for advanced model architectures (DQN, policy gradient).

Feature Categories:
  1. Spatial features — fire density, tree coverage per sector
  2. Temporal features — fire spread velocity, resource deployment history
  3. Contextual features — wind alignment, proximity to edges
  4. Derived features — fire perimeter length, containment ratio
"""

import numpy as np
from collections import deque


class FeatureExtractor:
    """Extracts enriched features from raw wildfire grid state."""

    def __init__(self, grid_size=10, num_sectors=4, history_len=5):
        self.grid_size = grid_size
        self.num_sectors = num_sectors
        self.history_len = history_len
        self.state_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)

    def reset(self):
        """Reset feature history at episode start."""
        self.state_history.clear()
        self.action_history.clear()

    def extract(self, state, grid=None, action=None, step=0, wind_direction="N"):
        """
        Extract a rich feature vector from the current state.

        Args:
            state: tuple of (burning_bin, tree_bin) per sector
            grid: raw numpy grid (optional, for spatial features)
            action: last action taken
            step: current timestep
            wind_direction: current wind direction string

        Returns:
            feature_dict: dict of named features
            feature_vector: numpy array for model input
        """
        features = {}

        # --- Sector-level features (from compressed state) ---
        for i in range(self.num_sectors):
            b_bin = state[i * 2]
            t_bin = state[i * 2 + 1]
            features[f"sector_{i}_burning_bin"] = b_bin
            features[f"sector_{i}_tree_bin"] = t_bin
            features[f"sector_{i}_fire_intensity"] = b_bin / 2.0  # normalised
            features[f"sector_{i}_vulnerability"] = t_bin * (1 + b_bin) / 6.0

        # --- Global features ---
        total_burning = sum(state[i * 2] for i in range(self.num_sectors))
        total_trees = sum(state[i * 2 + 1] for i in range(self.num_sectors))
        features["global_fire_pressure"] = total_burning / (self.num_sectors * 2)
        features["global_tree_coverage"] = total_trees / (self.num_sectors * 2)
        features["containment_ratio"] = 1.0 - features["global_fire_pressure"]

        # --- Temporal features ---
        features["timestep_normalised"] = step / 50.0  # normalised to max_steps
        features["time_pressure"] = max(0, (step - 25) / 25.0)  # urgency ramp

        self.state_history.append(state)
        if action is not None:
            self.action_history.append(action)

        # Fire spread velocity (change in total burning over history)
        if len(self.state_history) >= 2:
            prev_burn = sum(self.state_history[-2][i * 2] for i in range(self.num_sectors))
            curr_burn = total_burning
            features["fire_velocity"] = (curr_burn - prev_burn) / 2.0
        else:
            features["fire_velocity"] = 0.0

        # Action diversity (how many unique sectors deployed to recently)
        if len(self.action_history) > 0:
            unique_actions = len(set(self.action_history))
            features["action_diversity"] = unique_actions / self.num_sectors
        else:
            features["action_diversity"] = 0.0

        # --- Wind features ---
        wind_encoding = {
            "N": [1, 0, 0, 0], "S": [0, 1, 0, 0],
            "E": [0, 0, 1, 0], "W": [0, 0, 0, 1],
            "NE": [0.7, 0, 0.7, 0], "NW": [0.7, 0, 0, 0.7],
            "SE": [0, 0.7, 0.7, 0], "SW": [0, 0.7, 0, 0.7],
        }
        wind_vec = wind_encoding.get(wind_direction, [0, 0, 0, 0])
        for j, w in enumerate(wind_vec):
            features[f"wind_{j}"] = w

        # --- Spatial features from raw grid ---
        if grid is not None:
            fire_cells = np.argwhere(grid == 2)  # BURNING
            if len(fire_cells) > 0:
                fire_centroid = fire_cells.mean(axis=0)
                features["fire_centroid_r"] = fire_centroid[0] / self.grid_size
                features["fire_centroid_c"] = fire_centroid[1] / self.grid_size

                # Fire proximity to grid edges (escape risk)
                min_edge_dist = min(
                    fire_cells[:, 0].min(), fire_cells[:, 1].min(),
                    self.grid_size - 1 - fire_cells[:, 0].max(),
                    self.grid_size - 1 - fire_cells[:, 1].max(),
                )
                features["fire_edge_proximity"] = 1.0 - (min_edge_dist / (self.grid_size / 2))

                # Fire perimeter estimate
                perimeter = 0
                for r, c in fire_cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            if grid[nr, nc] != 2:
                                perimeter += 1
                        else:
                            perimeter += 1
                features["fire_perimeter"] = perimeter / (4 * self.grid_size)
            else:
                features["fire_centroid_r"] = 0.5
                features["fire_centroid_c"] = 0.5
                features["fire_edge_proximity"] = 0.0
                features["fire_perimeter"] = 0.0

        # Build feature vector
        feature_vector = np.array(list(features.values()), dtype=np.float32)

        return features, feature_vector

    @property
    def feature_dim(self):
        """Return the dimensionality of the feature vector."""
        # Calculate by doing a dummy extraction
        dummy_state = tuple([0] * (self.num_sectors * 2))
        _, vec = self.extract(dummy_state, step=0)
        return len(vec)


class StateEncoder:
    """Encode raw states for neural network consumption."""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    def one_hot_grid(self, grid):
        """Convert grid to one-hot encoding (5 channels for 5 cell types)."""
        channels = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        for cell_type in range(5):
            channels[cell_type] = (grid == cell_type).astype(np.float32)
        return channels

    def flat_normalised(self, grid):
        """Flatten and normalise grid to [0, 1] range."""
        return grid.flatten().astype(np.float32) / 4.0
