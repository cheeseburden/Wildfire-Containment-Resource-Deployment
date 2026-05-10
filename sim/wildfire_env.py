"""
Wildfire Containment Simulator
==============================
A grid-based wildfire spread environment for reinforcement learning.

The simulator models:
  - A 2D grid where each cell can be: EMPTY (0), TREE (1), BURNING (2), BURNED (3), FIREBREAK (4)
  - Fire spreads probabilistically to neighbouring tree cells each timestep
  - Wind direction biases fire spread probability
  - The RL agent deploys firefighting resources (ground crews / helicopters) to
    create firebreaks and suppress fire, with the goal of minimizing total burned area.

State Representation (compressed for Q-table):
  The grid is divided into sectors (e.g., 4 quadrants for a 10x10 grid).
  For each sector we track: (burning_count_bin, tree_count_bin).
  This gives a compact, hashable state that allows the Q-table to generalise.

Action Space:
  The agent selects a sector (quadrant) to deploy resources to.
  Resources are deployed to the burning cell closest to the fire front
  within that sector, or to a tree cell adjacent to fire (firebreak).

SDG Alignment: SDG 13 (Climate Action) + SDG 15 (Life on Land)
"""

import numpy as np


# Cell states
EMPTY     = 0
TREE      = 1
BURNING   = 2
BURNED    = 3
FIREBREAK = 4

# Wind direction offsets  (N, S, E, W, NE, NW, SE, SW)
NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, 1), (0, -1),
                    (-1, 1), (-1, -1), (1, 1), (1, -1)]


class WildfireEnv:
    """
    Grid-based Wildfire Containment Environment.

    State : tuple of (burning_bin, tree_bin) per sector — compact for Q-table
    Action: sector index (0 .. num_sectors-1) — which quadrant to deploy resources
    Reward: negative of newly burned cells this step (penalise fire spread)
    """

    def __init__(self, grid_size=10, num_resources=2,
                 base_spread_prob=0.3, wind_spread_bonus=0.2,
                 wind_direction="N", max_steps=50,
                 num_initial_fires=2, tree_density=0.85,
                 num_sectors_per_side=2):
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.base_spread_prob = base_spread_prob
        self.wind_spread_bonus = wind_spread_bonus
        self.wind_direction = wind_direction
        self.max_steps = max_steps
        self.num_initial_fires = num_initial_fires
        self.tree_density = tree_density
        self.num_sectors_per_side = num_sectors_per_side

        # Derived
        self.n_cells = grid_size * grid_size
        self.num_sectors = num_sectors_per_side ** 2   # e.g., 4 quadrants
        self.sector_size = grid_size // num_sectors_per_side
        self.action_size = self.num_sectors  # deploy to a sector
        self.state_size = self.num_sectors * 2  # (burning_bin, tree_bin) per sector

        # Wind bias map
        self._wind_map = self._build_wind_map()

        # Precompute sector boundaries
        self._sector_bounds = []
        for sr in range(num_sectors_per_side):
            for sc in range(num_sectors_per_side):
                r0 = sr * self.sector_size
                r1 = r0 + self.sector_size
                c0 = sc * self.sector_size
                c1 = c0 + self.sector_size
                self._sector_bounds.append((r0, r1, c0, c1))

        self.reset()

    # ------------------------------------------------------------------
    # Wind helpers
    # ------------------------------------------------------------------
    def _build_wind_map(self):
        """Return dict mapping (dr, dc) -> extra spread probability."""
        wind_vectors = {
            "N":  (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
            "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
        }
        wv = wind_vectors.get(self.wind_direction, (0, 0))
        wmap = {}
        for dr, dc in NEIGHBOR_OFFSETS:
            # Dot-product style bonus: neighbours in the wind direction get bonus
            dot = dr * wv[0] + dc * wv[1]
            bonus = max(0, dot) * self.wind_spread_bonus
            wmap[(dr, dc)] = bonus
        return wmap

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the environment and return the initial (discretised) state."""
        self.step_count = 0

        # Initialise grid — mostly trees
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if np.random.random() < self.tree_density:
                    self.grid[r, c] = TREE

        # Ignite initial fires at random tree cells
        tree_cells = list(zip(*np.where(self.grid == TREE)))
        if len(tree_cells) < self.num_initial_fires:
            fire_cells = tree_cells
        else:
            idxs = np.random.choice(len(tree_cells),
                                    self.num_initial_fires, replace=False)
            fire_cells = [tree_cells[i] for i in idxs]
        for r, c in fire_cells:
            self.grid[r, c] = BURNING

        # Resource positions (start at centre)
        mid = self.grid_size // 2
        self.resource_positions = [(mid, mid)] * self.num_resources

        self.total_burned = 0
        self.done = False
        return self._get_state()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Execute one step.
        action: sector index (0 .. num_sectors-1) — deploy resources to this sector.
        Returns: (state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        # --- 1. Deploy resources to the chosen sector ---
        cells_suppressed = 0
        r0, r1, c0, c1 = self._sector_bounds[action % self.num_sectors]

        # Find the best cell in this sector to deploy to:
        # Priority 1: Suppress a burning cell
        # Priority 2: Create firebreak on a tree cell adjacent to a burning cell
        deploy_r, deploy_c = None, None

        # Priority 1 — find burning cells in sector
        burning_in_sector = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                if self.grid[r, c] == BURNING:
                    burning_in_sector.append((r, c))

        if burning_in_sector:
            # Deploy to a random burning cell in this sector
            idx = np.random.randint(len(burning_in_sector))
            deploy_r, deploy_c = burning_in_sector[idx]
            self.grid[deploy_r, deploy_c] = FIREBREAK
            cells_suppressed += 1
        else:
            # Priority 2 — find trees adjacent to fire in sector
            adj_trees = []
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if self.grid[r, c] == TREE:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                                if self.grid[nr, nc] == BURNING:
                                    adj_trees.append((r, c))
                                    break
            if adj_trees:
                idx = np.random.randint(len(adj_trees))
                deploy_r, deploy_c = adj_trees[idx]
                self.grid[deploy_r, deploy_c] = FIREBREAK
            else:
                # Fallback — deploy to centre of sector (may hit empty)
                deploy_r = (r0 + r1) // 2
                deploy_c = (c0 + c1) // 2
                if self.grid[deploy_r, deploy_c] == TREE:
                    self.grid[deploy_r, deploy_c] = FIREBREAK

        # Suppress neighbours around deployment point
        if deploy_r is not None:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = deploy_r + dr, deploy_c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == BURNING and np.random.random() < 0.35:
                        self.grid[nr, nc] = FIREBREAK
                        cells_suppressed += 1

        # Update resource positions
        if deploy_r is not None:
            self.resource_positions = [(deploy_r, deploy_c)] * self.num_resources

        # --- 2. Fire spread phase ---
        burned_before = np.sum(self.grid == BURNED)
        new_grid = self.grid.copy()

        burning_cells = list(zip(*np.where(self.grid == BURNING)))
        for br, bc in burning_cells:
            # Current burning cell may burn out
            if np.random.random() < 0.15:
                new_grid[br, bc] = BURNED
            # Spread to neighbours
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == TREE:
                        spread_p = self.base_spread_prob + self._wind_map.get((dr, dc), 0)
                        if np.random.random() < spread_p:
                            new_grid[nr, nc] = BURNING

        self.grid = new_grid
        burned_after = np.sum((self.grid == BURNED) | (self.grid == BURNING))
        new_burned = max(0, burned_after - burned_before)

        self.total_burned += new_burned
        self.step_count += 1

        # --- 3. Check termination ---
        no_fire = np.sum(self.grid == BURNING) == 0
        if no_fire or self.step_count >= self.max_steps:
            self.done = True

        # --- 4. Reward ---
        # Negative of newly-burned cells, bonus for suppression
        reward = -new_burned + cells_suppressed * 2.0

        info = {
            "new_burned": new_burned,
            "total_burned": self.total_burned,
            "cells_suppressed": cells_suppressed,
            "burning_cells": int(np.sum(self.grid == BURNING)),
            "step": self.step_count,
        }

        return self._get_state(), reward, self.done, info

    # ------------------------------------------------------------------
    # State representation (compressed for Q-table)
    # ------------------------------------------------------------------
    def _get_state(self):
        """
        Return a compact, hashable state tuple for Q-table lookup.

        For each sector (quadrant), compute:
          - burning_bin: binned count of burning cells (0=none, 1=few, 2=many)
          - tree_bin:    binned count of remaining tree cells (0=few, 1=some, 2=lots)
        
        This yields a state space of 3^(2*num_sectors) which is tractable.
        For 4 sectors: 3^8 = 6561 possible states.
        """
        state_parts = []
        sector_area = self.sector_size * self.sector_size

        for r0, r1, c0, c1 in self._sector_bounds:
            sector = self.grid[r0:r1, c0:c1]
            burning = int(np.sum(sector == BURNING))
            trees = int(np.sum(sector == TREE))

            # Bin burning: 0 = none, 1 = 1-3, 2 = 4+
            if burning == 0:
                b_bin = 0
            elif burning <= 3:
                b_bin = 1
            else:
                b_bin = 2

            # Bin trees: 0 = <30%, 1 = 30-60%, 2 = >60%
            tree_frac = trees / max(sector_area, 1)
            if tree_frac < 0.3:
                t_bin = 0
            elif tree_frac < 0.6:
                t_bin = 1
            else:
                t_bin = 2

            state_parts.extend([b_bin, t_bin])

        return tuple(state_parts)

    def get_grid_copy(self):
        """Return a copy of the current grid for visualisation."""
        return self.grid.copy()

    # ------------------------------------------------------------------
    # Rendering (text-based)
    # ------------------------------------------------------------------
    def render(self):
        """Print the grid to stdout."""
        symbols = {EMPTY: '.', TREE: 'T', BURNING: '*',
                   BURNED: '#', FIREBREAK: 'B'}
        print(f"\n--- Step {self.step_count} ---")
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                row_str += symbols.get(self.grid[r, c], '?') + " "
            print(row_str)
        print(f"Burning: {np.sum(self.grid == BURNING)}  "
              f"Burned: {np.sum(self.grid == BURNED)}  "
              f"Firebreaks: {np.sum(self.grid == FIREBREAK)}")
