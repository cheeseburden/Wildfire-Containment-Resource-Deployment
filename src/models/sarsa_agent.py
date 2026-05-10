"""
SARSA Agent for Wildfire Containment
======================================
Algorithm: SARSA (on-policy, tabular)
Exploration: Epsilon-greedy with decay

Why SARSA?
  SARSA updates Q-values based on the action actually taken (on-policy),
  making it more conservative than Q-Learning. This is beneficial in
  safety-critical domains like wildfire management where the agent
  should avoid risky exploration strategies during deployment.
"""

import numpy as np
import pickle
import os
from collections import defaultdict


class SARSAAgent:
    """Tabular SARSA agent with epsilon-greedy exploration."""

    def __init__(self, state_size, action_size,
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy action selection."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            valid_q = {a: q_values[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update rule (on-policy):
          Q(s,a) <- Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        
        Key difference from Q-Learning: uses Q(s', a') instead of max_a' Q(s', a')
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the Q-table and parameters."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "algorithm": "SARSA",
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[SAVE] SARSA policy saved to {filepath}")

    def load(self, filepath):
        """Load a previously saved Q-table."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data["q_table"])
        self.epsilon = data["epsilon"]
        self.lr = data["lr"]
        self.gamma = data["gamma"]
        print(f"[LOAD] SARSA policy loaded from {filepath}")

    def get_q_table_size(self):
        return len(self.q_table)
