"""
Q-Learning Agent for Wildfire Containment
==========================================
Algorithm: Q-Learning (off-policy, tabular)
Exploration: Epsilon-greedy with decay

Why Q-Learning?
  "Q-learning is chosen because the state space (discretised grid + resource positions)
   is manageable for a tabular approach, and Q-learning's off-policy nature allows
   efficient learning from exploratory actions."
"""

import numpy as np
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

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

        # Q-table as defaultdict for sparse state spaces
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state, valid_actions=None):
        """
        Epsilon-greedy action selection.
        
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with highest Q-value (exploitation).
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(valid_actions)
        else:
            # Exploitation: best Q-value action
            q_values = self.q_table[state]
            # Among valid actions, pick the one with max Q
            valid_q = {a: q_values[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule:
          Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the Q-table and parameters to a pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
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
        print(f"[SAVE] Policy saved to {filepath}")

    def load(self, filepath):
        """Load a previously saved Q-table."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data["q_table"])
        self.epsilon = data["epsilon"]
        self.lr = data["lr"]
        self.gamma = data["gamma"]
        print(f"[LOAD] Policy loaded from {filepath} (epsilon={self.epsilon:.4f})")

    def get_q_table_size(self):
        """Return the number of states in the Q-table."""
        return len(self.q_table)
