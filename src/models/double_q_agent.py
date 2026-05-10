"""
Double Q-Learning Agent for Wildfire Containment
===================================================
Algorithm: Double Q-Learning (tabular)
Exploration: Epsilon-greedy with decay

Why Double Q-Learning?
  Standard Q-Learning suffers from maximisation bias — it overestimates
  Q-values because it uses the same Q-table to both select and evaluate
  actions. Double Q-Learning maintains TWO Q-tables and randomly selects
  which to update, reducing overestimation and producing more robust policies.

  This is particularly important in wildfire containment where overestimating
  the value of a sector deployment could lead to catastrophic fire spread.
"""

import os
import pickle
from collections import defaultdict

import numpy as np


class DoubleQLearningAgent:
    """Double Q-Learning agent with two Q-tables to reduce maximisation bias."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Two Q-tables
        self.q_table_a = defaultdict(lambda: np.zeros(action_size))
        self.q_table_b = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy using the sum of both Q-tables."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            combined_q = self.q_table_a[state] + self.q_table_b[state]
            valid_q = {a: combined_q[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Double Q-Learning update:
          With 50% prob, update Q_A using Q_B for evaluation, or vice versa.
        """
        if np.random.random() < 0.5:
            # Update Q_A, evaluate with Q_B
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table_a[next_state])
                target = reward + self.gamma * self.q_table_b[next_state][best_action]
            self.q_table_a[state][action] += self.lr * (
                target - self.q_table_a[state][action]
            )
        else:
            # Update Q_B, evaluate with Q_A
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table_b[next_state])
                target = reward + self.gamma * self.q_table_a[next_state][best_action]
            self.q_table_b[state][action] += self.lr * (
                target - self.q_table_b[state][action]
            )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        data = {
            "algorithm": "DoubleQLearning",
            "q_table_a": dict(self.q_table_a),
            "q_table_b": dict(self.q_table_b),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[SAVE] Double Q-Learning policy saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table_a = defaultdict(
            lambda: np.zeros(self.action_size), data["q_table_a"]
        )
        self.q_table_b = defaultdict(
            lambda: np.zeros(self.action_size), data["q_table_b"]
        )
        self.epsilon = data["epsilon"]
        print(f"[LOAD] Double Q-Learning policy loaded from {filepath}")

    def get_q_table_size(self):
        return len(self.q_table_a) + len(self.q_table_b)
