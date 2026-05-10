"""
DQN Agent for Wildfire Containment
=====================================
Algorithm: Deep Q-Network (simplified, NumPy-based)
Exploration: Epsilon-greedy with decay + experience replay

Why DQN?
  Scales better than tabular Q-Learning to larger state spaces.
  Uses a neural network to approximate Q(s,a) instead of storing
  explicit Q-table entries. Experience replay breaks temporal
  correlation for more stable learning.

Note: This is a lightweight NumPy implementation (no PyTorch/TF dependency)
  to keep deployment simple. For production, swap with torch.nn.
"""

import numpy as np
import pickle
import os
from collections import deque


class SimpleNeuralNet:
    """Minimal 2-layer neural network in pure NumPy for Q-value approximation."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.lr = lr
        # Xavier initialisation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim // 2)
        self.W3 = np.random.randn(hidden_dim // 2, output_dim) * np.sqrt(2.0 / (hidden_dim // 2))
        self.b3 = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass with ReLU activations."""
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def backward(self, x, target, output):
        """Backward pass with MSE loss gradient."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1:
            x = x.reshape(1, -1)
            target = target.reshape(1, -1)
            output = output.reshape(1, -1)

        # Output layer gradient
        d3 = (output - target) / batch_size
        dW3 = self.a2.T @ d3 if self.a2.ndim > 1 else np.outer(self.a2.flatten(), d3.flatten())
        db3 = d3.sum(axis=0) if d3.ndim > 1 else d3.flatten()

        # Hidden layer 2
        d2 = d3 @ self.W3.T * (self.z2 > 0).astype(float)
        dW2 = self.a1.T @ d2 if self.a1.ndim > 1 else np.outer(self.a1.flatten(), d2.flatten())
        db2 = d2.sum(axis=0) if d2.ndim > 1 else d2.flatten()

        # Hidden layer 1
        d1 = d2 @ self.W2.T * (self.z1 > 0).astype(float)
        dW1 = x.T @ d1 if x.ndim > 1 else np.outer(x.flatten(), d1.flatten())
        db1 = d1.sum(axis=0) if d1.ndim > 1 else d1.flatten()

        # Gradient clipping
        for g in [dW1, dW2, dW3, db1, db2, db3]:
            np.clip(g, -1.0, 1.0, out=g)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def get_params(self):
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
            "W3": self.W3.copy(), "b3": self.b3.copy(),
        }

    def set_params(self, params):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        self.W3 = params["W3"].copy()
        self.b3 = params["b3"].copy()


class DQNAgent:
    """Deep Q-Network agent with experience replay."""

    def __init__(self, state_size, action_size,
                 learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 hidden_dim=64, replay_size=5000, batch_size=32,
                 target_update_freq=50):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = SimpleNeuralNet(state_size, hidden_dim, action_size, lr)
        self.target_network = SimpleNeuralNet(state_size, hidden_dim, action_size, lr)
        self._sync_target()

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_size)
        self.train_step = 0

    def _sync_target(self):
        """Copy Q-network weights to target network."""
        self.target_network.set_params(self.q_network.get_params())

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy with DQN."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            state_arr = np.array(state, dtype=np.float32)
            q_values = self.q_network.forward(state_arr)
            valid_q = {a: q_values[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done,
        ))

    def learn(self, state=None, action=None, reward=None, next_state=None, done=None):
        """Train on a mini-batch from replay buffer."""
        if state is not None:
            self.store(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        for s, a, r, ns, d in batch:
            q_values = self.q_network.forward(s)
            target_q = q_values.copy()

            if d:
                target_q[a] = r
            else:
                next_q = self.target_network.forward(ns)
                target_q[a] = r + self.gamma * np.max(next_q)

            self.q_network.backward(s, target_q, q_values)

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._sync_target()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "algorithm": "DQN",
            "q_network": self.q_network.get_params(),
            "target_network": self.target_network.get_params(),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "train_step": self.train_step,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[SAVE] DQN policy saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_network.set_params(data["q_network"])
        self.target_network.set_params(data["target_network"])
        self.epsilon = data["epsilon"]
        self.train_step = data.get("train_step", 0)
        print(f"[LOAD] DQN policy loaded from {filepath}")

    def get_q_table_size(self):
        """Return replay buffer size as proxy for experience."""
        return len(self.replay_buffer)
