"""
Tests — PyroShield AI
========================
Comprehensive test suite for environment, agents, features, and API.
"""

import sys
import os
import numpy as np
import json

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent
from src.models.sarsa_agent import SARSAAgent
from src.models.dqn_agent import DQNAgent
from src.models.double_q_agent import DoubleQLearningAgent
from src.features.engineering import FeatureExtractor, StateEncoder


# ─── Environment Tests ────────────────────────────────────────

def test_env_creation():
    """Test environment initialisation."""
    env = WildfireEnv(grid_size=10, num_sectors_per_side=2)
    assert env.grid_size == 10
    assert env.action_size == 4
    assert env.state_size == 8
    print("✅ test_env_creation passed")


def test_env_reset():
    """Test environment reset produces valid state."""
    env = WildfireEnv(grid_size=10, num_sectors_per_side=2)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 8
    assert all(0 <= v <= 2 for v in state)
    print("✅ test_env_reset passed")


def test_env_step():
    """Test environment step returns valid outputs."""
    env = WildfireEnv(grid_size=10, num_sectors_per_side=2)
    state = env.reset()
    next_state, reward, done, info = env.step(0)
    assert isinstance(next_state, tuple)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    print("✅ test_env_step passed")


def test_env_episode():
    """Test running a full episode."""
    env = WildfireEnv(grid_size=10, max_steps=20)
    state = env.reset()
    total_reward = 0
    steps = 0
    while True:
        action = np.random.randint(0, env.action_size)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    assert steps > 0
    assert steps <= 20
    print(f"✅ test_env_episode passed (steps={steps}, reward={total_reward:.2f})")


# ─── Agent Tests ──────────────────────────────────────────────

def test_qlearning_agent():
    """Test Q-Learning agent basic operations."""
    agent = QLearningAgent(state_size=8, action_size=4)
    state = (0, 1, 0, 2, 1, 1, 0, 0)
    action = agent.choose_action(state)
    assert 0 <= action < 4

    agent.learn(state, action, -5.0, (1, 1, 0, 1, 1, 1, 0, 0), False)
    agent.decay_epsilon()
    assert agent.epsilon < 1.0
    print("✅ test_qlearning_agent passed")


def test_sarsa_agent():
    """Test SARSA agent."""
    agent = SARSAAgent(state_size=8, action_size=4)
    state = (0, 1, 0, 2, 1, 1, 0, 0)
    action = agent.choose_action(state)
    next_state = (1, 1, 0, 1, 1, 1, 0, 0)
    next_action = agent.choose_action(next_state)
    agent.learn(state, action, -5.0, next_state, next_action, False)
    assert agent.get_q_table_size() > 0
    print("✅ test_sarsa_agent passed")


def test_dqn_agent():
    """Test DQN agent."""
    agent = DQNAgent(state_size=8, action_size=4, hidden_dim=32, batch_size=4)
    state = (0, 1, 0, 2, 1, 1, 0, 0)
    action = agent.choose_action(state)
    assert 0 <= action < 4
    
    # Store some transitions
    for _ in range(10):
        next_state = tuple(np.random.randint(0, 3, 8))
        agent.learn(state, action, -5.0, next_state, False)
        state = next_state
        action = agent.choose_action(state)
    print("✅ test_dqn_agent passed")


def test_double_q_agent():
    """Test Double Q-Learning agent."""
    agent = DoubleQLearningAgent(state_size=8, action_size=4)
    state = (0, 1, 0, 2, 1, 1, 0, 0)
    action = agent.choose_action(state)
    agent.learn(state, action, -5.0, (1, 1, 0, 1, 1, 1, 0, 0), False)
    assert agent.get_q_table_size() > 0
    print("✅ test_double_q_agent passed")


def test_agent_save_load(tmp_path=None):
    """Test agent save/load roundtrip."""
    save_path = "tests/_test_policy.pkl"
    agent = QLearningAgent(state_size=8, action_size=4)
    state = (0, 1, 0, 2, 1, 1, 0, 0)
    agent.learn(state, 0, -5.0, (1, 1, 0, 1, 1, 1, 0, 0), False)
    agent.save(save_path)

    agent2 = QLearningAgent(state_size=8, action_size=4)
    agent2.load(save_path)
    assert agent.get_q_table_size() == agent2.get_q_table_size()

    os.remove(save_path)
    print("✅ test_agent_save_load passed")


# ─── Feature Engineering Tests ────────────────────────────────

def test_feature_extractor():
    """Test feature extraction."""
    fe = FeatureExtractor(grid_size=10, num_sectors=4)
    state = (0, 2, 1, 1, 0, 2, 2, 0)
    features, vector = fe.extract(state, step=10)
    
    assert isinstance(features, dict)
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    assert "global_fire_pressure" in features
    assert "containment_ratio" in features
    print(f"✅ test_feature_extractor passed (feature_dim={len(vector)})")


def test_state_encoder():
    """Test state encoder."""
    encoder = StateEncoder(grid_size=10)
    grid = np.random.randint(0, 5, (10, 10)).astype(np.int8)
    
    one_hot = encoder.one_hot_grid(grid)
    assert one_hot.shape == (5, 10, 10)
    
    flat = encoder.flat_normalised(grid)
    assert flat.shape == (100,)
    assert flat.max() <= 1.0
    print("✅ test_state_encoder passed")


# ─── Integration Tests ────────────────────────────────────────

def test_full_training_loop():
    """Test a complete training loop (small scale)."""
    env_cfg = {
        "grid_size": 10, "num_resources": 2, "base_spread_prob": 0.3,
        "wind_spread_bonus": 0.2, "wind_direction": "N", "max_steps": 20,
        "num_initial_fires": 2, "tree_density": 0.85, "num_sectors_per_side": 2,
    }
    
    env = WildfireEnv(**env_cfg)
    agent = QLearningAgent(state_size=env.state_size, action_size=env.action_size)
    
    rewards = []
    for ep in range(20):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        rewards.append(total_reward)
    
    assert len(rewards) == 20
    assert agent.get_q_table_size() > 0
    print(f"✅ test_full_training_loop passed (avg_reward={np.mean(rewards):.2f})")


def test_multi_model_comparison():
    """Test that all model types can train on the same environment."""
    env_cfg = {
        "grid_size": 10, "num_resources": 2, "base_spread_prob": 0.3,
        "wind_spread_bonus": 0.2, "wind_direction": "N", "max_steps": 15,
        "num_initial_fires": 2, "tree_density": 0.85, "num_sectors_per_side": 2,
    }

    models = {
        "Q-Learning": QLearningAgent(state_size=8, action_size=4),
        "SARSA": SARSAAgent(state_size=8, action_size=4),
        "Double-Q": DoubleQLearningAgent(state_size=8, action_size=4),
    }

    for name, agent in models.items():
        env = WildfireEnv(**env_cfg)
        state = env.reset()
        total_reward = 0
        
        if name == "SARSA":
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
        
        print(f"  {name}: reward={total_reward:.2f}")

    print("✅ test_multi_model_comparison passed")


# ─── Run All Tests ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PYROSHIELD AI — TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_env_creation,
        test_env_reset,
        test_env_step,
        test_env_episode,
        test_qlearning_agent,
        test_sarsa_agent,
        test_dqn_agent,
        test_double_q_agent,
        test_agent_save_load,
        test_feature_extractor,
        test_state_encoder,
        test_full_training_loop,
        test_multi_model_comparison,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}\n")
