import pytest
from space_mining.agents.ppo_agent import PPOAgent
import numpy as np
import os
from stable_baselines3 import PPO
from space_mining.envs import make_env

@pytest.fixture
def dummy_model():
    """Fixture to create a dummy PPO model for testing."""
    env = make_env()
    model = PPO('MlpPolicy', env, n_steps=2)
    yield model
    env.close()

def test_ppo_agent_load(dummy_model, tmp_path):
    """Test loading a PPOAgent from a saved model file."""
    model_path = tmp_path / "test_model.zip"
    dummy_model.save(model_path)
    agent = PPOAgent.load(model_path)
    assert agent.model is not None
    os.remove(model_path)

def test_ppo_agent_predict(dummy_model, tmp_path):
    """Test prediction functionality of PPOAgent with a sample observation."""
    model_path = tmp_path / "test_model.zip"
    dummy_model.save(model_path)
    agent = PPOAgent.load(model_path)
    env = make_env()
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)
    env.close()
    os.remove(model_path)

def test_ppo_agent_save(dummy_model, tmp_path):
    """Test saving a PPOAgent model to a file."""
    agent = PPOAgent(dummy_model)
    save_path = tmp_path / "saved_model.zip"
    agent.save(save_path)
    assert os.path.exists(save_path)
    os.remove(save_path)

def test_ppo_agent_learn(tmp_path):
    """Test training functionality of PPOAgent with a minimal number of timesteps."""
    env = make_env()
    agent = PPOAgent(env=env)
    save_path = tmp_path / "trained_model.zip"
    agent.learn(total_timesteps=10, progress_bar=False)  # Minimal training for test
    agent.save(save_path)
    assert os.path.exists(save_path)
    # Verify model can be loaded and used after training
    loaded_agent = PPOAgent.load(save_path, env=env)
    obs, _ = env.reset()
    action = loaded_agent.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    env.close()
    os.remove(save_path)

