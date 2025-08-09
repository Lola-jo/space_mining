import pytest
from space_mining.agents.ppo_agent import PPOAgent
import numpy as np
import os
from stable_baselines3 import PPO
from space_mining.envs import make_env

@pytest.fixture
def dummy_model():
    env = make_env()
    model = PPO('MlpPolicy', env, n_steps=2)
    yield model
    env.close()

def test_ppo_agent_load(dummy_model, tmp_path):
    model_path = tmp_path / "test_model.zip"
    dummy_model.save(model_path)
    agent = PPOAgent.load(model_path)
    assert agent.model is not None
    os.remove(model_path)

def test_ppo_agent_predict(dummy_model, tmp_path):
    model_path = tmp_path / "test_model.zip"
    dummy_model.save(model_path)
    agent = PPOAgent.load(model_path)
    env = make_env()
    obs, _ = env.reset()
    action, _ = agent.predict(obs)
    assert action.shape == (3,)
    env.close()
    os.remove(model_path)

def test_ppo_agent_save(dummy_model, tmp_path):
    agent = PPOAgent(dummy_model)
    save_path = tmp_path / "saved_model.zip"
    agent.save(save_path)
    assert os.path.exists(save_path)
    os.remove(save_path)

