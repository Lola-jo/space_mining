import pytest
from space_mining.envs import make_env
import numpy as np

def test_env_creation():
    env = make_env()
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    env.close()

def test_env_step():
    env = make_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()

def test_action_space():
    env = make_env()
    assert env.action_space.shape == (3,)
    action = env.action_space.sample()
    assert action.shape == (3,)
    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)
    env.close()

def test_observation_space():
    env = make_env()
    assert env.observation_space.shape == (53,)
    obs, _ = env.reset()
    assert obs.shape == (53,)
    assert env.observation_space.contains(obs)
    env.close()

def test_multiple_steps():
    env = make_env()
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

def test_render_rgb():
    env = make_env(render_mode='rgb_array')
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (800, 800, 3)
    env.close()

def test_energy_depletion():
    env = make_env()
    env.agent_energy = 0
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert terminated
    assert reward < 0
    env.close()
