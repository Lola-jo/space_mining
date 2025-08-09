import pytest
from space_mining.envs import make_env

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
