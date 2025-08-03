import pytest
from space_mining.envs import make_env

def test_env_creation():
    env = make_env()
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    env.close() 