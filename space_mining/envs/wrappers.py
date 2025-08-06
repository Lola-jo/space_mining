import gymnasium as gym
from .space_mining_env import SpaceMiningEnv
import numpy as np
from typing import Tuple, Dict, Any

class FlattenActionSpaceWrapper(gym.Wrapper):
    """
    Wrapper to ensure compatibility with standard RL algorithms in Gymnasium.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = env.action_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

def make_env(**kwargs) -> gym.Env:
    """
    Create and wrap the space mining environment.

    Returns:
        A wrapped environment compatible with standard RL algorithms
    """
    # For backwards compatibility, remove multi-agent parameters
    if 'num_agents' in kwargs:
        kwargs.pop('num_agents')

    # Remove communication radius if present (only used in multi-agent version)
    if 'communication_radius' in kwargs:
        kwargs.pop('communication_radius')

    env = SpaceMiningEnv(**kwargs)
    env = FlattenActionSpaceWrapper(env)
    return env 