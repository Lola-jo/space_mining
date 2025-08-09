"""Environment module for space_mining.
Registers the SpaceMining environment with Gymnasium.
"""
from gymnasium.envs.registration import register

from .space_mining_env import SpaceMiningEnv

def register_envs():
    """Register the space mining environments with Gymnasium."""
    register(
        id="SpaceMining-v0",
        entry_point="space_mining.envs:SpaceMiningEnv",
        max_episode_steps=1200,
    )

def make_env(**kwargs):
    """Create and return a SpaceMining environment instance."""
    return SpaceMiningEnv(**kwargs)

# Register environments on module import
register_envs()

__all__ = ["SpaceMiningEnv", "make_env"]