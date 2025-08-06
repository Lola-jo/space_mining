"""Space Mining Environment module.

This module provides the SpaceMiningEnv and related wrappers.
"""

from .space_mining_env import SpaceMiningEnv
from .wrappers import FlattenActionSpaceWrapper, make_env

from gymnasium.envs.registration import register

def register_envs():
    register(
        id='SpaceMining-v1',
        entry_point='space_mining.envs:make_env',
    )

register_envs()

__all__ = ['SpaceMiningEnv', 'FlattenActionSpaceWrapper', 'make_env']