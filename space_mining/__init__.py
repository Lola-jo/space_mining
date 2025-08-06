from gymnasium.envs.registration import register
from .envs.space_mining_env import SpaceMiningEnv
from .envs.wrappers import FlattenActionSpaceWrapper, make_env

def register_envs():
    register(
        id='SpaceMining-v1',
        entry_point='space_mining.envs:make_env',
    )

# Auto-register on import
register_envs()

__all__ = ['SpaceMiningEnv', 'FlattenActionSpaceWrapper', 'make_env'] 