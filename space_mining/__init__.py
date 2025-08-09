"""Top-level public API for space_mining.
Expose small, commonly-used factory functions and classes without importing heavy deps.
"""
from importlib import import_module

# Public lightweight interfaces (lazy imports)

def make_env(*args, **kwargs):
    """Create and return the SpaceMining environment instance.
    Lazy-import from space_mining.envs to avoid heavy imports at package import time.
    """
    mod = import_module(".envs", package=__name__)
    return mod.make_env(*args, **kwargs)


def PPOAgent(*args, **kwargs):
    """Factory for the PPOAgent wrapper class. Returns the class from agents.ppo_agent.
    Use a factory to avoid importing stable-baselines3 at top-level unless the user needs it.
    """
    mod = import_module(".agents.ppo_agent", package=__name__)
    return mod.PPOAgent(*args, **kwargs)


def save_gif(*args, **kwargs):
    mod = import_module(".scripts.make_gif", package=__name__)
    return mod.save_gif(*args, **kwargs)

__all__ = ["make_env", "PPOAgent", "save_gif"] 