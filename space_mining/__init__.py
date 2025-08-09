"""Top-level package for space_mining."""
from importlib import import_module

def make_env(**kwargs):
    mod = import_module(".envs", package=__name__)
    return mod.make_env(**kwargs)

def PPOAgent(*args, **kwargs):
    mod = import_module(".agents.ppo_agent", package=__name__)
    return mod.PPOAgent(*args, **kwargs)

# Assuming save_gif is in scripts or utils, but for now omit or add if created

__all__ = ["make_env", "PPOAgent"] 