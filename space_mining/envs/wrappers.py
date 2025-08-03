import gymnasium as gym
from .core import SpaceMiningEnv

class FlattenActionSpaceWrapper(gym.Wrapper):
    """
    Wrapper to ensure compatibility with standard RL algorithms in Gymnasium.
    """
    
    def __init__(self, env):
        """Initialize the wrapper."""
        super().__init__(env)
        
        # The action space is already a Box with shape (4,)
        # [fx, fy, fz, mine]
        # This wrapper exists for compatibility but does not modify the action space
        # for the simplified single-agent environment
        self.action_space = env.action_space
        
    def step(self, action):
        """Execute environment step using the flattened action."""
        return self.env.step(action)

def make_env(**kwargs):
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