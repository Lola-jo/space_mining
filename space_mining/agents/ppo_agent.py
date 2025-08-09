"""PPO Agent wrapper for space_mining.
Provides a simple interface for loading, predicting, and saving PPO models.
"""
from stable_baselines3 import PPO

class PPOAgent:
    """A wrapper class for the PPO model used in space_mining."""

    def __init__(self, policy="MlpPolicy", env=None, **kwargs):
        """Initialize the PPO agent.

        Args:
            policy (str): The policy type to use (default: 'MlpPolicy').
            env: The environment to train on (required for training).
            **kwargs: Additional arguments to pass to the PPO constructor.
        """
        self.model = PPO(policy, env, **kwargs) if env else None
        self.policy = policy
        self.kwargs = kwargs

    @classmethod
    def load(cls, path, env=None):
        """Load a trained PPO model from a file.

        Args:
            path (str): Path to the saved model file.
            env: The environment to associate with the model (if predicting).

        Returns:
            PPOAgent: An instance of PPOAgent with the loaded model.
        """
        model = PPO.load(path, env=env)
        agent = cls()
        agent.model = model
        return agent

    def predict(self, observation, deterministic=True):
        """Predict an action given an observation.

        Args:
            observation: The current observation from the environment.
            deterministic (bool): Whether to use deterministic actions (default: True).

        Returns:
            action: The predicted action.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load a model or provide an environment.")
        return self.model.predict(observation, deterministic=deterministic)[0]

    def learn(self, total_timesteps, **kwargs):
        """Train the PPO model.

        Args:
            total_timesteps (int): Total number of timesteps to train for.
            **kwargs: Additional arguments to pass to the learn method.

        Returns:
            self: The trained agent.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Provide an environment during initialization.")
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def save(self, path):
        """Save the PPO model to a file.

        Args:
            path (str): Path to save the model to.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save.")
        self.model.save(path)
