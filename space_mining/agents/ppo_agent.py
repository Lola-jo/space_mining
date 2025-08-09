from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path):
        model = PPO.load(path)
        return cls(model)

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        self.model.save(path)
