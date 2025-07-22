import gym

class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # 用自定义奖励函数替换 reward
        # 这里假设 reward_fn(self.env, action, obs, info) 返回 (reward, reward_info)
        custom_reward, reward_info = self.reward_fn(self.env, action, obs, info)
        info.update(reward_info)
        return obs, custom_reward, done, truncated, info