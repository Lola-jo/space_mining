import gymnasium as gym
from space_mining import register_envs

# Register the environments
register_envs()

# Create the environment
env = gym.make("SpaceMining-v1")

# Reset the environment
obs, info = env.reset()

# Take a step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("Observation:", obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close() 