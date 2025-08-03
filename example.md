# Space Mining Example Usage

## Register and Create Environment

```python
import gymnasium as gym
from space_mining import register_envs

# Register the environments
register_envs()

# Create the environment
env = gym.make("SpaceMining-v1")

# Reset
obs, info = env.reset()

# Step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
``` 