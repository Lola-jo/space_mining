# Using Space Mining with Gymnasium

Space Mining is designed to be fully compatible with the Gymnasium API, making it easy to use with standard reinforcement learning workflows.

## Installation

Install via pip:
```bash
pip install space-mining
```

## Basic Usage

```python
import gymnasium as gym
import space_mining

# The environment is auto-registered on import
env = gym.make('SpaceMining-v1')

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Environment Details

- **Observation Space**: Box with agent state, nearby asteroids, and mothership position.
- **Action Space**: Box with thrust (2D) and mining action.
- **Render Modes**: 'human' and 'rgb_array'.

For more details, see the [Gymnasium documentation](https://gymnasium.farama.org/).

## Training Example

Use with Stable Baselines3:
```python
from stable_baselines3 import PPO

env = gym.make('SpaceMining-v1')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

## Advanced Usage

### Using Wrappers
The Space Mining environment comes with a `FlattenActionSpaceWrapper` that ensures compatibility with algorithms expecting flat action spaces. Here's how to apply it manually:

```python
from space_mining.envs import SpaceMiningEnv, FlattenActionSpaceWrapper

env = SpaceMiningEnv()
env = FlattenActionSpaceWrapper(env)
```

This wrapper flattens the action space from a potential Dict to a Box, making it suitable for most RL algorithms.

### Customizing Environment Parameters
You can customize various parameters when creating the environment using the `make_env` function:

```python
from space_mining.envs import make_env

env = make_env(
    render_mode='human',
    max_episode_steps=1500,
    grid_size=100,
    observation_radius=20,
    mining_range=10.0
)
```

Available parameters include:
- `max_episode_steps`: Maximum steps per episode (default: 1200)
- `grid_size`: Size of the 2D grid (default: 80)
- `max_asteroids`: Maximum number of asteroids (default: 12)
- `observation_radius`: Agent's visibility range (default: 15)
- `mining_range`: Distance for mining activation (default: 8.0)
- And many more - see the `core.py` for full list.

### Vectorized Environments
For faster training, use Gymnasium's vectorized environments:

```python
from gymnasium.vector import SyncVectorEnv
import space_mining

def make_sm_env():
    return gym.make('SpaceMining-v1')

vec_env = SyncVectorEnv([make_sm_env for _ in range(4)])
```

This creates 4 parallel environments for simultaneous data collection.

## Environment Specifications

### Observation Space
The observation is a Box space with shape (53,) consisting of:
- Agent position (x, y): 2 floats
- Agent velocity (vx, vy): 2 floats
- Energy level: 1 float [0, 100]
- Inventory level: 1 float [0, 100]
- Up to 15 nearby asteroids: Each with relative (dx, dy, resource): 45 floats
- Mothership relative position (dx, dy): 2 floats

Values are normalized to [-1, 1] for most components.

### Action Space
Box(-1.0, 1.0, (3,)) where:
- actions[0]: Thrust in x direction
- actions[1]: Thrust in y direction
- actions[2]: Mining activation (threshold at 0.5)

### Reward Function
The default reward includes:
- +8.0 per unit mined
- +12.0 per unit delivered
- +0.5 per unit energy recharged
- -10.0 for obstacle collision
- -20.0 for energy depletion
- -0.5 for invalid mining attempts
- Small bonuses for exploration and efficiency

You can customize rewards by subclassing `SpaceMiningEnv`.

### Termination and Truncation
- Termination: Energy depletion or collision with boundary/obstacle
- Truncation: Reaching max_episode_steps

## Integration with RL Libraries

### With Stable Baselines3
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make('SpaceMining-v1')])
model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=1000000)
model.save('space_mining_ppo')
```

### With RLlib
```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()
config = PPOConfig().environment('SpaceMining-v1').framework('torch')
algo = config.build()
for _ in range(10):
    algo.train()
```

### With Tianshou
```python
import tianshou as ts
from tianshou.env import DummyVectorEnv

train_envs = DummyVectorEnv([lambda: gym.make('SpaceMining-v1') for _ in range(10)])
policy = ts.policy.PPOPolicy(...)  # Configure policy
collector = ts.data.Collector(policy, train_envs)
```

## Tips and Tricks

### Debugging Rendering
If rendering is slow, use 'rgb_array' mode and visualize separately:
```python
env = gym.make('SpaceMining-v1', render_mode='rgb_array')
obs, _ = env.reset()
frame = env.render()
# Use matplotlib to display frame
```

### Handling Partial Observability
The limited observation radius means agents need memory. Consider using recurrent policies or frame stacking.

### Performance Optimization
- Use vectorized environments for parallel training
- Normalize observations if your algorithm doesn't do it automatically
- Tune learning rate based on environment complexity

### Common Issues
- **Action Space Mismatch**: Ensure your algorithm handles Box actions
- **Rendering Errors**: Install pygame and set render_mode appropriately
- **Dependency Conflicts**: Use virtual environments

## FAQ

### How do I register a custom version?
Use gymnasium.register:
```python
gym.register(
    id='SpaceMining-v2',
    entry_point='space_mining.envs:make_env',
    kwargs={'grid_size': 100}
)
```

### Can I use discrete actions?
The environment uses continuous actions, but you can add a discretization wrapper.

### What's the expected reward range?
Episodic rewards typically range from -50 (poor) to +5000 (excellent), depending on configuration.

### How to record videos?
Use Gymnasium's RecordVideo wrapper:
```python
from gymnasium.wrappers import RecordVideo
env = RecordVideo(gym.make('SpaceMining-v1'), 'videos')
```

### Is multi-agent supported?
Currently single-agent, but the code can be extended for multi-agent scenarios.
