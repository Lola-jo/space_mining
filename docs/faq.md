# Frequently Asked Questions (FAQ) for SpaceMining

This document addresses common questions about the SpaceMining environment, providing clear answers and solutions to help you get started and troubleshoot issues.

## General Questions

### What is SpaceMining?

SpaceMining is a reinforcement learning environment designed to simulate asteroid mining in a 2D space. The agent, a mining robot, must collect resources from asteroids and deliver them to a mothership while managing energy levels and avoiding obstacles. It features realistic physics, partial observability, and a comprehensive reward system, making it a challenging testbed for RL algorithms.

### Why was SpaceMining created?

SpaceMining was developed to evaluate the ability of large language models and RL algorithms to design reward functions and solve complex tasks in unfamiliar environments. It serves as a custom environment to assess generalization capabilities without prior knowledge from pretraining data, addressing concerns about prompt leakage in standard RL benchmarks.

### Is SpaceMining compatible with Gymnasium?

Yes, SpaceMining is fully compatible with the Gymnasium API. You can create the environment using `gymnasium.make('SpaceMining-v0')` after importing the `space_mining` package, which auto-registers the environment.

### Can I use SpaceMining with Stable-Baselines3?

Absolutely. SpaceMining is designed to work seamlessly with Stable-Baselines3. The project includes training scripts and examples using PPO from Stable-Baselines3, and the environment adheres to Gymnasium standards, making it compatible with other algorithms from the library as well.

## Installation Issues

### I'm getting dependency conflicts when installing SpaceMining. What should I do?

Dependency conflicts often occur when other packages in your environment have strict version requirements. To resolve this, use a virtual environment to isolate SpaceMining's dependencies:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
pip install space-mining
```

If conflicts persist, consider installing only the necessary dependencies manually or using tools like `poetry` for dependency resolution.

### Pygame fails to install or throws errors during rendering. How can I fix it?

Pygame requires system-level libraries for graphics and sound. On Ubuntu or Debian, install the prerequisites:

```bash
sudo apt-get update
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

On macOS, use Homebrew:

```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

On Windows, ensure you have a compatible Python version (3.8-3.11), and install Pygame via pip. If issues persist, check the [Pygame installation guide](https://www.pygame.org/wiki/Compilation) for detailed instructions.

## Environment Usage

### How do I create a SpaceMining environment with custom parameters?

You can customize the environment using the `make_env` function from `space_mining`:

```python
from space_mining import make_env

env = make_env(
    max_episode_steps=1500,
    grid_size=100,
    max_asteroids=15,
    observation_radius=20,
    mining_range=10.0,
    render_mode='human'
)
```

Refer to the [Environment Details](./environment_details.md) for a full list of customizable parameters.

### Why does the agent only see a limited number of asteroids?

SpaceMining implements partial observability to simulate a realistic challenge. The agent can only observe asteroids within its `observation_radius` (default: 15 units). This encourages exploration strategies and may require memory-based policies (e.g., recurrent networks or frame stacking) to track unseen asteroids.

### How does the reward system work in SpaceMining?

The reward system follows the GOODREWARD pattern, which includes components for energy efficiency, exploration, path optimization, speed control, and strategic guidance (mining or delivery focus). Additionally, immediate rewards are given for mining resources (+8.0 per unit), delivering resources to the mothership (+12.0 per unit), and recharging energy (+0.5 per unit), with penalties for collisions (-10.0) and energy depletion (-10.0). For a detailed breakdown, see the [Environment Details](./environment_details.md).

### Can I modify the reward function?

Yes, you can subclass `SpaceMiningEnv` to customize the reward function. Here's an example of increasing the exploration bonus:

```python
from space_mining.envs import SpaceMiningEnv

class CustomRewardEnv(SpaceMiningEnv):
    def compute_reward(self, action, observation, info):
        reward, reward_info = super().compute_reward(action, observation, info)
        reward_info['exploration_reward'] *= 2.0  # Double the exploration bonus
        return sum(reward_info.values()), reward_info

env = CustomRewardEnv()
```

## Training and Algorithms

### Which RL algorithm should I use with SpaceMining?

Proximal Policy Optimization (PPO) is recommended due to its stability with continuous action spaces and is the default in provided training scripts. Other suitable algorithms include Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3). Algorithms like DQN are not ideal unless the action space is discretized. See [Training Tips](./training_tips.md) for more guidance.

### How long does it take to train an agent on SpaceMining?

Training typically requires 1 to 3 million timesteps for good performance with PPO, depending on hyperparameters, environment settings, and hardware. Using vectorized environments (multiple parallel instances) can significantly reduce wall-clock time. Expect longer training for more complex policies or if the observation radius is very small.

### My agent isn't learning or performs poorly. What can I do?

Poor performance can stem from several issues:

- **Insufficient Exploration**: Increase the entropy coefficient (`ent_coef`) in PPO or adjust the exploration bonus in the reward function.
- **Hyperparameter Issues**: Tune learning rate (try lowering to 1e-4), batch size, or clip range. Refer to [Training Tips](./training_tips.md) for recommended settings.
- **Energy Depletion**: If the agent runs out of energy quickly, consider increasing the energy recharge reward or reducing consumption rates via environment parameters.
- **Partial Observability**: Use recurrent policies or frame stacking to handle limited visibility.
- **Visualization**: Render the environment to observe behavior and identify failure modes.

## Advanced Usage

### Can SpaceMining support multi-agent scenarios?

Currently, SpaceMining is designed for single-agent scenarios. However, the codebase can be extended for multi-agent support by subclassing `SpaceMiningEnv` and modifying the observation and action spaces to accommodate multiple agents. Contributions for multi-agent features are welcome!

### How can I record videos or GIFs of agent performance?

Use the provided `make_gif.py` script in `space_mining/scripts/` to generate GIFs from a trained model checkpoint:

```bash
python -m space_mining.scripts.make_gif --checkpoint path_to_model.zip --output performance.gif
```

Alternatively, use Gymnasium's `RecordVideo` wrapper for video recording:

```python
from gymnasium.wrappers import RecordVideo
from space_mining import make_env

env = make_env(render_mode='rgb_array')
env = RecordVideo(env, 'videos/')
obs, _ = env.reset()
for _ in range(1200):
    action = env.action_space.sample()  # Replace with your policy
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
```

### Is it possible to use discrete actions instead of continuous?

The default action space is continuous, but you can apply a discretization wrapper to convert it to discrete actions if needed for algorithms like DQN. Gymnasium provides utilities for this, or you can write a custom wrapper to map discrete actions to continuous ranges.

## Troubleshooting

### I'm getting an error that the environment 'SpaceMining-v0' is not found. Why?

This error occurs if the environment is not registered. Ensure you've imported `space_mining` before using `gymnasium.make`:

```python
import space_mining
import gymnasium as gym
env = gym.make('SpaceMining-v0')  # Should work now
```

If the issue persists, verify your installation by reinstalling the package or checking if the environment ID matches the registered one in `space_mining/envs/__init__.py`.

### Rendering is slow or not working. What can I do?

Rendering can be slow if using `human` mode during training. Switch to `rgb_array` mode and visualize frames separately using matplotlib or save them as a video/GIF. Ensure Pygame is installed correctly (see installation troubleshooting). If rendering isn't needed during training, set `render_mode=None` to disable it.

### My training process crashes without clear errors. How can I debug?

Crashes can result from memory issues, especially with vectorized environments, or from environment-specific errors. Try these steps:

- Reduce the number of parallel environments if using vectorization.
- Enable verbose logging in your RL library to get more detailed output.
- Add custom logging in the environment's `step` method to track state before crashes.
- Run a single episode with rendering to visually inspect for anomalies.

## Additional Resources

- **Project Repository**: [GitHub](https://github.com/Lola-jo/space_mining) for issues, discussions, and source code.
- **Documentation**: Check other guides in this folder for installation, environment details, and training tips.

If your question isn't answered here, please open an issue on GitHub or refer to the broader documentation for more detailed information.
