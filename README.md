# SpaceMining: Single-Agent Reinforcement Learning Environment

SpaceMining is a challenging single-agent RL environment simulating asteroid mining in space. The agent (a mining robot) must collect resources from asteroids and deliver them to the mothership, while managing energy and avoiding obstacles.

## Features
- **Physics Simulation**: Thrust, inertia, collisions, and simplified gravity
- **Resource & Energy Management**: The agent must balance mining, delivery, and energy recharge
- **Partial Observability**: The agent perceives only a limited radius
- **Dynamic Environment**: Randomized asteroids, obstacles, and resource distribution
- **Customizable Reward Structure**: Flexible reward and fitness evaluation

## Installation

```bash
pip install gymnasium numpy pygame
```

## Quick Start

1. **Test the Environment**
   ```bash
   python test_env.py
   ```

2. **Train with PPO**
   ```python
   import gymnasium as gym
   from stable_baselines3 import PPO
   from env_code.env import SpaceMiningEnv
   env = SpaceMiningEnv(render_mode=None)
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=500000)
   model.save("space_mining_model")
   ```

## Environment Details

### Observation Space
- **Agent State**: [position(3), velocity(3), energy(1), inventory(1)]
- **Asteroids**: Up to 15 visible asteroids [relative position(3), resource amount(1)]
- **Mothership**: [relative position(3)]

### Action Space
- **Thrust**: [fx, fy, fz] in [-1.0, 1.0]
- **Mining**: Scalar in [0.0, 1.0] (active if >0.5)

### Reward Structure (default)
- Mining success: `+2.0 × mined amount`
- Resource delivery: `+10.0 × delivered amount`
- Energy recharge: `+0.2 × recharge amount`
- Obstacle collision: `-10.0`
- Energy depletion: `-20.0`
- Boundary collision: `-1.0`
- Invalid mining: `-0.5` (if no asteroid nearby)

### Configurable Parameters (see env.py)
- `max_episode_steps` (default 1600)
- `grid_size` (default 100)
- `max_asteroids` (default 25)
- `max_resource_per_asteroid` (default 80)
- `observation_radius` (default 50)
- `render_mode`: None, "human", or "rgb_array"

### Rendering
- 2D isometric projection
- Green: agent, Orange: asteroid, Blue: mothership, Red: obstacle
- Energy bar and inventory indicator

### Fitness Score
A comprehensive score for agent performance, considering:
- Resource collection efficiency
- Energy management
- Proximity to resource-rich asteroids
- Proximity to mothership when carrying resources

## Task Description
See `task_description.txt` for a detailed English description of the single-agent mining task, including goals, behaviors, physical properties, evaluation metrics, and reward design.

## Training Configuration
See `space_mining_ollama.yml` for a full training and evaluation configuration (PPO, hyperparameters, etc).

## Visualization & Web Demo
- **Success GIFs**: See `assets/gif/episode_1_20250722_121511.gif`, etc. for successful agent behaviors.
- **Failure GIFs**: See `assets/gif/episode_4_20250722_121936.gif` for failed training runs.
- You can embed these GIFs in your project website to showcase agent learning and failure cases.

## Citation
If you use SpaceMining in your research, please cite this repository.

---

For more details, see the code in `env_code/` and the training scripts. Contributions and issues are welcome! 