# SpaceMining: Multi-Agent Reinforcement Learning Environment

SpaceMining is a challenging multi-agent RL environment simulating collaborative asteroid mining in space. Agents (mining robots) must collect resources from asteroids and deliver them to the mothership, while managing energy, avoiding obstacles, and collaborating with each other.

## Features
- **Physics Simulation**: Thrust, inertia, collisions, and simplified gravity
- **Resource & Energy Management**: Agents must balance mining, delivery, and energy recharge
- **Partial Observability**: Each agent perceives only a limited radius
- **Dynamic Environment**: Randomized asteroids, obstacles, and resource distribution
- **Multi-Agent Collaboration**: Agents can communicate and coordinate

## Installation

```bash
pip install gymnasium numpy pygame
```

## Quick Start

1. **Test the Environment**
   ```bash
   python envs/space_mining/test_env.py
   ```

2. **Train with PPO**
   ```bash
   python stable_eureka/main.py --config envs/space_mining/space_mining_ollama.yml
   ```

3. **Manual Training Example**
   ```python
   import gymnasium as gym
   from stable_baselines3 import PPO
   from envs.space_mining.env_code.env import SpaceMiningEnv
   env = SpaceMiningEnv()
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=500000)
   model.save("space_mining_model")
   ```

## Environment Details

### Observation Space
- **Agent State**: [position(3), velocity(3), energy(1), inventory(1)]
- **Asteroids**: Up to 10 visible asteroids [relative position(3), resource amount(1)]
- **Mothership**: [relative position(3)]

### Action Space
- **Thrust**: [fx, fy, fz] in [-1.0, 1.0]
- **Mining**: Scalar in [0.0, 1.0] (active if >0.5)

### Reward Structure
- Mining success: `+2.0 × mined amount`
- Resource delivery: `+10.0 × delivered amount`
- Energy recharge: `+0.2 × recharge amount`
- Obstacle collision: `-10.0`
- Energy depletion: `-50.0`
- Boundary collision: `-2.0`
- Invalid mining: `-0.5` (if no asteroid nearby)

### Configurable Parameters
- `max_episode_steps` (default 1000)
- `grid_size` (default 100)
- `max_asteroids` (default 15)
- `max_resource_per_asteroid` (default 50)
- `observation_radius` (default 30)
- `render_mode`: None, "human", or "rgb_array"

### Rendering
- 2D isometric projection
- Green: agent, Brown: asteroid, Blue: mothership, Red: obstacle
- Energy bar and inventory indicator

### Fitness Score
A comprehensive score for agent performance, considering:
- Resource collection efficiency
- Energy management
- Proximity to resource-rich asteroids
- Proximity to mothership when carrying resources

## Task Description
See `task_description.txt` for a detailed English description of the multi-agent collaborative mining task, including goals, behaviors, physical properties, evaluation metrics, and reward design.

## Training Configuration
See `space_mining_ollama.yml` for a full training and evaluation configuration (PPO, hyperparameters, parallelism, etc).

## Visualization & Web Demo
- **Success GIFs**: See `experiments/space_mining_llama3/2025-07-15-17-21/code/iteration_4/sample_2/episode_1_*.gif` for successful agent behaviors.
- **Failure GIFs**: See `experiments/space_mining_llama3/2025-07-15-17-21/code/iteration_0/sample_1/eval.gif` for failed training runs.
- You can embed these GIFs in your project website to showcase agent learning and failure cases.

## Citation
If you use SpaceMining in your research, please cite this repository.

---

For more details, see the code in `env_code/` and the training scripts. Contributions and issues are welcome! 