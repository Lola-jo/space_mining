# SpaceMining: Single-Agent Reinforcement Learning Environment

SpaceMining is a challenging single-agent RL environment simulating asteroid mining in a 2D space environment. The agent (a mining robot) must collect resources from asteroids and deliver them to the mothership, while managing energy consumption and avoiding moving obstacles. The environment features realistic physics simulation, partial observability, and a comprehensive fitness scoring system with a target range of approximately 3000 points.

## Research Purpose

This environment was specifically designed to evaluate large language models' ability to design reward functions for unfamiliar environments without prior knowledge. Recent studies have raised concerns that large language models may carry prior knowledge from pretraining data about standard RL environments (like CartPole, BipedalWalker, Ant), leading to potential prompt leakage and evaluation biases. To address this issue, SpaceMining serves as a custom environment to assess true generalization capabilities on tasks free from such pretrained knowledge.

**Project Website**: [https://lola-jo.github.io/space_mining/](https://lola-jo.github.io/space_mining/)

## Visualization & Demo

The GIF demonstrations showcase different agent behaviors and training outcomes:

### Successful Behaviors
- **Complete Episode (1200 steps)**: `assets/gif/space_mining_demo_episode_5_20250730_224355.gif` - Agent completes full episode with optimal resource collection
- **All Asteroids Depleted**: `assets/gif/space_mining_demo_episode_4_20250730_222922.gif` - Successful exploration and complete resource extraction
- **Efficient Mining**: `assets/gif/space_mining_demo_episode_4_20250730_225245.gif` - Strategic resource collection and energy management

### Learning Phases & Failures
- **Energy Depletion**: `assets/gif/space_mining_demo_episode_3_20250730_224120.gif` - Agent fails to return to mothership for recharging
- **Poor Obstacle Avoidance**: `assets/gif/space_mining_demo_episode_4_20250730_222922.gif` - Multiple collisions due to inadequate obstacle avoidance
- **Early Exploration**: `assets/gif/space_mining_demo_episode_1_20250730_230841.gif` - Initial learning phase with slow movement and random exploration

### Visual Elements
- **Green Circle**: Agent (mining robot)
- **Blue Circle**: Mothership (store resources and replenish energy for intelligent agents)
- **Yellow Circles**: Asteroids (mining targets) with health bars showing remaining resources
- **Red Circles**: Moving obstacles (avoid)
- **Red Ring**: Mining range indicator
- **Light Blue Ring**: Observation range
- **Status Display**: Top-left shows inventory, energy, and step count

## Features
- **2D Physics Simulation**: Realistic movement with thrust, inertia, drag, and collision detection
- **Resource & Energy Management**: The agent must balance mining, delivery, and energy recharge
- **Partial Observability**: Limited observation radius (15 units) requiring strategic exploration
- **Dynamic Environment**: Randomized asteroids, moving obstacles, and resource distribution
- **Comprehensive Fitness Scoring**: Multi-component evaluation system targeting ~3000 points
- **Real-time Visualization**: Health bars, status indicators, and visual feedback

## Installation

### Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Lola-jo/space_mining.git
cd space_mining

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation

```bash
# Clone the repository
git clone https://github.com/Lola-jo/space_mining.git
cd space_mining

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Test the Environment**
   ```bash
   python test_env.py
   ```

2. **Train the Agent**
   ```bash
   # Train with default settings
   python train_space_mining.py
   
   # Or train with custom parameters
   python train_space_mining.py --mode train
   ```

3. **Evaluate Trained Model**
   ```bash
   # Evaluate a trained model
   python train_space_mining.py --mode evaluate --model_path ./space_mining/train_*/final_model
   ```

4. **Visualize Results**
   ```bash
   python enjoy_trained_model.py
   ```

5. **Quick Training Example**
   ```python
   import gymnasium as gym
   from stable_baselines3 import PPO
   from env_code.env import SpaceMiningEnv
   
   # Create environment
   env = SpaceMiningEnv(render_mode=None)
   
   # Train model
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=3000000)
   model.save("space_mining_model")
   ```

## Environment Details

### Observation Space
- **Agent State**: [position(2), velocity(2), energy(1), inventory(1)] - 6 dimensions
- **Asteroids**: Up to 15 visible asteroids [relative position(2), resource amount(1)] - 45 dimensions
- **Mothership**: [relative position(2)] - 2 dimensions
- **Total**: 53 dimensions

### Action Space
- **Thrust Control**: [fx, fy] in [-1.0, 1.0] - 2 dimensions
- **Mining Action**: Scalar in [0.0, 1.0] (active if >0.5) - 1 dimension
- **Total**: 3 dimensions

### Reward Structure (default)
- Mining success: `+8.0 × mined amount`
- Resource delivery: `+12.0 × delivered amount`
- Energy recharge: `+0.5 × recharge amount`
- Obstacle collision: `-10.0`
- Energy depletion: `-20.0`
- Boundary collision: `-1.0`
- Invalid mining: `-0.5` (if no asteroid nearby)
- Exploration bonus: `+0.1` for discovering new asteroids
- Path efficiency bonus: `+0.05` for efficient movement

### Configurable Parameters (see env.py)
- `max_episode_steps` (default 1200)
- `grid_size` (default 80)
- `max_asteroids` (default 12)
- `max_resource_per_asteroid` (default 40)
- `observation_radius` (default 15)
- `mining_range` (default 8.0)
- `max_inventory` (default 100)
- `render_mode`: None, "human", or "rgb_array"

### Rendering
- 2D top-down view with centered 80x80 grid
- **Green Circle**: Agent (mining robot)
- **Blue Circle**: Mothership (delivery point)
- **Yellow Circles**: Asteroids (mining targets) with health bars
- **Red Circles**: Moving obstacles (avoid)
- **Red Ring**: Mining range indicator
- **Light Blue Ring**: Observation range
- **Status Display**: Top-left shows inventory, energy, and step count

### Fitness Score
A comprehensive score for agent performance targeting ~3000 points, considering:
- Resource collection efficiency (weighted heavily)
- Energy management and remaining energy
- Resource depletion ratio
- Proximity to resource-rich asteroids
- Proximity to mothership when carrying resources
- Completion bonus for total resource collection
- Efficiency bonus for resource collection per step
- Survival bonus for episode duration

## Task Description
The agent is deployed in a 2D space environment (80x80 grid) with randomly distributed asteroids and a central mothership. The agent must navigate efficiently to discover and mine resource-rich asteroids, manage energy consumption and return to the mothership for recharging, avoid collisions with moving obstacles, and deliver resources to maximize collection efficiency. The environment presents a medium difficulty challenge with limited observation radius, energy management constraints, and resource depletion mechanics requiring efficient mining strategies.

## Training Configuration
The environment is optimized for PPO and other modern RL algorithms. Recommended parameters:

```python
# Environment Parameters
max_episode_steps: 1200
grid_size: 80
max_asteroids: 12
max_resource_per_asteroid: 40
observation_radius: 15
mining_range: 8.0
max_inventory: 100

# Training Parameters
policy: MlpPolicy
learning_rate: 0.0003
total_timesteps: 3,000,000
batch_size: 64
n_steps: 2048
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
```

## Citation
If you use SpaceMining in your research, please cite this repository.

## Contributing
Contributions and issues are welcome! Please feel free to submit pull requests or report issues on the [GitHub repository](https://github.com/Lola-jo/space_mining).

---

For more details, see the code in `env_code/` and the training scripts. 