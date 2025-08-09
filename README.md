# SpaceMining: A Reinforcement Learning Environment for Asteroid Mining

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/)

SpaceMining is a Gymnasium-compatible reinforcement learning (RL) environment designed to simulate asteroid mining in a 2D space. The agent, a mining robot, must collect resources from asteroids and deliver them to a central mothership while managing energy levels and avoiding moving obstacles. Featuring realistic physics, partial observability, and a comprehensive reward system, SpaceMining offers a challenging testbed for RL algorithms.

## Key Features

- **Gymnasium Compatibility**: Fully adheres to the Gymnasium API for seamless integration with standard RL workflows.
- **Stable-Baselines3 Support**: Optimized for training with PPO and other algorithms from Stable-Baselines3.
- **Complex Environment**: Includes energy management, partial observability (limited observation radius), continuous action spaces, and dynamic obstacles.
- **Comprehensive Reward System**: Implements the GOODREWARD pattern, rewarding energy efficiency, exploration, path optimization, and strategic behavior.
- **Visualization Tools**: Scripts to render episodes and generate GIFs for performance analysis.

## Installation

### From Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/Lola-jo/space_mining.git
   cd space_mining
   ```
2. Install in a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   pip install .
   ```
   For development mode:
   ```bash
   pip install -e '.[dev]'
   ```

### From PyPI (For Users)

Once published to PyPI, install directly:
```bash
pip install space-mining
```

For detailed installation instructions and troubleshooting, see [Installation Guide](docs/installation.md).

## Quick Start

### Running a Random Agent Example

Run the provided example script to see a random agent in action:
```bash
python example.py
```

### Training a PPO Agent

Train a PPO agent using the provided script:
```bash
python -m space_mining.agents.train_ppo --total_timesteps 1000000 --output_dir my_training
```

### Generating a GIF of a Trained Agent

Visualize a trained agent's performance as a GIF:
```bash
python -m space_mining.scripts.make_gif --checkpoint my_training/model.zip --output performance.gif
```

## Environment Overview

- **Observation Space**: Box shape `(53,)`, including agent position, velocity, energy, inventory, nearby asteroids (up to 15 within observation radius), and mothership position.
- **Action Space**: Continuous Box shape `(3,)`, controlling thrust in x/y directions and mining activation.
- **Reward Structure**: Combines immediate rewards for mining (+8.0/unit) and delivery (+12.0/unit) with GOODREWARD components for efficiency, exploration, and strategic guidance.
- **Termination Conditions**: Energy depletion or excessive collisions.
- **Customization**: Adjustable parameters like grid size, observation radius, and episode length.

For a detailed breakdown, refer to [Environment Details](docs/environment_details.md).

## Project Structure

- **`space_mining/envs/`**: Gymnasium environment implementations and wrappers.
- **`space_mining/agents/`**: PPO training and inference logic using Stable-Baselines3.
- **`space_mining/scripts/`**: Utility scripts for visualization (e.g., rendering episodes, generating GIFs).
- **`docs/`**: Comprehensive documentation on installation, usage, training, and contributing.
- **`tests/`**: Unit tests for environment and agent functionality.
- **`example.py`**: Demonstration script for running the environment with a random or trained agent.

## Training Agents

SpaceMining is optimized for training with PPO from Stable-Baselines3. Key tips:
- **Algorithm Choice**: PPO is recommended for continuous action spaces.
- **Training Duration**: Typically 1-3 million timesteps for good performance.
- **Hyperparameters**: Start with learning rate=0.0003, adjust based on stability.
- **Vectorization**: Use multiple parallel environments for faster training.

For detailed guidance, see [Training Tips](docs/training_tips.md).

## Visualization and Evaluation

Evaluate and visualize agent performance with provided scripts:
- **Render Episode**: Watch an agent perform in real-time.
  ```bash
  python -m space_mining.scripts.render_episode --model_path my_training/model.zip
  ```
- **Generate GIF**: Create a shareable GIF of agent behavior.
  ```bash
  python -m space_mining.scripts.make_gif --checkpoint my_training/model.zip --output performance.gif
  ```

## Contributing

We welcome contributions to SpaceMining! Whether it's bug fixes, new features, or documentation improvements, your help is appreciated. Please follow these steps:
1. Read the [Contributing Guide](docs/contributing.md) for setup and guidelines.
2. Fork the repository and create a branch for your changes.
3. Submit a pull request with a clear description of your contribution.

## FAQ and Troubleshooting

Have questions or encountering issues? Check the [FAQ](docs/faq.md) for common solutions, or open an issue on GitHub for assistance.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/Lola-jo/space_mining/issues).

---

*SpaceMining is a unique RL environment designed to test generalization in unfamiliar settings, avoiding prompt leakage issues in standard benchmarks. Join us in exploring the challenges of asteroid mining through reinforcement learning!* 