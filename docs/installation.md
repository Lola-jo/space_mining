# Installation Guide for Space Mining

This guide provides step-by-step instructions for installing the Space Mining environment and its dependencies.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended for isolation)

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/Lola-jo/space_mining.git
   cd space_mining
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the package and dependencies:
   ```bash
   pip install .
   ```

   Or, if you plan to modify the code and want to install in development mode:
   ```bash
   pip install -e .
   ```

### Method 2: Install from PyPI (For Users)

Once the package is published to PyPI, you can install it directly:

```bash
pip install space-mining
```

### Method 3: Install with Poetry (For Advanced Users)

If you prefer using Poetry for dependency management:

1. Install Poetry if you don't have it:
   ```bash
   pip install poetry
   ```

2. Clone the repository and navigate to the project directory as shown in Method 1.

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Verifying Installation

After installation, verify that the environment is correctly installed by running:

```python
import gymnasium as gym
env = gym.make('SpaceMining-v0')
obs, info = env.reset()
print('Installation successful! Environment created.')
env.close()
```

## Troubleshooting

### Common Issues

- **Dependency Conflicts**: If you encounter version conflicts, use a virtual environment to isolate dependencies.
- **Pygame Installation Errors**: Ensure you have the necessary system libraries installed. On Ubuntu, you might need:
  ```bash
  sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  ```
- **Environment Not Found**: Make sure the environment is registered by importing `space_mining` before using `gym.make`.

### Getting Help

If you encounter issues not covered here, please open an issue on the [GitHub repository](https://github.com/Lola-jo/space_mining/issues) or contact the maintainers.

## Additional Components

### Installing for Development

If you plan to contribute to the project, install additional development dependencies:

```bash
pip install -e '.[dev]'
```

This includes tools like `pytest`, `black`, and `isort` for testing and code formatting.

### Installing Documentation Tools

To build or contribute to the documentation:

```bash
pip install -e '.[docs]'
```

This installs Sphinx and related packages for generating documentation.
