# Using Space Mining with Stable Baselines3

## Training

```python
from space_mining.agents.train import train_ppo

model = train_ppo(total_timesteps=1000000, output_dir='my_training')
```

## Advanced Training Configuration

### Hyperparameter Tuning
Stable Baselines3 PPO has many tunable parameters. Recommended starting point for Space Mining:

```yaml
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
```

Use Optuna or Ray Tune for systematic tuning.

### Using Callbacks
Implement custom callbacks for early stopping or custom logging:

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Custom logic
        return True

model.learn(total_timesteps=1000000, callback=CustomCallback())
```

### Vectorized Environments
For faster training:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

n_envs = 8
env = SubprocVecEnv([lambda: gym.make('SpaceMining-v1') for i in range(n_envs)])
model = PPO('MlpPolicy', env, verbose=1)
```

## Evaluation and Visualization

### Custom Evaluation
Evaluate with multiple episodes:

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f'Mean reward: {mean_reward} +/- {std_reward}')
```

### TensorBoard Logging
View training metrics:

```bash
tensorboard --logdir ./train_output/tensorboard_logs/
```

### Advanced GIF Generation
Generate GIFs with custom rendering:

```python
def generate_custom_gif(model_path, gif_path, episodes=5):
    model = PPO.load(model_path)
    env = gym.make('SpaceMining-v1', render_mode='rgb_array')
    
    frames = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            done = terminated or truncated
    imageio.mimsave(gif_path, frames, fps=30)
```

## Saving and Loading

### Checkpoints
Use CheckpointCallback to save during training:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/')
model.learn(total_timesteps=1000000, callback=checkpoint_callback)
```

### Replay Buffer
Save and load replay buffer:

```python
model.save_replay_buffer('replay_buffer')
model.load_replay_buffer('replay_buffer')
```

## Advanced Usage

### Custom Policy
Define a custom policy architecture:

```python
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_arch = [dict(pi=[128, 128], vf=[128, 128])]

model = PPO(CustomPolicy, env)
```

### HER for Goal-Oriented Tasks
Although Space Mining isn't goal-based, you can adapt it with HER.

## Examples

### Full Training Script
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

def train():
    env = DummyVecEnv([lambda: gym.make('SpaceMining-v1')])
    eval_env = DummyVecEnv([lambda: gym.make('SpaceMining-v1')])
    
    model = PPO('MlpPolicy', env, verbose=1)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)
    
    model.learn(total_timesteps=2000000, callback=eval_callback)
    model.save('final_model')
    
if __name__ == '__main__':
    train()
```

## FAQ

### Why PPO for Space Mining?
PPO is stable and works well with continuous action spaces like this environment.

### How many timesteps are needed?
Typically 1M-3M timesteps for good performance, depending on hyperparameters.

### Can I use other algorithms?
Yes, try SAC or TD3 for continuous actions:

```python
from stable_baselines3 import SAC
model = SAC('MlpPolicy', env, verbose=1)
```

### How to handle large observation spaces?
Use CNN policies if adding visual observations, but current is MLP-friendly.

### Training is slow, what to do?
Use more parallel environments or GPU acceleration if available.

This expansion provides in-depth guidance for using Stable Baselines3 with Space Mining."
