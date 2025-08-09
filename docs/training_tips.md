# Training Tips for SpaceMining Environment

This guide offers tips and best practices for training reinforcement learning agents on the SpaceMining environment to achieve optimal performance.

## Understanding the Environment

Before training, familiarize yourself with the environment's key features:

- **Partial Observability**: The agent can only see asteroids within a limited radius (default: 15 units), requiring exploration strategies.
- **Energy Management**: Balancing energy consumption with mining and delivery is crucial. Returning to the mothership for recharge is essential.
- **Continuous Action Space**: Actions are continuous (thrust in x/y directions and mining), which may require algorithms suited for continuous control.
- **Complex Reward Structure**: The GOODREWARD pattern rewards energy efficiency, exploration, path optimization, and strategic behavior, in addition to immediate mining and delivery rewards.

## Choosing the Right Algorithm

- **PPO (Proximal Policy Optimization)**: Recommended for its stability and performance on continuous action spaces. It's the default choice in the provided training scripts.
- **SAC (Soft Actor-Critic)**: Good for continuous action spaces and can handle stochastic policies well, potentially useful for exploration in partial observability.
- **TD3 (Twin Delayed DDPG)**: Another option for continuous control, though it may require more tuning for this environment.
- **DQN**: Not recommended due to the continuous action space, unless discretized.

Example of setting up PPO with Stable-Baselines3:

```python
from stable_baselines3 import PPO
from space_mining import make_env

env = make_env()
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=1000000)
model.save('space_mining_ppo')
```

## Hyperparameter Tuning

Hyperparameters significantly impact training success. Start with these recommended settings for PPO with Stable-Baselines3:

- `learning_rate`: 0.0003 (adjust based on training stability; lower if unstable)
- `n_steps`: 2048 (number of steps per update; higher can improve sample efficiency but increases computation)
- `batch_size`: 64 (size of mini-batches for training)
- `gamma`: 0.99 (discount factor; high value for long-term reward focus)
- `gae_lambda`: 0.95 (lambda for advantage estimation; balances bias and variance)
- `clip_range`: 0.2 (PPO clipping parameter; controls policy update magnitude)
- `ent_coef`: 0.01 (entropy coefficient; encourages exploration)

Use tools like Optuna or Ray Tune for systematic hyperparameter optimization:

```python
import optuna
from stable_baselines3 import PPO

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    n_steps = trial.suggest_int('n_steps', 512, 4096)
    env = make_env()
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, verbose=0)
    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Training Duration

- **Expected Timesteps**: Typically, 1 to 3 million timesteps are needed for good performance, depending on the algorithm and hyperparameters.
- **Early Stopping**: Implement early stopping based on a performance threshold (e.g., average reward over 100 episodes) to avoid unnecessary computation.

## Environment Customization

Adjust environment parameters to balance difficulty and training feasibility:

- **Increase `observation_radius`**: Makes the environment easier by allowing the agent to see more asteroids (e.g., set to 20).
- **Decrease `grid_size`**: Reduces the search space (e.g., set to 60), though it may limit exploration challenges.
- **Adjust `max_episode_steps`**: Longer episodes (e.g., 1500) allow more time for resource collection, but may increase training time.
- **Modify `mining_range`**: A larger range (e.g., 10.0) makes mining easier.

Example:

```python
from space_mining import make_env

env = make_env(observation_radius=20, grid_size=60, max_episode_steps=1500, mining_range=10.0)
```

## Vectorized Environments

Use vectorized environments to parallelize data collection, significantly speeding up training:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from space_mining import make_env

n_envs = 8
env = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
```

## Callbacks for Monitoring and Saving

Implement callbacks to monitor training progress, save checkpoints, and apply early stopping:

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='ppo_model')
eval_env = make_env()
eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=5000, deterministic=True, render=False)

model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])
```

## Evaluation and Visualization

Regularly evaluate your agent's performance and visualize its behavior to diagnose issues:

- **Quantitative Evaluation**: Use `evaluate_policy` to get mean and standard deviation of rewards over multiple episodes.
  ```python
  from stable_baselines3.common.evaluation import evaluate_policy
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
  print(f'Mean reward: {mean_reward} +/- {std_reward}')
  ```
- **Qualitative Visualization**: Render episodes to observe agent behavior, especially after significant training progress.
  ```python
  env = make_env(render_mode='human')
  obs, _ = env.reset()
  for _ in range(1200):
      action, _ = model.predict(obs, deterministic=True)
      obs, _, terminated, truncated, _ = env.step(action)
      if terminated or truncated:
          obs, _ = env.reset()
  env.close()
  ```
- **GIF Generation**: Create GIFs to share or analyze agent behavior over time.
  ```python
  from space_mining.scripts.make_gif import generate_trajectory, save_gif
  frames = generate_trajectory('path_to_model.zip', num_steps=1200, deterministic=True)
  save_gif(frames, 'agent_performance.gif', fps=30)
  ```

## Handling Partial Observability

The limited observation radius introduces a partially observable Markov decision process (POMDP) challenge. Consider these strategies:

- **Recurrent Policies**: Use LSTM or GRU-based policies to maintain memory of past observations.
  ```python
  from stable_baselines3.common.policies import ActorCriticCnnPolicy
  model = PPO('MlpPolicy', env, policy_kwargs={'lstm_hidden_size': 128}, verbose=1)
  ```
- **Frame Stacking**: Stack multiple recent observations to provide temporal context.
  ```python
  from gymnasium.wrappers import FrameStack
  env = FrameStack(make_env(), num_stack=4)
  ```

## Reward Shaping

If the default reward structure isn't yielding desired behavior, consider subclassing `SpaceMiningEnv` to modify the reward function. For example, increase the reward for exploration if the agent isn't discovering enough asteroids:

```python
from space_mining.envs import SpaceMiningEnv

class CustomRewardEnv(SpaceMiningEnv):
    def compute_reward(self, action, observation, info):
        reward, reward_info = super().compute_reward(action, observation, info)
        # Increase exploration bonus
        reward_info['exploration_reward'] *= 2.0
        return reward + reward_info['exploration_reward'] - reward_info.get('exploration_reward', 0.0), reward_info
```

## Common Training Issues and Solutions

- **Agent Not Exploring Enough**: Increase the entropy coefficient (`ent_coef`) or exploration bonus in the reward function. Ensure the observation radius isn't too small.
- **Agent Dying Quickly**: Check if energy consumption rates are too high; consider reducing them or increasing recharge rewards. Adjust `max_episode_steps` if episodes are too short.
- **Unstable Training**: Lower the learning rate, increase the batch size, or adjust the clip range in PPO.
- **Slow Training**: Use more parallel environments with vectorized setups, or optimize hyperparameters to converge faster.
- **Agent Stuck at Boundaries**: Ensure boundary penalties are sufficient to discourage staying near edges; check if gravity towards the mothership is working as intended.

## Debugging Tips

- **Logging**: Add detailed logging to track energy levels, inventory, collisions, and rewards per step to identify failure points.
- **Render Debugging**: Set `render_mode='human'` during training to visually inspect agent behavior for a few episodes.
- **Reward Breakdown**: Log individual reward components (from `info` dictionary) to see which aspects the agent is optimizing or neglecting.

By following these tips and adjusting strategies based on observed performance, you can effectively train agents to excel in the SpaceMining environment. For further assistance, refer to other documentation or raise issues on the project's GitHub repository.
