import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from envs.space_mining.env_code.env_1 import SpaceMiningEnv
from envs.space_mining.env_code.reward_wrapper import RewardFunctionWrapper
from envs.space_mining.env_code import reward_functions
import imageio

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_reward_fn():
    # 这里假设你用的是 sample_compute_reward
    return reward_functions.sample_compute_reward

def make_env_fn(env_kwargs, reward_fn):
    def _init():
        env = SpaceMiningEnv(
            max_episode_steps=env_kwargs.get('max_episode_steps', 1600),
            grid_size=env_kwargs.get('grid_size', 100),
            max_asteroids=env_kwargs.get('max_asteroids', 15),
            max_resource_per_asteroid=env_kwargs.get('max_resource_per_asteroid', 80),
            observation_radius=env_kwargs.get('observation_radius', 25),
            render_mode=env_kwargs.get('render_mode', None)
        )
        env = RewardFunctionWrapper(env, reward_fn)
        return env
    return _init

def train_and_save(config_path="envs/space_mining/space_mining_ollama.yml"):
    # 1. 读取配置
    config = load_config(config_path)
    env_cfg = config['environment']
    rl_cfg = config['rl']
    algo_params = rl_cfg['algo_params']
    train_cfg = rl_cfg['training']
    num_envs = train_cfg.get('num_envs', 1)
    total_timesteps = train_cfg.get('total_timesteps', 1_000_000)
    seed = train_cfg.get('seed', 42)
    log_dir = config.get('logging', {}).get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)

    # 2. 构建奖励函数
    reward_fn = get_reward_fn()

    # 3. 构建环境
    env_kwargs = env_cfg.get('kwargs', {})
    env_kwargs['max_episode_steps'] = env_cfg.get('max_episode_steps', 1600)
    # 多进程环境
    if num_envs > 1:
        env = SubprocVecEnv([make_env_fn(env_kwargs, reward_fn) for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env_fn(env_kwargs, reward_fn)])

    # 4. 训练
    model = PPO(
        policy=algo_params['policy'],
        env=env,
        verbose=1,
        seed=seed,
        **{k: v for k, v in algo_params.items() if k != 'policy'}
    )
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(log_dir, "ppo_space_mining_custom_reward")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 5. 评估并保存GIF
    if config['rl']['evaluation'].get('save_gif', True):
        save_gif_of_model(model_path, env_kwargs, reward_fn, config)

def save_gif_of_model(model_path, env_kwargs, reward_fn, config):
    # 单环境评估
    env_kwargs = env_kwargs.copy()
    env_kwargs['render_mode'] = 'rgb_array'
    env = make_env_fn(env_kwargs, reward_fn)()
    model = PPO.load(model_path, env=env)
    num_episodes = config['rl']['evaluation'].get('num_episodes', 5)
    gif_frames = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        frames = []
        while not (done or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        gif_frames.extend(frames)
    gif_path = os.path.splitext(model_path)[0] + ".gif"
    imageio.mimsave(gif_path, gif_frames, fps=config['rl']['evaluation'].get('gif_fps', 30))
    print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    train_and_save()