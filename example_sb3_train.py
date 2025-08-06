#!/usr/bin/env python3
"""
SpaceMining Environment Training Script
Train custom SpaceMiningEnv using stable-baselines3
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from space_mining.envs import make_env

def train_ppo(total_timesteps=100000, output_dir='train_output'):
    os.makedirs(output_dir, exist_ok=True)
    
    env = DummyVecEnv([lambda: Monitor(make_env())])
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=os.path.join(output_dir, 'tensorboard_logs')
    )
    
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, 'best_model'),
        log_path=os.path.join(output_dir, 'logs'),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(output_dir, 'checkpoints'),
        name_prefix='ppo_model'
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    model.save(os.path.join(output_dir, 'final_model'))
    
    return model

if __name__ == '__main__':
    train_ppo()