#!/usr/bin/env python3
"""
SpaceMining Environment Enjoy Script
Run a trained model in SpaceMiningEnv and generate a GIF
"""

import os
import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from space_mining.envs import make_env

# Setup virtual display for headless rendering
try:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(800, 800))
    display.start()
    print("Virtual display started")
except ImportError:
    print("pyvirtualdisplay not installed, rendering might fail if no display is available")

def enjoy_ppo(model_path='train_output/final_model', output_gif='enjoy_output/space_mining.gif', episodes=10, max_steps=1000):
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    
    env = DummyVecEnv([lambda: Monitor(make_env(render_mode='rgb_array'))])
    model = PPO.load(model_path)
    
    frames = []
    for episode in range(episodes):
        obs = env.reset()
        for step in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
            if done:
                break
    env.close()
    
    # Save as GIF
    imageio.mimsave(output_gif, frames, fps=30)
    print(f'GIF saved as {output_gif}')

if __name__ == '__main__':
    enjoy_ppo()
    # Stop virtual display if it was started
    try:
        display.stop()
        print("Virtual display stopped")
    except NameError:
        pass