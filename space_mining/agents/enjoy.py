#!/usr/bin/env python3
# ==============================================================================
# File Description: Evaluate a trained space mining model and generate a demo GIF
# 
# Instructions:
# Before running the script, please configure the following:
#
# 1. Environment: Ensure the 'env' file is set up correctly for the execution environment.
# 2. Model Path: Specify the path to the trained model file using the --model_path argument.
# 3. GIF Output: Define the output filename and path for the generated GIF using --gif_name.
#
# ==============================================================================

import os
import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO
from space_mining.envs import make_env

def generate_gif(model_path, gif_path, episodes=1, max_steps=1000, fps=30):
    model = PPO.load(model_path)
    env = make_env(render_mode='rgb_array')
    
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
            if terminated or truncated:
                break
    
    imageio.mimsave(gif_path, frames, fps=fps)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gif_path', default='demo.gif')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    generate_gif(args.model_path, args.gif_path, args.episodes, args.max_steps, args.fps) 