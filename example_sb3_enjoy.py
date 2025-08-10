#!/usr/bin/env python3
"""
Example script to enjoy a trained PPO model on SpaceMiningEnv and generate a GIF.
"""

import os
import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO
from space_mining.envs import make_env

def enjoy_ppo(model_path, output_gif='enjoy_output/space_mining.gif', episodes=1, max_steps=1000, fps=30, device='cpu'):
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    
    model = PPO.load(model_path, device=device)
    env = make_env(render_mode='rgb_array')
    
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_frames = []
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            if terminated or truncated:
                break
        frames.extend(episode_frames)
        print(f"Episode {ep+1} completed in {len(episode_frames)} steps")
    
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f'GIF saved as {output_gif}')
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enjoy trained PPO on SpaceMiningEnv')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_gif', type=str, default='enjoy_output/space_mining.gif', help='Output GIF path')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=30, help='GIF frames per second')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference (default: cpu)')
    args = parser.parse_args()
    
    enjoy_ppo(args.model_path, args.output_gif, args.episodes, args.max_steps, args.fps, args.device)