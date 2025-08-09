#!/usr/bin/env python3
import argparse
from stable_baselines3 import PPO
from space_mining.envs import make_env

def render_episode(model_path, max_steps=1000):
    model = PPO.load(model_path)
    env = make_env(render_mode='human')
    
    obs, _ = env.reset()
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_steps', type=int, default=1000)
    args = parser.parse_args()
    
    render_episode(args.model_path, args.max_steps)
