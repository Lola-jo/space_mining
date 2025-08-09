#!/usr/bin/env python3
"""Script to render an episode of the SpaceMining environment using a trained PPO model.
Can be run as a CLI script to visualize agent behavior.
"""
import argparse

from space_mining import make_env, PPOAgent

def render_episode(model_path: str, max_steps: int = 1000) -> None:
    """Render an episode of the SpaceMining environment using a trained PPO model.

    Args:
        model_path (str): Path to the trained PPO model checkpoint.
        max_steps (int): Maximum number of steps to run the episode for.
    """
    # Load agent
    env = make_env(render_mode='human', max_episode_steps=max_steps)
    agent = PPOAgent.load(model_path, env=env)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run episode with rendering
    for _ in range(max_steps):
        action = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
    
    env.close()
    print(f"Episode rendering completed. Total steps: {min(max_steps, _ + 1)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render an episode of SpaceMining environment using a trained PPO model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained PPO model checkpoint")
    parser.add_argument('--max_steps', type=int, default=1000, help="Maximum number of steps to run the episode")
    args = parser.parse_args()
    
    render_episode(args.model_path, args.max_steps)
