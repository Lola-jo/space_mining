"""Script to generate GIFs from SpaceMining environment trajectories or checkpoints.
Can be run as a CLI script or imported as a function.
"""
import argparse
import os
import numpy as np
from PIL import Image
import gymnasium as gym
from space_mining import make_env, PPOAgent

def save_gif(frames, output_path, fps=30):
    """Save a sequence of frames as a GIF.

    Args:
        frames (list): List of frames (numpy arrays or PIL Images).
        output_path (str): Path to save the GIF file.
        fps (int): Frames per second for the GIF.
    """
    # Convert frames to PIL Images if they are numpy arrays
    frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"GIF saved to {output_path}")

def generate_trajectory(checkpoint_path, num_steps=1200, render_mode="rgb_array", deterministic=True, device="cpu"):
    """Generate a trajectory from a checkpoint.

    Args:
        checkpoint_path (str): Path to the PPO checkpoint file.
        num_steps (int): Number of steps to run the trajectory for.
        render_mode (str): Render mode for the environment.
        deterministic (bool): Whether to use deterministic predictions.
        device (str): Device to use ('cpu' or 'cuda', default: 'cpu').

    Returns:
        list: List of frames from the trajectory.
    """
    # Create environment
    env = make_env(render_mode=render_mode, max_episode_steps=num_steps)

    # Load agent
    agent = PPOAgent.load(checkpoint_path, env=env, device=device)

    # Reset environment
    obs, _ = env.reset()
    frames = [env.render()]

    # Run trajectory
    for _ in range(num_steps):
        action = agent.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    return frames

def main():
    """Main function to parse arguments and generate GIF."""
    parser = argparse.ArgumentParser(description="Generate a GIF from a SpaceMining PPO checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the PPO checkpoint file")
    parser.add_argument("--output", type=str, default="output.gif",
                        help="Path to save the output GIF")
    parser.add_argument("--steps", type=int, default=1200,
                        help="Number of steps to run for the trajectory")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the GIF")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic predictions")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for inference (default: cpu)")

    args = parser.parse_args()

    # Generate trajectory
    frames = generate_trajectory(
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        deterministic=args.deterministic,
        device=args.device
    )

    # Save as GIF
    save_gif(frames, args.output, fps=args.fps)

if __name__ == "__main__":
    main()
