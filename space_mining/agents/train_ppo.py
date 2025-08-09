"""Training script for PPO agent in space_mining environment.
Can be run as a CLI script or imported as a function.
"""
import argparse
import os
from stable_baselines3 import PPO
from space_mining import make_env

def train_ppo(
    output_dir="./checkpoints",
    total_timesteps=3000000,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    render_mode=None
):
    """Train a PPO model on the SpaceMining environment.

    Args:
        output_dir (str): Directory to save checkpoints and logs.
        total_timesteps (int): Total number of timesteps to train for.
        learning_rate (float): Learning rate for the PPO optimizer.
        n_steps (int): Number of steps to run per update.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        gae_lambda (float): Lambda for Generalized Advantage Estimation.
        clip_range (float): Clipping parameter for PPO.
        verbose (int): Verbosity level.
        render_mode (str): Render mode for the environment (None, 'human', 'rgb_array').

    Returns:
        PPO: Trained PPO model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    env = make_env(render_mode=render_mode, max_episode_steps=1200)

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=verbose,
        tensorboard_log=os.path.join(output_dir, "tensorboard_logs")
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    return model

def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on SpaceMining environment.")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--total-timesteps", type=int, default=3000000,
                        help="Total number of timesteps to train for")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="Learning rate for PPO optimizer")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps to run per update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="Clipping parameter for PPO")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    parser.add_argument("--render-mode", type=str, default=None,
                        choices=[None, "human", "rgb_array"],
                        help="Render mode for the environment")

    args = parser.parse_args()

    train_ppo(
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=args.verbose,
        render_mode=args.render_mode
    )

if __name__ == "__main__":
    main()
