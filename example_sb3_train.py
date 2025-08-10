"""
Example script to train a PPO model on SpaceMiningEnv using Stable Baselines3.
This is a simple wrapper around the existing train_ppo functionality.
"""

from space_mining.agents.train_ppo import train_ppo

if __name__ == '__main__':
    # Simple example of training PPO on SpaceMining
    print("Starting PPO training on SpaceMining environment...")

    model = train_ppo(
        total_timesteps=1000000, 
        output_dir="train_output"  # Adjust as needed
    )

    print("Training completed! Model saved in 'example_training_output/final_model'")
