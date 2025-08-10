"""
Example script to enjoy a trained PPO model on SpaceMiningEnv and generate a GIF.
This is a simple wrapper around the existing make_gif functionality.
"""

from space_mining.scripts.make_gif import generate_trajectory, save_gif

if __name__ == '__main__':
    # Example: Generate GIF from a trained model
    print("Generating GIF from trained SpaceMining model...")

    # You would need to have a trained model first
    model_path = "train_output/final_model.zip"

    # For demonstration, we'll show the function call
    frames = generate_trajectory(
        checkpoint_path=model_path,
        num_steps=1200,
        deterministic=True
    )
    save_gif(frames, "output_gif/final_model.gif", fps=30)

