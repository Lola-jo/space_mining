"""Example script demonstrating how to use the SpaceMining environment and PPO agent."""
import gymnasium as gym
from space_mining import make_env, PPOAgent

def run_example():
    """Run a simple example with the SpaceMining environment."""
    # Create the environment
    env = make_env(render_mode="human", max_episode_steps=1200)

    # Reset the environment
    obs, info = env.reset()
    total_reward = 0

    # Run for a few steps (random actions for demonstration)
    for _ in range(500):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0

    env.close()

def run_trained_agent(checkpoint_path):
    """Run a trained PPO agent on the SpaceMining environment.

    Args:
        checkpoint_path (str): Path to the trained PPO model checkpoint.
    """
    # Create the environment
    env = make_env(render_mode="human", max_episode_steps=1200)

    # Load the trained agent
    agent = PPOAgent.load(checkpoint_path, env=env)

    # Reset the environment
    obs, info = env.reset()
    total_reward = 0

    # Run the agent
    for _ in range(1200):
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0

    env.close()

if __name__ == "__main__":
    print("Running random agent example...")
    run_example()

    # Uncomment the following lines to run a trained agent
    # checkpoint_path = "./checkpoints/final_model"
    # print(f"Running trained agent from {checkpoint_path}...")
    # run_trained_agent(checkpoint_path) 