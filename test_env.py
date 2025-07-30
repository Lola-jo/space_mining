import sys
import os
import numpy as np

# Add the environment code path to Python path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'env_code')))

from env import SpaceMiningEnv, make_env
from stable_baselines3 import PPO

def test_space_mining_env():
    """Test basic functionality of the SpaceMining environment"""
    print("Creating SpaceMining environment...")
    
    # Create environment instance
    env = SpaceMiningEnv(
        grid_size=80,
        max_asteroids=12,
        max_resource_per_asteroid=40,
        observation_radius=15,
        render_mode="human"  # Set to None to disable rendering, "human" to show window
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    print("\nResetting environment...")
    observations, info = env.reset(seed=42)
    print(f"Initial info: {info}")
    
    # Run several random steps
    print("\nExecuting 10 random steps:")
    for step in range(10):
        # Generate random action for the agent
        action = env.action_space.sample()
        
        # Execute step
        observations, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Agent inventory: {info.get('agent_inventory', 0):.2f}")
        print(f"  Agent energy: {info.get('agent_energy', 0):.2f}")
        print(f"  Collision count: {info.get('collision_count', 0)}")
        
        # Check if episode is terminated
        if terminated or truncated:
            print("\nEpisode terminated, stopping simulation")
            break
    
    print("\nClosing environment...")
    env.close()
    print("Test completed!")

if __name__ == "__main__":
    test_space_mining_env()

# Test environment creation with make_env wrapper
print("\n=== Testing make_env wrapper ===")
env = make_env()
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test if SB3 can accept this environment
print("\n=== Testing SB3 compatibility ===")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)
print("Training successful!") 