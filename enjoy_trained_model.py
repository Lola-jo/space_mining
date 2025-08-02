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

import argparse
import numpy as np
import sys
from pathlib import Path

# Add environment path
sys.path.append('./2025-07-29-12-39/code/iteration_1/sample_2/env_code')
from env import make_env

from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model: {model_path}")
    
    # Determine algorithm type based on filename
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "SAC" in model_path:
        model = SAC.load(model_path)
    elif "DDPG" in model_path:
        model = DDPG.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    else:
        # Default to PPO
        model = PPO.load(model_path)
    
    return model

def create_env_with_render():
    """Create environment with rendering"""
    def make_env_wrapper():
        env = make_env(render_mode="rgb_array")
        env = Monitor(env)
        return env
    
    return make_env_wrapper()

def generate_gif(model, env, gif_name, num_episodes=3, fps=30, max_steps=200):
    """Generate GIF"""
    try:
        import imageio
        import os
    except ImportError:
        print("Need to install imageio: pip install imageio")
        return
    
    print(f"Generating GIF: {gif_name}")
    print(f"Parameters: {num_episodes} episodes, {fps} FPS, max {max_steps} steps")
    
    frames = []
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset(seed=episode)
        episode_frames = []
        step_count = 0
        
        while step_count < max_steps:
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render current frame
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Add frames from this episode
        frames.extend(episode_frames)
        print(f"Episode {episode + 1} completed, {len(episode_frames)} frames")
    
    # Save GIF
    if frames:
        imageio.mimsave(gif_name, frames, fps=fps)
        print(f"GIF saved: {gif_name}")
        print(f"Total frames: {len(frames)}")
    else:
        print("No frames to save")

def generate_gif_with_annotations(model, env, gif_name, num_episodes=3, fps=30, max_steps=200):
    """Generate GIF with annotations"""
    try:
        import imageio
        import os
        import pygame
        import pygame.freetype
        from datetime import datetime
    except ImportError:
        print("Need to install imageio and pygame: pip install imageio pygame")
        return
    
    print(f"Generating annotated GIF: {gif_name}")
    print(f"Parameters: {num_episodes} episodes, {fps} FPS, max {max_steps} steps")
    
    # Generate separate GIF for each episode
    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset(seed=episode)
        episode_frames = []
        step_count = 0
        
        # Track behavior state
        last_action = None
        last_reward = 0
        behavior_text = ""
        behavior_duration = 0
        debug_messages = []
        
        while step_count < max_steps:
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Analyze behavior and debug information
            behavior_text, behavior_duration, debug_messages = analyze_behavior_detailed(
                action, reward, info, last_action, last_reward, step_count
            )
            
            # Render current frame
            frame = env.render()
            if frame is not None:
                # Add annotations to frame
                annotated_frame = add_detailed_annotation_to_frame(
                    frame, behavior_text, step_count, reward, info, debug_messages
                )
                episode_frames.append(annotated_frame)
            
            last_action = action
            last_reward = reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Generate separate GIF for this episode
        if episode_frames:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_gif_name = f"{gif_name.replace('.gif', '')}_episode_{episode+1}_{timestamp}.gif"
            
            imageio.mimsave(episode_gif_name, episode_frames, fps=fps)
            print(f"Episode {episode + 1} GIF saved: {episode_gif_name}")
            print(f"Frame count: {len(episode_frames)}")
        else:
            print(f"Episode {episode + 1} no frames to save")

def analyze_behavior_detailed(action, reward, info, last_action, last_reward, step_count):
    """Detailed analysis of agent behavior and debug information"""
    behavior_text = ""
    duration = 0
    debug_messages = []
    
    # Get environment state
    agent_energy = info.get('agent_energy', 150)
    agent_inventory = info.get('agent_inventory', 0)
    agent_position = info.get('agent_position', [0, 0])  # 2D position
    mothership_pos = info.get('mothership_pos', [50, 50])  # 2D position
    collision_count = info.get('collision_count', 0)
    steps_count = info.get('steps_count', step_count)
    mining_asteroid_id = info.get('mining_asteroid_id', None)
    
    # Calculate distance to mothership
    distance_to_mothership = np.linalg.norm(np.array(agent_position) - np.array(mothership_pos))
    
    # Check energy usage
    if last_reward != 0 and agent_energy < 150:
        energy_used = 150 - agent_energy
        debug_messages.append(f"Energy used: {energy_used:.2f}")
        debug_messages.append(f"Remaining: {agent_energy:.2f}")
    
    # Check collision
    if reward < -10:
        behavior_text = "COLLISION!"
        duration = 20
        debug_messages.append(f"Collision! Total: {collision_count}")
    
    # Check too many collisions
    if collision_count >= 8:
        behavior_text = "TOO MANY COLLISIONS"
        duration = 30
        debug_messages.append("Episode terminated due to collisions")
    
    # Check mining behavior
    elif action[2] > 0.5:  # Mining action (2D environment, index 2 is mine)
        if mining_asteroid_id is not None:
            behavior_text = f"MINING ASTEROID {mining_asteroid_id}"
            duration = 15
            debug_messages.append(f"Continuously mining asteroid {mining_asteroid_id}")
        else:
            behavior_text = "STARTING MINING"
            duration = 10
            debug_messages.append("Starting to mine asteroid...")
        
        if agent_inventory < 100:  # Updated to new max inventory
            debug_messages.append("Mining in progress...")
        else:
            debug_messages.append("Inventory full, cannot mine")
    
    # Check delivery to mothership
    elif reward > 5 and agent_inventory == 0 and last_reward <= 0:
        behavior_text = "DELIVERED TO MOTHERSHIP"
        duration = 25
        debug_messages.append("Resources delivered to mothership")
        debug_messages.append("Inventory cleared")
    
    # Check returning to mothership
    elif agent_inventory > 0 and distance_to_mothership < 10:
        behavior_text = "RETURNING TO MOTHERSHIP"
        duration = 15
        debug_messages.append(f"Carrying {agent_inventory:.1f} resources")
        debug_messages.append("Approaching mothership")
    
    # Check carrying resources
    elif agent_inventory > 0:
        behavior_text = "CARRYING RESOURCES"
        duration = 8
        debug_messages.append(f"Carrying {agent_inventory:.1f} resources")
    
    # Check movement
    elif np.any(np.abs(action[:2]) > 0.1):  # 2D movement (fx, fy)
        behavior_text = "MOVING"
        duration = 5
        debug_messages.append("Moving to target")
    
    # Check low energy
    elif agent_energy < 30:
        behavior_text = "LOW ENERGY"
        duration = 10
        debug_messages.append(f"Low energy: {agent_energy:.1f}")
    
    # Check energy depleted
    elif agent_energy <= 0:
        behavior_text = "ENERGY DEPLETED"
        duration = 30
        debug_messages.append("Energy depleted!")
    
    # Check exploration
    else:
        behavior_text = "EXPLORING"
        duration = 3
        debug_messages.append("Exploring environment")
    
    # Add position information
    debug_messages.append(f"Pos: ({agent_position[0]:.1f}, {agent_position[1]:.1f})")  # 2D position
    debug_messages.append(f"Distance to mothership: {distance_to_mothership:.1f}")
    
    return behavior_text, duration, debug_messages

def add_detailed_annotation_to_frame(frame, behavior_text, step_count, reward, info, debug_messages):
    """Add detailed annotations to frame"""
    # Return original frame without any annotations
    return frame

def main():
    parser = argparse.ArgumentParser(description='Generate GIF for trained space mining model')
    parser.add_argument('--model_path', type=str, 
                       default='./2025-07-29-12-39/code/iteration_1/sample_2/train_20250730_200654/final_model.zip',
                       help='Model file path')
    parser.add_argument('--gif_name', type=str, 
                       default='./2025-07-29-12-39/code/iteration_1/sample_2/train_20250730_200654/space_mining_demo.gif',
                       help='GIF filename')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to record')
    parser.add_argument('--fps', type=int, default=30,
                       help='GIF frame rate')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--live', action='store_true',
                       help='Real-time display environment (no GIF generation)')
    parser.add_argument('--no_annotations', action='store_true',
                       help='No behavior annotations')
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file does not exist: {model_path}")
        print("Please run training script first to generate model")
        return
    
    # Ensure GIF output directory exists
    gif_dir = Path(args.gif_name).parent
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(str(model_path))
    
    # Create environment
    if args.live:
        # Real-time display mode
        env = make_env(render_mode="human")
        env = Monitor(env)
        
        print("Real-time display mode - Press any key to continue...")
        
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1} ===")
            obs, info = env.reset(seed=episode)
            step_count = 0
            
            while step_count < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                step_count += 1
                print(f"Step {step_count}: reward={reward:.2f}")
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1} completed, total steps: {step_count}")
        
        env.close()
        
    else:
        # GIF generation mode
        env = create_env_with_render()
        
        if args.no_annotations:
            generate_gif(
                model=model,
                env=env,
                gif_name=args.gif_name,
                num_episodes=args.episodes,
                fps=args.fps,
                max_steps=args.max_steps
            )
        else:
            generate_gif_with_annotations(
                model=model,
                env=env,
                gif_name=args.gif_name,
                num_episodes=args.episodes,
                fps=args.fps,
                max_steps=args.max_steps
            )
        
        env.close()

if __name__ == "__main__":
    main() 