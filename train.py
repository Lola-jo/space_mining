#!/usr/bin/env python3
"""
SpaceMining Environment Training Script
Train custom SpaceMiningEnv using stable-baselines3
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Add environment path
sys.path.append('./space_mining/env_code')
from env import make_env

def create_env():
    """Create environment"""
    def _make_env():
        env = make_env()
        env = Monitor(env)  # Add monitoring
        return env
    
    return _make_env

def train_space_mining():
    """Train SpaceMining environment"""
    
    # Determine experiment path and output directory
    experiment_path = './space_mining'
    env_code_path = f'{experiment_path}/env_code'
    
    # Create training output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{experiment_path}/train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Experiment Information ===")
    print(f"Experiment path: {experiment_path}")
    print(f"Environment code: {env_code_path}")
    print(f"Training output: {output_dir}")
    print(f"Training timestamp: {timestamp}")
    
    # Create environment
    env = create_env()()
    
    print("\n=== Environment Information ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Environment: {env}")
    
    # Test environment
    print("\n=== Testing Environment ===")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs.shape}")
    
    # Test several random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break
    
    env.close()
    
    # Create vectorized environment
    env = DummyVecEnv([create_env() for _ in range(1)])  # Single environment training
    
    # Select algorithm (can choose different algorithms)
    algorithm = "PPO"  # Options: "PPO", "SAC", "DDPG", "TD3"
    
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    elif algorithm == "DDPG":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[400, 300]),
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    
    # 创建回调函数
    eval_env = DummyVecEnv([create_env() for _ in range(1)])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best_model",
        log_path=f"{output_dir}/logs",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{output_dir}/checkpoints",
        name_prefix=f"{algorithm}_model"
    )
    
    # 创建组件记录回调
    class ComponentLoggerCallback(BaseCallback):
        def __init__(self, save_path, log_freq=1200, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
            self.log_freq = log_freq
            self.component_data = []
            self.step_count = 0

        def _on_step(self):
            self.step_count += 1
            if self.step_count % self.log_freq == 0:
                # Get current environment state
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                
                # Calculate fair comparison metrics
                episode_progress = 0.0
                resource_progress = 0.0
                fitness_score = 0.0
                
                if hasattr(env, 'steps_count') and hasattr(env, 'max_episode_steps'):
                    episode_progress = env.steps_count / env.max_episode_steps
                
                if hasattr(env, 'asteroid_resources') and hasattr(env, 'max_resource_per_asteroid') and hasattr(env, 'max_asteroids'):
                    total_initial_resources = env.max_resource_per_asteroid * env.max_asteroids
                    remaining_resources = float(np.sum(env.asteroid_resources))
                    resource_progress = 1.0 - (remaining_resources / total_initial_resources) if total_initial_resources > 0 else 0.0
                
                # Calculate fitness score - important evaluation metric
                if hasattr(env, 'compute_fitness_score'):
                    fitness_score = float(env.compute_fitness_score())
                
                component_info = {
                    "step": self.step_count,
                    "timestamp": datetime.now().isoformat(),
                    "agent_position": env.agent_position.tolist() if hasattr(env, 'agent_position') else [0, 0, 0],
                    "agent_energy": float(env.agent_energy) if hasattr(env, 'agent_energy') else 0.0,
                    "agent_inventory": float(env.agent_inventory) if hasattr(env, 'agent_inventory') else 0.0,
                    "collision_count": int(env.collision_count) if hasattr(env, 'collision_count') else 0,
                    "asteroid_count": len(env.asteroid_positions) if hasattr(env, 'asteroid_positions') else 0,
                    "remaining_resources": float(np.sum(env.asteroid_resources)) if hasattr(env, 'asteroid_resources') else 0.0,
                    "distance_to_mothership": float(np.linalg.norm(env.agent_position - env.mothership_pos)) if hasattr(env, 'agent_position') and hasattr(env, 'mothership_pos') else 0.0,
                    "obstacle_count": len(env.obstacle_positions) if hasattr(env, 'obstacle_positions') else 0,
                    "episode_progress": episode_progress,  # Fair comparison metric
                    "resource_progress": resource_progress,  # Fair comparison metric
                    "fitness_score": fitness_score,  # Important evaluation metric
                    "episode_step": env.steps_count if hasattr(env, 'steps_count') else 0,
                    "max_episode_steps": env.max_episode_steps if hasattr(env, 'max_episode_steps') else 0
                }
                self.component_data.append(component_info)
                import json
                with open(f"{self.save_path}/component_logs.json", 'w') as f:
                    json.dump(self.component_data, f, indent=2)
                print(f"Component data recorded (step {self.step_count}): {len(self.component_data)} records, Fitness Score: {fitness_score:.2f}")
            return True
    
    component_logger = ComponentLoggerCallback(output_dir)
    
    # Start training
    print(f"\n=== Starting {algorithm} training ===")
    print(f"Training logs saved in: {output_dir}")
    
    try:
        model.learn(
            total_timesteps=3000000,  # Increased training steps to 3 million
            callback=[eval_callback, checkpoint_callback, component_logger],
            progress_bar=True
        )
        
        # Save final model
        model.save(f"{output_dir}/final_model")
        print(f"Training completed! Model saved in: {output_dir}/final_model")
        
        # Save training information
        training_info = {
            "experiment_path": experiment_path,
            "env_code_path": env_code_path,
            "training_timestamp": timestamp,
            "algorithm": algorithm,
            "total_timesteps": 3000000,
            "model_path": f"{output_dir}/final_model",
            "best_model_path": f"{output_dir}/best_model/best_model",
            "component_logs_path": f"{output_dir}/component_logs.json",
            "metrics_description": {
                "fitness_score": "Comprehensive performance score including resource collection, energy management, path efficiency, etc.",
                "episode_progress": "Current episode progress (0-1)",
                "resource_progress": "Resource collection progress (0-1)",
                "agent_inventory": "Current carried resources",
                "agent_energy": "Current energy level",
                "collision_count": "Collision count",
                "remaining_resources": "Total remaining resources"
            }
        }
        
        import json
        with open(f"{output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training information saved in: {output_dir}/training_info.json")
        print(f"Component logs saved in: {output_dir}/component_logs.json")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving current model...")
        model.save(f"{output_dir}/interrupted_model")
        print(f"Interrupted model saved in: {output_dir}/interrupted_model")
    
    env.close()
    eval_env.close()

def evaluate_model(model_path, num_episodes=5):
    """Evaluate trained model"""
    print(f"\n=== Evaluating model: {model_path} ===")
    
    # Load model
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "SAC" in model_path:
        model = SAC.load(model_path)
    elif "DDPG" in model_path:
        model = DDPG.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    else:
        model = PPO.load(model_path)  # Default
    
    # Create environment
    env = create_env()()
    
    total_rewards = []
    episode_lengths = []
    fitness_scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        episode_fitness_scores = []
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Record fitness score
            if 'fitness_score' in info:
                episode_fitness_scores.append(info['fitness_score'])
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate average fitness score
        avg_fitness = np.mean(episode_fitness_scores) if episode_fitness_scores else 0.0
        fitness_scores.append(avg_fitness)
        
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}, avg_fitness={avg_fitness:.2f}")
    
    env.close()
    
    print(f"\nEvaluation results:")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Fitness Score: {np.mean(fitness_scores):.2f} ± {np.std(fitness_scores):.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SpaceMining environment")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", 
                       help="Training mode or evaluation mode")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Model path to evaluate")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()

    if args.mode == "train":
        train_space_mining()
    elif args.mode == "evaluate":
        if args.model_path is None:
            print("Error: Evaluation mode requires model path")
            exit(1)
        evaluate_model(args.model_path, args.episodes)