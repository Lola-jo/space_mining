from pathlib import Path
import argparse
import yaml
import time
import numpy as np

from stable_eureka.logger import get_logger
from stable_eureka.utils import make_env
from stable_eureka.rl_evaluator import RLEvaluator
from custom_rl_evaluator import CustomRLEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model and generate GIFs for Space Mining environment')
    parser.add_argument('--exp_path', type=str, required=True, help='Path to experiment directory')
    parser.add_argument('--iteration', type=int, default=None, help='Iteration number (optional)')
    parser.add_argument('--sample', type=int, default=None, help='Sample number (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for evaluation')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--benchmark', action='store_true', help='Use benchmark model instead of trained model')
    parser.add_argument('--gif_name', type=str, default=None, help='Custom name for the GIF file')
    parser.add_argument('--gif_fps', type=int, default=30, help='Frames per second for the GIF')
    parser.add_argument('--render_mode', type=str, default="rgb_array", choices=["rgb_array", "human"], 
                        help='Render mode ("rgb_array" for GIF, "human" for live viewing)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum steps per episode (overrides config)')
    parser.add_argument('--skip_frames', type=int, default=2, help='Skip every N frames in the GIF to reduce size')
    
    args = parser.parse_args()
    
    # Set up experiment path
    exp_path = Path(args.exp_path)
    if not exp_path.exists():
        raise ValueError(f"Experiment path {exp_path} does not exist")
    
    # Load configuration
    config_path = exp_path / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config file {config_path} not found")
    
    config = yaml.safe_load(open(config_path, 'r'))
    
    # Determine model path
    if args.benchmark:
        model_path = exp_path / 'code' / 'benchmark' / 'model.zip'
        print(f"Using benchmark model at: {model_path}")
    elif args.iteration is not None and args.sample is not None:
        model_path = exp_path / 'code' / f'iteration_{args.iteration}' / f'sample_{args.sample}' / 'model.zip'
        print(f"Using model from iteration {args.iteration}, sample {args.sample} at: {model_path}")
    else:
        # Try to find best model from best_reward.json
        best_reward_path = exp_path / 'code' / 'best_reward.json'
        if best_reward_path.exists():
            import json
            with open(best_reward_path, 'r') as f:
                best_info = json.load(f)
            
            if 'iteration' in best_info and 'sample' in best_info:
                best_iteration = best_info['iteration']
                best_sample = best_info['sample']
                model_path = exp_path / 'code' / f'iteration_{best_iteration}' / f'sample_{best_sample}' / 'model.zip'
                print(f"Using best model from iteration {best_iteration}, sample {best_sample} at: {model_path}")
            else:
                raise ValueError("Could not determine best model from best_reward.json")
        else:
            raise ValueError("Either specify --iteration and --sample, or --benchmark, or ensure best_reward.json exists")
    
    if not model_path.exists():
        raise ValueError(f"Model file {model_path} not found")
    
    # Determine environment name from config
    env_name = None
    if 'environment' in config:
        if 'name' in config['environment'] and config['environment']['name'] == 'space_mining':
            # Import specific module to ensure registration works
            import sys
            from gymnasium.envs.registration import register
            
            # Register the environment from the specific path for best model
            if args.benchmark:
                # Use benchmark environment
                if 'benchmark' in config['environment'] and config['environment']['benchmark']:
                    env_name = config['environment']['benchmark']
                else:
                    # If no benchmark specified, use SpaceMiningEnv directly
                    from envs.space_mining.env_code.env import SpaceMiningEnv, make_env
                    env_name = 'SpaceMiningEnv-v0'
                    # Register the environment
                    register(
                        id=env_name,
                        entry_point='envs.space_mining.env_code.env:make_env',
                        max_episode_steps=args.max_steps or config['environment']['max_episode_steps']
                    )
            else:
                # Get module path for specific iteration/sample
                if args.iteration is None:
                    iteration = best_iteration
                    sample = best_sample
                else:
                    iteration = args.iteration
                    sample = args.sample
                
                # Use the module path for the specific iteration/sample
                module_name = f"{config['experiment']['parent']}.{config['experiment']['name']}"
                if config['experiment']['use_datetime']:
                    exp_datetime = exp_path.name
                    module_name += f".{exp_datetime}"
                
                module_name += f".code.iteration_{iteration}.sample_{sample}.env_code.env"
                env_name = f'iteration_{iteration}_sample_{sample}_env-v0'
                
                # Try to register the environment
                try:
                    register(
                        id=env_name,
                        entry_point=f"{module_name}:make_env",
                        max_episode_steps=args.max_steps or config['environment']['max_episode_steps']
                    )
                except Exception as e:
                    print(f"Warning: Could not register environment: {e}")
                    print("Trying to use direct import...")
                    
                    # Get the specific environment path
                    env_path = exp_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}' / 'env_code' / 'env.py'
                    
                    if env_path.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("env_module", env_path)
                        env_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(env_module)
                        
                        env_name = 'SpaceMiningEnv-v0'
                        register(
                            id=env_name,
                            entry_point='env_module:make_env',
                            max_episode_steps=args.max_steps or config['environment']['max_episode_steps']
                        )
        else:
            # For standard Gym environments
            env_name = config['environment'].get('benchmark', 'SpaceMiningEnv-v0')
    
    if env_name is None:
        raise ValueError("Could not determine environment name from config")
    
    print(f"Using environment: {env_name}")
    
    # Create environment
    env_kwargs = config['environment'].get('kwargs', {})
    if env_kwargs is None:
        env_kwargs = {}
    
    # Add render_mode to kwargs
    env_kwargs['render_mode'] = args.render_mode
    
    # Create environment
    env = make_env(
        env_class=env_name,
        env_kwargs=env_kwargs,
        n_envs=1,
        is_atari=config['rl']['training'].get('is_atari', False),
        state_stack=config['rl']['training'].get('state_stack', 1),
        multithreaded=config['rl']['training'].get('multithreaded', False)
    )
    
    # Set up evaluator with specified seed
    logger = get_logger()
    evaluator = CustomRLEvaluator(model_path, algo=config['rl']['algo'])
    
    print(f"Evaluating model with seed {args.seed} for {args.episodes} episodes...")
    
    if args.render_mode == "rgb_array":
        print("This will generate GIFs in the current directory")
    else:
        print("You will see the live rendering of the environment")
    
    # Seed the environment
    env.seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare GIF options
    gif_options = {
        'fps': args.gif_fps,
        'skip_frames': args.skip_frames
    }
    
    if args.gif_name:
        gif_options['gif_name'] = args.gif_name
    
    # Run evaluation and generate GIFs
    evaluator.run(
        env, 
        seed=args.seed,
        n_episodes=args.episodes,
        logger=logger, 
        save_gif=args.render_mode == "rgb_array",
        gif_options=gif_options
    )
    
    print("Evaluation complete!") 