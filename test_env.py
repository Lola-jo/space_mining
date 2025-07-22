import sys
import os
import numpy as np

# 添加项目根目录到Python路径，以便导入环境
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from envs.space_mining.env_code.env import SpaceMiningEnv, make_env
from stable_baselines3 import PPO

def test_space_mining_env():
    """测试太空采矿环境的基本功能"""
    print("创建太空采矿环境...")
    
    # 创建环境实例
    env = SpaceMiningEnv(
        num_agents=2,
        grid_size=50,
        max_asteroids=10,
        observation_radius=20,
        communication_radius=30,
        render_mode="human"  # 设置为None以禁用渲染，human以显示窗口
    )
    
    print(f"环境创建成功！")
    print(f"动作空间: {env.action_space}")
    print(f"观察空间: {env.observation_space}")
    
    print("\n重置环境...")
    observations, info = env.reset(seed=42)
    print(f"初始信息: {info}")
    
    # 运行几个随机步骤
    print("\n执行10个随机步骤:")
    for step in range(10):
        # 为每个智能体生成随机动作
        actions = []
        for agent_idx in range(env.num_agents):
            agent_action = {
                "thrust": np.random.uniform(-0.5, 0.5, size=3),
                "mine": np.random.choice([0, 1]),
                "communicate": np.random.uniform(-1.0, 1.0, size=2)
            }
            actions.append(agent_action)
        
        # 执行步骤
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"步骤 {step+1}:")
        print(f"  奖励: {rewards}")
        print(f"  终止: {terminated}")
        print(f"  截断: {truncated}")
        print(f"  活跃智能体: {info['active_agents']}")
        print(f"  已收集资源: {info['total_resources_collected']}")
        
        # 检查是否所有智能体都已终止
        if all(terminated):
            print("\n所有智能体已终止，停止模拟")
            break
    
    print("\n关闭环境...")
    env.close()
    print("测试完成！")

if __name__ == "__main__":
    test_space_mining_env()

# 创建环境
env = make_env(num_agents=2)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# 测试SB3能否接受这个环境
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)
print("Training successful!") 