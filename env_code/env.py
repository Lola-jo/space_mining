import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class FlattenActionSpaceWrapper(gym.Wrapper):
    """
    Flatten the action space from Dict to Box for compatibility with standard RL algorithms.
    """
    
    def __init__(self, env):
        """Initialize the wrapper."""
        super().__init__(env)
        
        # The action space is already a Box with shape (4,)
        # [fx, fy, fz, mine]
        # This wrapper exists for compatibility but does not modify the action space
        # for the simplified single-agent environment
        self.action_space = env.action_space
        
    def step(self, action):
        """Execute environment step using the flattened action."""
        return self.env.step(action)


def make_env(**kwargs):
    """
    Create and wrap the space mining environment.
    
    Returns:
        A wrapped environment compatible with standard RL algorithms
    """
    # For backwards compatibility, remove multi-agent parameters
    if 'num_agents' in kwargs:
        kwargs.pop('num_agents')
    
    # Remove communication radius if present (only used in multi-agent version)
    if 'communication_radius' in kwargs:
        kwargs.pop('communication_radius')
        
    env = SpaceMiningEnv(**kwargs)
    env = FlattenActionSpaceWrapper(env)
    return env


class SpaceMiningEnv(gym.Env):
    """
    Space Mining Environment (Simplified Single-Agent Version)
    
    Agent (mining robot) must collect resources from asteroids
    and return them to the mothership while managing energy and avoiding obstacles.
    
    Features:
    - Physics simulation with gravity, inertia, and collisions
    - Dynamic resource distribution
    - Energy management
    - Partial observability
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        max_episode_steps: int = 1600, 
        grid_size: int = 100,
        max_asteroids: int = 25,  
        max_resource_per_asteroid: int = 80,  
        observation_radius: int = 50,  
        render_mode: Optional[str] = None,
    ):
        """Initialize the Space Mining environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            grid_size: Size of the 3D space grid (cubic)
            max_asteroids: Maximum number of asteroids
            max_resource_per_asteroid: Maximum resources per asteroid
            observation_radius: Agent observation radius
            render_mode: Rendering mode
        """
        self.max_episode_steps = max_episode_steps
        self.grid_size = grid_size
        self.max_asteroids = max_asteroids
        self.max_resource_per_asteroid = max_resource_per_asteroid
        self.observation_radius = observation_radius
        self.render_mode = render_mode
        
        # Constants for physics simulation
        self.dt = 0.1  # Time step for physics
        self.mass = 10.0  # Mass of agent
        self.max_force = 10.0  # Maximum thrust force
        self.drag_coef = 0.1  # Drag coefficient
        self.gravity_strength = 0.05  # Strength of gravity fields
        self.obstacle_penalty = -10.0  # Penalty for hitting obstacles
        self.energy_consumption_rate = 0.05  # 降低基础能量消耗
        self.mining_energy_cost = 1.0  # 降低采矿能量消耗
        
        # Initialize state variables
        self.steps_count = 0
        self.mothership_pos = np.array([grid_size/2, grid_size/2, grid_size/2])
        
        # Define the action space
        # [fx, fy, fz, mine]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Maximum number of observable resources
        self.max_obs_asteroids = 15  
        
        # Define observation space
        # Agent state: position (3), velocity (3), energy (1), inventory (1)
        # Asteroids: relative position (3) and resource amount (1) for each visible asteroid
        # Mothership: relative position (3)
        agent_state_dim = 8
        asteroids_dim = self.max_obs_asteroids * 4
        mothership_dim = 3
        
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(agent_state_dim + asteroids_dim + mothership_dim,),
            dtype=np.float32
        )
        
        # Initialize rendering
        self.window = None
        self.clock = None
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        self.steps_count = 0
        
        # Initialize agent position, velocity, energy level, and inventory
        self.agent_position = self.np_random.uniform(
            low=0, high=self.grid_size, size=(3,)
        )
        self.agent_velocity = np.zeros(3, dtype=np.float32)
        self.agent_energy = 150.0  
        self.agent_inventory = 0.0
        
        # Initialize asteroids and their resources
        num_asteroids = self.np_random.integers(10, self.max_asteroids)  
        self.asteroid_positions = self.np_random.uniform(
            low=0, high=self.grid_size, size=(num_asteroids, 3)
        )
        self.asteroid_resources = self.np_random.uniform(
            low=15, high=self.max_resource_per_asteroid, size=(num_asteroids,)  
        )
        
        # Simple obstacles (non-mining competitive elements)
        num_obstacles = self.np_random.integers(2, 5)  
        self.obstacle_positions = self.np_random.uniform(
            low=0, high=self.grid_size, size=(num_obstacles, 3)
        )
        self.obstacle_velocities = self.np_random.uniform(
            low=-0.3, high=0.3, size=(num_obstacles, 3)  
        )
        
        # Initialize tracking variables for reward calculation
        self.prev_min_asteroid_distance = float('inf')
        self.prev_inventory = 0.0
        self.prev_energy = self.agent_energy
        self.prev_distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        self.discovered_asteroids = set()
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict for reset
        info = {
            "total_resources_collected": 0,
            "obstacle_collisions": 0,
            "energy_remaining": self.agent_energy
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with the given action."""
        self.steps_count += 1
        
        # Extract actions
        thrust = action[:3] * self.max_force
        mine = action[3] > 0.5  # Convert to boolean
        
        # Apply physics and update agent state
        reward = 0.0
        
        # Skip if agent has no energy
        if self.agent_energy <= 0:
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
            
        # Apply thrust force
        acceleration = thrust / self.mass
        
        # Apply drag force: F_drag = -k * v
        drag_acceleration = -self.drag_coef * self.agent_velocity / self.mass
        
        # Apply gravity towards mothership (simplified)
        to_mothership = self.mothership_pos - self.agent_position
        distance_to_mothership = np.linalg.norm(to_mothership)
        if distance_to_mothership > 0:
            gravity_acceleration = (to_mothership / distance_to_mothership) * self.gravity_strength
        else:
            gravity_acceleration = np.zeros(3)
        
        # Update velocity using Euler integration
        self.agent_velocity += (acceleration + drag_acceleration + gravity_acceleration) * self.dt
        
        # Update position
        self.agent_position += self.agent_velocity * self.dt
        
        # Enforce boundary conditions (simple bounce)
        for axis in range(3):
            if self.agent_position[axis] < 0:
                self.agent_position[axis] = 0
                self.agent_velocity[axis] = -0.5 * self.agent_velocity[axis]  # Bounce with damping
                reward += -1.0  # 降低边界碰撞惩罚
            elif self.agent_position[axis] > self.grid_size:
                self.agent_position[axis] = self.grid_size
                self.agent_velocity[axis] = -0.5 * self.agent_velocity[axis]  # Bounce with damping
                reward += -1.0  # 降低边界碰撞惩罚
        
        # Consume energy based on actions
        energy_used = self.energy_consumption_rate  # Base energy consumption
        
        # Additional energy for thrust
        energy_used += np.sum(np.abs(thrust)) * 0.01  # 降低推力能量消耗
        
        # Energy for mining if performed
        if mine:
            energy_used += self.mining_energy_cost
        
        self.agent_energy -= energy_used
        
        # Check for energy depletion
        if self.agent_energy <= 0:
            self.agent_energy = 0
            reward += -20.0  # 降低能量耗尽的惩罚
            terminated = True
        else:
            terminated = False
        
        # Check for collisions with obstacles
        obstacle_collisions = 0
        for obstacle_pos in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_pos)
            if distance < 3.0:  # Collision threshold
                reward += self.obstacle_penalty
                obstacle_collisions += 1
                
                # Simple collision response
                to_obstacle = self.agent_position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    # Bounce away from obstacle
                    self.agent_velocity += to_obstacle * 2.0
        
        # Process mining action
        if mine and self.agent_energy > 0:
            # Check for nearby asteroids to mine
            mined_something = False
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                if self.asteroid_resources[i] <= 0:
                    continue
                    
                distance = np.linalg.norm(self.agent_position - asteroid_pos)
                if distance < 5.0:  
                    # Extract resources from asteroid
                    mined_amount = min(5.0, self.asteroid_resources[i])
                    self.asteroid_resources[i] -= mined_amount
                    self.agent_inventory += mined_amount
                    
                    reward += mined_amount * 2.0  # Reward for mining
                    mined_something = True
                    break  # Can only mine one asteroid at a time
            
            # Small penalty for trying to mine with nothing nearby
            if not mined_something:
                reward -= 0.5
        
        # Check for delivery to mothership
        distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        if distance_to_mothership < 8.0 and self.agent_inventory > 0:  
            # Deliver resources
            delivered_amount = self.agent_inventory
            reward += delivered_amount * 10.0  # Large reward for successful delivery
            self.agent_inventory = 0
            
            # Recharge energy when at mothership
            energy_recharged = min(30.0, 150.0 - self.agent_energy)  
            self.agent_energy += energy_recharged
            reward += energy_recharged * 0.2  # Small reward for recharging
        
        # Update obstacles
        for i in range(len(self.obstacle_positions)):
            self.obstacle_positions[i] += self.obstacle_velocities[i] * self.dt
            
            # Simple boundary reflection
            for axis in range(3):
                if self.obstacle_positions[i][axis] < 0 or self.obstacle_positions[i][axis] > self.grid_size:
                    self.obstacle_velocities[i][axis] *= -1
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode should be truncated due to time limit
        truncated = self.steps_count >= self.max_episode_steps
        
        # Get info dict
        info = self._get_info()
        info["obstacle_collisions"] = obstacle_collisions
        
        # Compute advanced reward
        advanced_reward, reward_info = self.compute_reward(action, observation, info)
        reward += advanced_reward
        
        # Compute fitness score
        fitness_score = self.compute_fitness_score()
        info["fitness_score"] = fitness_score
        info.update(reward_info)
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the observation vector."""
        # Agent's own state
        agent_state = np.concatenate([
            self.agent_position,     # position (3)
            self.agent_velocity,     # velocity (3)
            [self.agent_energy],     # energy (1)
            [self.agent_inventory]   # inventory (1)
        ])
        
        # Nearby asteroids
        asteroid_obs = np.zeros((self.max_obs_asteroids, 4), dtype=np.float32)
        asteroid_count = 0
        
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0:
                continue
                
            rel_pos = asteroid_pos - self.agent_position
            distance = np.linalg.norm(rel_pos)
            
            if distance <= self.observation_radius:
                if asteroid_count < self.max_obs_asteroids:
                    asteroid_obs[asteroid_count] = np.concatenate([
                        rel_pos,
                        [self.asteroid_resources[i]]
                    ])
                    asteroid_count += 1
        
        # Mothership
        mothership_rel_pos = self.mothership_pos - self.agent_position
        
        # Concatenate all observation components
        observation = np.concatenate([
            agent_state,
            asteroid_obs.flatten(),
            mothership_rel_pos
        ])
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get the info dictionary."""
        return {
            "agent_position": self.agent_position.copy(),
            "agent_velocity": self.agent_velocity.copy(),
            "agent_energy": float(self.agent_energy),
            "agent_inventory": float(self.agent_inventory),
            "asteroid_resources": self.asteroid_resources.copy(),
            "mothership_pos": self.mothership_pos.copy(),
            "asteroid_positions": self.asteroid_positions.copy()
        }
    
    def compute_fitness_score(self) -> float:
        """
        Compute a comprehensive fitness score for evaluating agent performance.
        
        Returns:
            float: Overall fitness score
        """
        # Basic score components
        resources_collected = self.agent_inventory
        energy_remaining = self.agent_energy / 100.0  # Normalized energy
        
        # Resources in asteroids
        remaining_resources = np.sum(self.asteroid_resources)
        total_resources_initial = self.max_resource_per_asteroid * self.max_asteroids
        resource_depletion_ratio = 1.0 - (remaining_resources / total_resources_initial)
        
        # Distance to nearest asteroid with resources
        nearest_asteroid_dist = float('inf')
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] > 0:
                dist = np.linalg.norm(self.agent_position - asteroid_pos)
                nearest_asteroid_dist = min(nearest_asteroid_dist, dist)
        
        if nearest_asteroid_dist == float('inf'):
            nearest_asteroid_dist = 0  # No resources left
        else:
            nearest_asteroid_dist = 1.0 - min(1.0, nearest_asteroid_dist / self.grid_size)  # Normalize and invert
        
        # Distance to mothership (normalized and inverted)
        distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        mothership_proximity = 1.0 - min(1.0, distance_to_mothership / self.grid_size)
        
        # Calculate fitness score as weighted sum
        fitness = (
            resources_collected * 5.0 +               # Current inventory 
            energy_remaining * 30.0 +                 # Energy management
            resource_depletion_ratio * 20.0 +         # Resource collection efficiency
            nearest_asteroid_dist * 10.0 * (1 - resource_depletion_ratio) +  # Proximity to resources (if still available)
            mothership_proximity * 10.0 * (self.agent_inventory > 0)  # Proximity to mothership (if carrying resources)
        )
        
        return fitness
    
   
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 800))
            pygame.display.set_caption("Space Mining Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 12)
        
        self.window.fill((0, 0, 0))  # Space background (black)
        
        # Helper function to convert 3D coordinates to 2D screen coordinates
        def to_screen(pos, scale=8.0):
            x, y, z = pos
            # Use a simple isometric projection
            screen_x = 400 + (x - y) * scale
            screen_y = 400 + (x + y - z * 2) * scale / 2
            return int(screen_x), int(screen_y)
        
        # Draw mothership
        mothership_pos_2d = to_screen(self.mothership_pos)
        pygame.gfxdraw.filled_circle(
            self.window, 
            mothership_pos_2d[0], 
            mothership_pos_2d[1], 
            15, 
            (50, 150, 200)
        )
        
        # Draw asteroids
        for i, pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0:
                continue
                
            asteroid_pos_2d = to_screen(pos)
            size = int(5 + self.asteroid_resources[i] / 10)
            color_intensity = min(255, int(100 + self.asteroid_resources[i] * 3))
            pygame.gfxdraw.filled_circle(
                self.window, 
                asteroid_pos_2d[0], 
                asteroid_pos_2d[1], 
                size, 
                (color_intensity, color_intensity // 2, 0)
            )
        
        # Draw obstacles
        for pos in self.obstacle_positions:
            obstacle_pos_2d = to_screen(pos)
            pygame.gfxdraw.filled_circle(
                self.window, 
                obstacle_pos_2d[0], 
                obstacle_pos_2d[1], 
                7, 
                (200, 50, 50)
            )
        
        # Draw agent
        agent_pos_2d = to_screen(self.agent_position)
        
        # Draw agent body
        pygame.gfxdraw.filled_circle(
            self.window, 
            agent_pos_2d[0], 
            agent_pos_2d[1], 
            10, 
            (50, 200, 50)
        )
        
        # Draw inventory indicator
        if self.agent_inventory > 0:
            pygame.gfxdraw.filled_circle(
                self.window, 
                agent_pos_2d[0], 
                agent_pos_2d[1], 
                int(3 + self.agent_inventory / 10),
                (200, 200, 0)
            )
        
        # Draw energy bar
        energy_width = int(40 * (self.agent_energy / 100))
        pygame.draw.rect(
            self.window,
            (0, 0, 0),
            (agent_pos_2d[0] - 20, agent_pos_2d[1] - 20, 40, 5)
        )
        pygame.draw.rect(
            self.window,
            (0, 200, 0),
            (agent_pos_2d[0] - 20, agent_pos_2d[1] - 20, energy_width, 5)
        )
        
        # Display info on screen
        info_text = [
            f"Step: {self.steps_count}/{self.max_episode_steps}",
            f"Energy: {self.agent_energy:.1f}",
            f"Inventory: {self.agent_inventory:.1f}",
            f"Remaining Asteroids: {np.sum(self.asteroid_resources > 0)}/{len(self.asteroid_positions)}"
        ]
        
        for i, text in enumerate(info_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (10, 10 + i * 20))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


   