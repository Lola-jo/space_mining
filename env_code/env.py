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
        max_episode_steps: int = 2000, 
        grid_size: int = 80,  # Reduced from 100 to 80 for smaller, more manageable space
        max_asteroids: int = 12,  # Increased from 8 to 12 for more resources and higher potential score
        max_resource_per_asteroid: int = 40,  # Increased from 25 to 40 for more resources per asteroid
        observation_radius: int = 15,  # Reduced from 40 to 15 for more reasonable visibility
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
        self.max_asteroids = max(max_asteroids, 18)
        self.max_resource_per_asteroid = max_resource_per_asteroid
        self.observation_radius = observation_radius
        self.render_mode = render_mode
        
        # Constants for physics simulation
        self.dt = 0.1  # Time step for physics
        self.mass = 3.0  # Further reduced mass from 5.0 to 3.0 for much faster movement
        self.max_force = 20.0  # Further increased max force from 15.0 to 20.0 for much faster movement
        self.drag_coef = 0.02  # Further reduced drag coefficient from 0.05 to 0.02 for much less resistance
        self.gravity_strength = 0.01  # Further reduced gravity from 0.03 to 0.01 for much easier movement
        self.obstacle_penalty = -10.0  # Penalty for hitting obstacles
        self.energy_consumption_rate = 0.05  # Reduce base energy consumption
        self.mining_energy_cost = 1.0  # Reduce mining energy consumption
        
        # Initialize state variables
        self.steps_count = 0
        self.mothership_pos = np.array([grid_size/2, grid_size/2])  # 2D position
        
        # Define the action space
        # [fx, fy, mine] - 2D environment
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Maximum number of observable resources
        self.max_obs_asteroids = 15  
        
        # Increased inventory limit for easier mining and higher scores
        self.max_inventory = 100  # Increased from 60 to 100 for much easier mining and higher potential scores
        
        # Increased observation and mining range for easier exploration and mining
        # self.observation_radius is already set from parameter (60)
        self.mining_range = 8.0  # Increased from 5.0 to 8.0 for much easier mining
        
        # Define observation space
        # Agent state: position (2), velocity (2), energy (1), inventory (1)
        # Asteroids: relative position (2) and resource amount (1) for each visible asteroid
        # Mothership: relative position (2)
        agent_state_dim = 6
        asteroids_dim = self.max_obs_asteroids * 3
        mothership_dim = 2
        
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
        self.collision_count = 0
        
        # Initialize agent position, velocity, energy level, and inventory
        self.agent_position = self.np_random.uniform(
            low=0, high=self.grid_size, size=(2,)  # 2D position
        )
        self.agent_velocity = np.zeros(2, dtype=np.float32)  # 2D velocity
        self.agent_energy = 150.0  
        self.agent_inventory = 0.0
        self.cumulative_mining_amount = 0.0  # Initialize cumulative mining amount
        
        # Initialize asteroids and their resources 
        # Increase number of asteroids and resources per asteroid
        min_asteroids = min(8, self.max_asteroids)  # Increased from 4 to 8
        low = min_asteroids
        high = min(12, self.max_asteroids) + 1  # Increased max from 6 to 12
        if low >= high:
            low = max(6, self.max_asteroids)
        num_asteroids = self.np_random.integers(low, high)
        self.asteroid_positions = self.np_random.uniform(
            low=15, high=self.grid_size-15, size=(num_asteroids, 2)  # 2D positions
        )
        # Increase resources per asteroid for higher potential score
        self.asteroid_resources = self.np_random.uniform(
            low=25, high=40, size=(num_asteroids,)  # Increased from 10-20 to 25-40
        )
        
        # More obstacles for challenge
        num_obstacles = self.np_random.integers(4, 8)  # Increased from 1-3 to 4-8
        # Place obstacles randomly only, away from center
        self.obstacle_positions = self.np_random.uniform(
            low=20, high=self.grid_size-20, size=(num_obstacles, 2)  # 2D positions
        )
        # Faster obstacle movement
        self.obstacle_velocities = self.np_random.uniform(
            low=-0.2, high=0.2, size=(num_obstacles, 2)  # 2D velocities
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
        
        # Clear step-specific flags
        if hasattr(self, 'tried_depleted_asteroid'):
            delattr(self, 'tried_depleted_asteroid')
        
        # Extract actions
        thrust = action[:2] * self.max_force  # 2D thrust
        mine = action[2] > 0.5  # Convert to boolean
        
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
            gravity_acceleration = np.zeros(2)
        
        # Update velocity using Euler integration
        self.agent_velocity += (acceleration + drag_acceleration + gravity_acceleration) * self.dt
        
        # Update position
        self.agent_position += self.agent_velocity * self.dt
        
        # Enforce boundary conditions with stronger containment
        boundary_margin = 5.0  # Keep agent away from edges
        for axis in range(2):  # 2D boundaries
            if self.agent_position[axis] < boundary_margin:
                self.agent_position[axis] = boundary_margin
                self.agent_velocity[axis] = -0.3 * self.agent_velocity[axis]  # Stronger bounce
                reward += -1.0  # Increased boundary penalty
            elif self.agent_position[axis] > self.grid_size - boundary_margin:
                self.agent_position[axis] = self.grid_size - boundary_margin
                self.agent_velocity[axis] = -0.3 * self.agent_velocity[axis]  # Stronger bounce
                reward += -1.0  # Increased boundary penalty
        
        # Consume energy based on actions - much more efficient
        energy_used = self.energy_consumption_rate * 0.5  # Reduced base energy consumption from 2.0 to 0.5
        # Additional energy for thrust
        energy_used += np.sum(np.abs(thrust)) * 0.01  # Reduced thrust energy consumption from 0.02 to 0.01
        # Energy for mining if performed
        if mine:
            energy_used += self.mining_energy_cost * 0.5  # Reduced mining energy consumption from 2.0 to 0.5
        self.agent_energy -= energy_used
        # Track energy usage for display
        if not hasattr(self, 'last_energy_used'):
            self.last_energy_used = 0.0
        self.last_energy_used = energy_used
        
        # Check for energy depletion
        if self.agent_energy <= 0:
            self.agent_energy = 0
            reward += -10.0  # Reduced penalty for energy depletion
            terminated = True
        else:
            terminated = False
        
        # Check for collisions with obstacles
        obstacle_collisions = 0
        for obstacle_pos in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_pos)
            if distance < 1.5:  # Stricter collision threshold
                reward += -10.0  # Reduced penalty for collisions
                obstacle_collisions += 1
                self.collision_count += 1
                # Track collision for display
                if not hasattr(self, 'last_collision_step'):
                    self.last_collision_step = 0
                self.last_collision_step = self.steps_count
                # Simple collision response
                to_obstacle = self.agent_position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    self.agent_velocity += to_obstacle * 2.0
        # Terminate if too many collisions - increased tolerance
        if self.collision_count >= 12:  # Increased from 3 to 8
            print(f"[EPISODE END] Step {self.steps_count}: Too many collisions, terminating episode.")
            terminated = True
        
        # Enhanced mining action - much easier and more rewarding
        if mine and self.agent_energy > 0 and self.agent_inventory < self.max_inventory:
            mined_something = False
            tried_depleted_asteroid = False
            
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                distance = np.linalg.norm(self.agent_position - asteroid_pos)
                if distance < self.mining_range:
                    if self.asteroid_resources[i] <= 0.1:
                        # Agent tried to mine a depleted asteroid
                        tried_depleted_asteroid = True
                        continue
                    
                    # Extract resources from asteroid (1-2 mining attempts to deplete)
                    mining_efficiency = 0.6  # 60% of remaining resources per mining attempt
                    max_possible = min(self.asteroid_resources[i] * mining_efficiency, self.max_inventory - self.agent_inventory)
                    if max_possible > 0:
                        self.asteroid_resources[i] -= max_possible
                        self.agent_inventory += max_possible
                        
                        # Track cumulative mining amount
                        if not hasattr(self, 'cumulative_mining_amount'):
                            self.cumulative_mining_amount = 0.0
                        self.cumulative_mining_amount += max_possible
                        
                        # Track mining for display
                        if not hasattr(self, 'last_mining_info'):
                            self.last_mining_info = {}
                        self.last_mining_info = {
                            'step': self.steps_count,
                            'asteroid_id': i,
                            'extracted': max_possible,
                            'inventory': self.agent_inventory,
                            'cumulative_mining': self.cumulative_mining_amount,
                            'asteroid_depleted': self.asteroid_resources[i] <= 0.1
                        }
                        # Set mining asteroid ID for display
                        self.mining_asteroid_id = i
                        reward += max_possible * 8.0  # Increased mining reward from 4.0 to 8.0 for higher scores
                        mined_something = True
                        self.agent_velocity *= 0.8  # Reduced speed reduction from 0.7 to 0.8 for less slowdown
                        break
            
            if not mined_something:
                if tried_depleted_asteroid:
                    reward -= 0.2  # Penalty for trying to mine depleted asteroid
                    # Track depleted asteroid mining attempt for display
                    self.tried_depleted_asteroid = True
                else:
                    reward -= 0.1  # Reduced penalty for failed mining from 0.2 to 0.1
                # Clear mining asteroid ID if not mining
                if hasattr(self, 'mining_asteroid_id'):
                    delattr(self, 'mining_asteroid_id')
                
        elif mine and self.agent_inventory >= self.max_inventory:
            # print(f"[DEBUG] Step {self.steps_count}: Inventory full, cannot mine.")
            reward -= 0.2  # Reduced penalty for trying to mine when full from 0.5 to 0.2
            # Clear mining asteroid ID if not mining
            if hasattr(self, 'mining_asteroid_id'):
                delattr(self, 'mining_asteroid_id')
        elif not mine:
            # Clear mining asteroid ID if not mining
            if hasattr(self, 'mining_asteroid_id'):
                delattr(self, 'mining_asteroid_id')
        
        # Check for delivery to mothership - Much easier delivery
        distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        if distance_to_mothership < 12.0 and self.agent_inventory > 0:  # Increased delivery range from 8.0 to 12.0
            delivered_amount = self.agent_inventory
            reward += delivered_amount * 12.0  # Increased delivery reward from 6.0 to 12.0 for higher scores
            # Track delivery for display
            if not hasattr(self, 'last_delivery_info'):
                self.last_delivery_info = {}
            # Calculate energy recharge before setting to full
            energy_recharged = 150.0 - self.agent_energy  # Full recharge to 150.0
            self.last_delivery_info = {
                'step': self.steps_count,
                'delivered': delivered_amount,
                'energy_recharged': energy_recharged  # Correct calculation
            }
            self.agent_inventory = 0
            # Fully recharge energy when at mothership
            self.agent_energy = 150.0  # Set to full energy
            reward += energy_recharged * 0.5  # Increased recharge reward from 0.2 to 0.5
        
        # Update obstacles
        for i in range(len(self.obstacle_positions)):
            self.obstacle_positions[i] += self.obstacle_velocities[i] * self.dt
            
            # Simple boundary reflection
            for axis in range(2):  # 2D boundaries
                if self.obstacle_positions[i][axis] < 0 or self.obstacle_positions[i][axis] > self.grid_size:
                    self.obstacle_velocities[i][axis] *= -1
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode should be truncated due to time limit
        truncated = self.steps_count >= self.max_episode_steps
        if truncated:
            print(f"[EPISODE END] Step {self.steps_count}: Time limit reached ({self.max_episode_steps} steps) - Episode truncated")
        
        
        # Terminate if all asteroids are depleted (exploration complete)
        if np.all(self.asteroid_resources <= 0.1):
            terminated = True
            info = self._get_info()
            info["exploration_complete"] = True
            print(f"[EPISODE END] Step {self.steps_count}: All asteroids depleted - Episode completed successfully")
        
        
        # Get info dict
        info = self._get_info()
        info["obstacle_collisions"] = obstacle_collisions
        
        # Compute advanced reward (complements the immediate rewards from step function)
        advanced_reward, reward_info = self.compute_reward(action, observation, info)
        reward += advanced_reward
        
        # Compute fitness score for evaluation
        fitness_score = self.compute_fitness_score()
        info["fitness_score"] = fitness_score
        info.update(reward_info)
        
        # Add immediate reward components to info for debugging
        info["immediate_rewards"] = {
            "mining_reward": reward - advanced_reward,  # The reward from step function before advanced reward
            "total_reward": reward
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the observation vector."""
        # Agent's own state
        agent_state = np.concatenate([
            self.agent_position,     # position (2)
            self.agent_velocity,     # velocity (2)
            [self.agent_energy],     # energy (1)
            [self.agent_inventory]   # inventory (1)
        ])
        
        # Nearby asteroids
        asteroid_obs = np.zeros((self.max_obs_asteroids, 3), dtype=np.float32)  # 2D position + resource
        asteroid_count = 0
        
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0.1:
                continue
                
            rel_pos = asteroid_pos - self.agent_position
            distance = np.linalg.norm(rel_pos)
            
            if distance <= self.observation_radius:
                if asteroid_count < self.max_obs_asteroids:
                    asteroid_obs[asteroid_count] = np.concatenate([
                        rel_pos,  # 2D relative position
                        [self.asteroid_resources[i]]
                    ])
                    asteroid_count += 1
        
        # Mothership
        mothership_rel_pos = self.mothership_pos - self.agent_position  # 2D relative position
        
        # Concatenate all observation components
        observation = np.concatenate([
            agent_state,
            asteroid_obs.flatten(),
            mothership_rel_pos
        ])
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get the info dictionary."""
        info = {
            "agent_position": self.agent_position.copy(),
            "agent_velocity": self.agent_velocity.copy(),
            "agent_energy": float(self.agent_energy),
            "agent_inventory": float(self.agent_inventory),
            "asteroid_resources": self.asteroid_resources.copy(),
            "mothership_pos": self.mothership_pos.copy(),
            "asteroid_positions": self.asteroid_positions.copy(),
            "collision_count": self.collision_count,
            "steps_count": self.steps_count
        }
        
        # Add cumulative mining amount
        if hasattr(self, 'cumulative_mining_amount'):
            info["cumulative_mining_amount"] = float(self.cumulative_mining_amount)
        else:
            info["cumulative_mining_amount"] = 0.0
        
        # Add mining status
        if hasattr(self, 'mining_asteroid_id'):
            info["mining_asteroid_id"] = self.mining_asteroid_id
        else:
            info["mining_asteroid_id"] = None
            
        return info
    
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
            if self.asteroid_resources[i] > 0.1:
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
            resources_collected * 50.0 +              # Current inventory  
            energy_remaining * 300.0 +                # Energy management 
            resource_depletion_ratio * 200.0 +        # Resource collection efficiency 
            nearest_asteroid_dist * 100.0 * (1 - resource_depletion_ratio) +  # Proximity to resources 
            mothership_proximity * 100.0 * (self.agent_inventory > 0)  # Proximity to mothership 
        )
        

        completion_bonus = resource_depletion_ratio * 500.0  

        efficiency_bonus = (self.steps_count > 0) * (resource_depletion_ratio / max(1, self.steps_count)) * 1000.0 

        survival_bonus = self.steps_count * 0.5  
        
        fitness += completion_bonus + efficiency_bonus + survival_bonus
        
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
            self.font = pygame.font.SysFont("Arial", 16) 
        
        self.window.fill((0, 0, 0))  # Space background (black)
        
        # Helper function to convert 2D coordinates to screen coordinates
        def to_screen(pos, scale=8.0):
            x, y = pos
            # Center the 80x80 grid in the 800x800 screen
            # Map 0-80 to 0-800, centered at 400
            screen_x = int(400 + (x - 40) * scale)  # Center at 400, scale from -40 to +40
            screen_y = int(400 + (y - 40) * scale)  # Center at 400, scale from -40 to +40
            return screen_x, screen_y
        
        # Draw mothership
        mothership_pos_2d = to_screen(self.mothership_pos)
        pygame.gfxdraw.filled_circle(
            self.window, 
            mothership_pos_2d[0], 
            mothership_pos_2d[1], 
            15, 
            (50, 150, 200)
        )
        
        # Draw asteroids with health bars
        for i, pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0.1:
                # Draw depleted asteroid as gray cross
                asteroid_pos_2d = to_screen(pos)
                pygame.gfxdraw.filled_circle(
                    self.window, 
                    asteroid_pos_2d[0], 
                    asteroid_pos_2d[1], 
                    8, 
                    (100, 100, 100)  # Gray for depleted
                )
                # Draw X mark for depleted asteroid
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] - 5, asteroid_pos_2d[1] - 5),
                    (asteroid_pos_2d[0] + 5, asteroid_pos_2d[1] + 5),
                    2
                )
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] + 5, asteroid_pos_2d[1] - 5),
                    (asteroid_pos_2d[0] - 5, asteroid_pos_2d[1] + 5),
                    2
                )
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
            
            # Draw health bar for asteroid
            health_ratio = self.asteroid_resources[i] / 40.0  # Normalize to max resource
            health_width = int(20 * health_ratio)
            health_x = asteroid_pos_2d[0] - 10
            health_y = asteroid_pos_2d[1] - size - 8
            
            # Background (gray)
            pygame.draw.rect(
                self.window,
                (100, 100, 100),
                (health_x, health_y, 20, 4)
            )
            # Health bar (green to red based on remaining resources)
            health_color = (int(255 * health_ratio), int(255 * (1 - health_ratio)), 0)  # Green to Red
            pygame.draw.rect(
                self.window,
                health_color,
                (health_x, health_y, health_width, 4)
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
        
        # Draw agent body - change color based on state
        agent_color = (50, 200, 50)  # Default green
        if hasattr(self, 'mining_asteroid_id'):
            agent_color = (255, 165, 0)  # Orange when mining
        elif self.agent_inventory > 0:
            agent_color = (255, 255, 0)  # Yellow when carrying resources
        
        pygame.gfxdraw.filled_circle(
            self.window, 
            agent_pos_2d[0], 
            agent_pos_2d[1], 
            20,  # Increased size from 12 to 20
            agent_color
        )
        
        # Draw white outline
        pygame.gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            20,  # Increased size from 12 to 20
            (255, 255, 255)
        )
        
        # Draw mining indicator
        if hasattr(self, 'mining_asteroid_id'):
            # Draw mining beam/effect
            asteroid_pos_2d = to_screen(self.asteroid_positions[self.mining_asteroid_id])
            pygame.draw.line(
                self.window,
                (255, 255, 0),  # Yellow beam
                agent_pos_2d,
                asteroid_pos_2d,
                3
            )
            # Draw mining text
            mining_text = f"MINING ASTEROID {self.mining_asteroid_id}"
            text_surface = self.font.render(mining_text, True, (255, 255, 0))
            self.window.blit(text_surface, (agent_pos_2d[0] - 50, agent_pos_2d[1] - 30))
        
        # Draw inventory indicator
        if self.agent_inventory > 0:
            pygame.gfxdraw.filled_circle(
                self.window, 
                agent_pos_2d[0], 
                agent_pos_2d[1], 
                int(5 + self.agent_inventory / 5),  # Larger indicator
                (200, 200, 0)
            )
        
        # Draw energy bar
        energy_width = int(40 * (self.agent_energy / 150.0))  # Fixed: use 150.0 as max energy
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
        
        # Draw observation radius (exploration range)
        pygame.gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.observation_radius * 8.0),  # scale to screen
            (100, 100, 255)
        )
        
        # Draw mining range 
        pygame.gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.mining_range * 8.0),  # scale to screen
            (255, 0, 0)  # Pure red for mining range
        )
        
        # Draw mining range indicator text
        mining_text = f"MINING RANGE: {self.mining_range:.1f}"
        text_surface = self.font.render(mining_text, True, (255, 0, 0))
        self.window.blit(text_surface, (agent_pos_2d[0] - 50, agent_pos_2d[1] + 30))
        
        # Display status info on top-left
        cumulative_mining = getattr(self, 'cumulative_mining_amount', 0.0)
        status_text = [
            f"Energy: {self.agent_energy:.0f}/150",
            f"Inventory: {self.agent_inventory:.0f}/{self.max_inventory}",
            f"Total Mined: {cumulative_mining:.1f}",
            f"Collisions: {self.collision_count}",
            f"Step: {self.steps_count}/{self.max_episode_steps}",
            f"Asteroids: {np.sum(self.asteroid_resources > 0)}/{len(self.asteroid_positions)}"
        ]
        
        # Add mining status
        if hasattr(self, 'mining_asteroid_id'):
            status_text.append(f"MINING: Asteroid {self.mining_asteroid_id}")
        elif self.agent_inventory > 0:
            status_text.append("CARRYING RESOURCES")
        else:
            status_text.append("EXPLORING")
        
        # Add DEBUG information display
        if hasattr(self, 'last_mining_info') and self.last_mining_info.get('step', 0) == self.steps_count:
            status_text.append(f"MINED: {self.last_mining_info['extracted']:.1f} from Asteroid {self.last_mining_info['asteroid_id']}")
            status_text.append(f"TOTAL MINED: {self.last_mining_info.get('cumulative_mining', 0.0):.1f}")
            if self.last_mining_info.get('asteroid_depleted', False):
                status_text.append("ASTEROID DEPLETED!")
        
        if hasattr(self, 'last_delivery_info') and self.last_delivery_info.get('step', 0) == self.steps_count:
            status_text.append(f"DELIVERED: {self.last_delivery_info['delivered']:.1f} resources")
            status_text.append(f"FULLY RECHARGED!")
        
        if hasattr(self, 'last_collision_step') and self.last_collision_step == self.steps_count:
            status_text.append("COLLISION DETECTED!")
        
        # Add depleted asteroid mining attempt indicator
        if hasattr(self, 'tried_depleted_asteroid') and self.tried_depleted_asteroid:
            status_text.append("TRIED TO MINE DEPLETED ASTEROID!")
        
        # Display status on top-left
        for i, text in enumerate(status_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (10, 10 + i * 20))
        
        # Display legend on bottom-right
        legend_text = [
            "LEGEND:",
            "Green Circle = Agent",
            "Blue Circle = Mothership", 
            "Red Circles = Obstacles",
            "Yellow Circles = Asteroids",
            "Gray X = Depleted Asteroids",
            "Blue Ring = Observation Range",
            "Red Ring = Mining Range"
        ]
        
        # Display legend on bottom-right - fully to the right
        for i, text in enumerate(legend_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (600, 600 + i * 20))  # Moved from 400 to 600 for full right alignment
        
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
    # Generated code by stable-eureka
    def compute_reward(self, action, observation, info):
        """
        Computes the reward for the agent.

        Args:
            action: The agent's action.
            observation: The agent's observation.
            info: Additional information about the environment.

        Returns:
            Total reward (float), dictionary of individual reward components
        """

        # Constants
        SPEED_LIMIT = 10.0  # Increased from 8.0 for more movement freedom
        EFFICIENCY_THRESHOLD = 0.5  # Lowered from 0.7 for easier energy management
        EXPLORATION_BONUS = 3.0  # Increased from 2.0 for better exploration
        PATH_EFFICIENCY_BONUS = 2.0  # Increased from 1.0 for better path guidance
        MINING_GUIDANCE_BONUS = 2.0  # New bonus for mining guidance
        DELIVERY_GUIDANCE_BONUS = 3.0  # New bonus for delivery guidance
        
        # Initialize reward components
        speed_penalty = 0.0
        efficiency_reward = 0.0
        exploration_reward = 0.0
        path_efficiency_reward = 0.0
        mining_guidance_reward = 0.0
        delivery_guidance_reward = 0.0
        
        # 1. Speed control penalty (encourage moderate speeds)
        speed = np.linalg.norm(observation[2:4])  # velocity from observation (2D)
        if speed > SPEED_LIMIT:
            # Use quadratic penalty instead of exponential for more stable training
            speed_penalty = -0.05 * (speed - SPEED_LIMIT) ** 2  # Reduced penalty from 0.1 to 0.05
        
        # 2. Energy efficiency reward (encourage energy conservation)
        energy = observation[4]  # energy from observation
        energy_ratio = energy / 150.0  # normalize energy
        if energy_ratio > EFFICIENCY_THRESHOLD:
            efficiency_reward = 1.0 * energy_ratio  # Increased reward from 0.5 to 1.0
        
        # 3. Exploration reward (encourage discovering new asteroids)
        if not hasattr(self, 'discovered_asteroids'):
            self.discovered_asteroids = set()
        
        # Check for newly discovered asteroids
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0.1:
                continue
            distance = np.linalg.norm(self.agent_position - asteroid_pos)
            if distance <= self.observation_radius and i not in self.discovered_asteroids:
                self.discovered_asteroids.add(i)
                exploration_reward += EXPLORATION_BONUS
        
        # 4. Path efficiency reward (encourage direct paths to objectives)
        inventory = observation[5]  # inventory from observation (2D)
        
        if inventory > 0:
            # When carrying resources, reward being close to mothership
            distance_to_mothership = np.linalg.norm(observation[0:2] - self.mothership_pos)  # 2D position
            if distance_to_mothership < 15.0:  # Increased delivery range from 12.0 to 15.0
                path_efficiency_reward = PATH_EFFICIENCY_BONUS * 2.0 * (1.0 - distance_to_mothership / 15.0)  # Increased reward
                delivery_guidance_reward = DELIVERY_GUIDANCE_BONUS * (1.0 - distance_to_mothership / 15.0)  # New delivery guidance
        else:
            # When not carrying resources, reward being close to asteroids with resources
            nearest_asteroid_dist = float('inf')
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                if self.asteroid_resources[i] > 0.1:
                    dist = np.linalg.norm(self.agent_position - asteroid_pos)
                    nearest_asteroid_dist = min(nearest_asteroid_dist, dist)
            
            if nearest_asteroid_dist < float('inf'):
                if nearest_asteroid_dist < 10.0:  # Increased mining range from 8.0 to 10.0
                    path_efficiency_reward = PATH_EFFICIENCY_BONUS * 2.0 * (1.0 - nearest_asteroid_dist / 10.0)  # Increased reward
                    mining_guidance_reward = MINING_GUIDANCE_BONUS * (1.0 - nearest_asteroid_dist / 10.0)  # New mining guidance
        
        # 5. Additional guidance rewards
        # Reward for being near mothership when low on energy
        if energy_ratio < 0.3 and inventory == 0:
            distance_to_mothership = np.linalg.norm(observation[0:2] - self.mothership_pos)  # 2D position
            if distance_to_mothership < 20.0:
                delivery_guidance_reward += 1.0 * (1.0 - distance_to_mothership / 20.0)
        
        # Calculate total reward
        total_reward = (speed_penalty + efficiency_reward + exploration_reward + 
                       path_efficiency_reward + mining_guidance_reward + delivery_guidance_reward)
        
        # Return total reward and reward components
        return total_reward, {
            "speed_penalty": speed_penalty,
            "efficiency_reward": efficiency_reward,
            "exploration_reward": exploration_reward,
            "path_efficiency_reward": path_efficiency_reward,
            "mining_guidance_reward": mining_guidance_reward,
            "delivery_guidance_reward": delivery_guidance_reward,
        }