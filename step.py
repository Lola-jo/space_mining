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
                reward += -2.0  # Penalty for hitting boundary
            elif self.agent_position[axis] > self.grid_size:
                self.agent_position[axis] = self.grid_size
                self.agent_velocity[axis] = -0.5 * self.agent_velocity[axis]  # Bounce with damping
                reward += -2.0  # Penalty for hitting boundary
        
        # Consume energy based on actions
        energy_used = self.energy_consumption_rate  # Base energy consumption
        
        # Additional energy for thrust
        energy_used += np.sum(np.abs(thrust)) * 0.02
        
        # Energy for mining if performed
        if mine:
            energy_used += self.mining_energy_cost
        
        self.agent_energy -= energy_used
        
        # Check for energy depletion
        if self.agent_energy <= 0:
            self.agent_energy = 0
            reward += -50.0  # Penalty for running out of energy
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
                if distance < 3.0:  # Mining range
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
        if distance_to_mothership < 5.0 and self.agent_inventory > 0:
            # Deliver resources
            delivered_amount = self.agent_inventory
            reward += delivered_amount * 10.0  # Large reward for successful delivery
            self.agent_inventory = 0
            
            # Recharge energy when at mothership
            energy_recharged = min(20.0, 100.0 - self.agent_energy)
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