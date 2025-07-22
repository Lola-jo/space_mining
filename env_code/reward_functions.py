import numpy as np

def sample_compute_reward(self, actions, observations, info):
    """
    计算每个智能体的完整奖励函数（示例实现）。
    
    这个函数整合了基础奖励（在step函数中计算的采矿、递送、碰撞等奖励）
    和高级协作奖励（协作、能量管理、探索等）。
    
    Args:
        actions: 每个智能体的动作列表
        observations: 每个智能体的观察字典
        info: 环境信息字典
        
    Returns:
        tuple: (rewards, individual_rewards) 其中rewards是每个智能体的最终奖励数组，
               individual_rewards是包含各奖励组件详细信息的字典
    """
    num_agents = len(actions)
    rewards = np.zeros(num_agents, dtype=np.float32)
    
    # 记录每个智能体上一步的状态（如果存在）
    if not hasattr(self, 'prev_agent_positions'):
        self.prev_agent_positions = self.agent_positions.copy()
        self.prev_agent_energy = self.agent_energy.copy()
        self.prev_agent_inventories = self.agent_inventories.copy()
        self.prev_distances_to_mothership = np.array([
            np.linalg.norm(self.agent_positions[i] - self.mothership_pos)
            for i in range(num_agents)
        ])
        self.discovered_asteroids = [set() for _ in range(num_agents)]
        self.prev_communications = [np.zeros(2) for _ in range(num_agents)]
        self.resource_discoveries = np.zeros((num_agents, num_agents), dtype=np.float32)
        self.consecutive_mining = np.zeros(num_agents, dtype=np.int32)
        
        # 创建空的奖励信息字典
        reward_info = {
            "reward_collaboration": np.zeros(num_agents, dtype=np.float32),
            "reward_energy": np.zeros(num_agents, dtype=np.float32),
            "reward_exploration": np.zeros(num_agents, dtype=np.float32),
            "reward_communication": np.zeros(num_agents, dtype=np.float32),
            "reward_path": np.zeros(num_agents, dtype=np.float32),
        }
        return rewards, reward_info
    
    # 1. 协作奖励计算
    collaboration_rewards = self._compute_collaboration_reward(actions, observations, info)
    
    # 2. 能量效率奖励计算
    energy_efficiency_rewards = self._compute_energy_efficiency_reward(actions, observations, info)
    
    # 3. 探索奖励计算
    exploration_rewards = self._compute_exploration_reward(actions, observations, info)
    
    # 4. 通信效率奖励/惩罚
    communication_rewards = self._compute_communication_reward(actions, observations, info)
    
    # 5. 路径规划奖励
    path_planning_rewards = self._compute_path_planning_reward(actions, observations, info)
    
    # 创建奖励信息字典
    reward_info = {
        "reward_collaboration": collaboration_rewards.copy(),
        "reward_energy": energy_efficiency_rewards.copy(),
        "reward_exploration": exploration_rewards.copy(),
        "reward_communication": communication_rewards.copy(),
        "reward_path": path_planning_rewards.copy(),
    }
    
    # 组合所有奖励组件
    for i in range(num_agents):
        if self.agent_energy[i] <= 0:
            continue
            
        # 加权组合不同的奖励组件
        rewards[i] += (
            collaboration_rewards[i] * 1.0 +
            energy_efficiency_rewards[i] * 0.5 +
            exploration_rewards[i] * 0.3 +
            communication_rewards[i] * 0.2 +
            path_planning_rewards[i] * 0.4
        )
    
    # 更新历史状态
    self.prev_agent_positions = self.agent_positions.copy()
    self.prev_agent_energy = self.agent_energy.copy()
    self.prev_agent_inventories = self.agent_inventories.copy()
    self.prev_distances_to_mothership = np.array([
        np.linalg.norm(self.agent_positions[i] - self.mothership_pos)
        for i in range(num_agents)
    ])
    self.prev_communications = [action["communicate"] if "communicate" in action else np.zeros(2)
                               for action in actions]
    
    return rewards, reward_info

def _compute_collaboration_reward(self, actions, observations, info):
    """计算协作奖励组件"""
    num_agents = len(actions)
    collaboration_rewards = np.zeros(num_agents, dtype=np.float32)
    
    # 1. 识别采矿协作 - 当多个智能体在同一小行星附近开采时
    mining_agents = [(i, action) for i, action in enumerate(actions) 
                    if "mine" in action and action["mine"] == 1 and self.agent_energy[i] > 0]
    
    # 统计每个小行星附近的采矿智能体
    asteroid_miners = {}
    for agent_idx, _ in mining_agents:
        nearby_asteroids = []
        for j, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[j] <= 0:
                continue
                
            distance = np.linalg.norm(self.agent_positions[agent_idx] - asteroid_pos)
            if distance < 5.0:  # 稍大于采矿范围，以识别附近的智能体
                nearby_asteroids.append(j)
                
        for asteroid_idx in nearby_asteroids:
            if asteroid_idx not in asteroid_miners:
                asteroid_miners[asteroid_idx] = []
            asteroid_miners[asteroid_idx].append(agent_idx)
    
    # 为同一小行星附近的多个采矿智能体提供协作奖励
    for asteroid_idx, miners in asteroid_miners.items():
        if len(miners) > 1 and self.asteroid_resources[asteroid_idx] > 0:
            # 小行星资源越多，协作奖励越高
            base_reward = min(2.0, self.asteroid_resources[asteroid_idx] / 10.0)
            for agent_idx in miners:
                collaboration_rewards[agent_idx] += base_reward
    
    # 2. 识别信息分享协作 - 当智能体通过通信传递的信息找到资源
    for i in range(num_agents):
        if self.agent_energy[i] <= 0:
            continue
            
        # 检查每个智能体的通信
        if "communicate" in actions[i] and np.any(actions[i]["communicate"] != 0):
            comm_vector = actions[i]["communicate"]
            
            # 识别通信可能指向的小行星（简化模型：假设通信向量指向资源方向）
            # 实际上，通信内容的解释需要在训练过程中学习
            for j, asteroid_pos in enumerate(self.asteroid_positions):
                if self.asteroid_resources[j] <= 0:
                    continue
                    
                # 计算智能体到小行星的方向向量
                to_asteroid = asteroid_pos - self.agent_positions[i]
                if np.linalg.norm(to_asteroid) > 0:
                    to_asteroid = to_asteroid / np.linalg.norm(to_asteroid)
                
                # 通信向量和小行星方向之间的相似度
                if np.linalg.norm(comm_vector) > 0:
                    comm_direction = comm_vector / np.linalg.norm(comm_vector)
                    similarity = np.dot(to_asteroid[:2], comm_direction)
                    
                    # 如果通信似乎指向小行星，记录这种关联
                    if similarity > 0.7:  # 相似度阈值
                        # 其他智能体接收信息并采取行动
                        for k in range(num_agents):
                            if k == i or self.agent_energy[k] <= 0:
                                continue
                                
                            # 检查接收者是否向该方向移动
                            if "thrust" in actions[k]:
                                thrust_direction = actions[k]["thrust"]
                                if np.linalg.norm(thrust_direction) > 0:
                                    thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
                                    
                                    # 如果接收者朝着相似方向移动
                                    thrust_similarity = np.dot(to_asteroid[:2], thrust_direction[:2])
                                    if thrust_similarity > 0.6:
                                        # 奖励发送者和接收者
                                        collaboration_rewards[i] += 0.5  # 发送有用信息的奖励
                                        collaboration_rewards[k] += 0.3  # 利用信息的奖励
                                        
                                        # 如果接收者后来真的找到了资源，那么奖励更高
                                        distance_to_asteroid = np.linalg.norm(self.agent_positions[k] - asteroid_pos)
                                        if distance_to_asteroid < self.prev_distances_to_asteroids[k][j]:
                                            collaboration_rewards[i] += 0.5
                                            collaboration_rewards[k] += 0.5
    
    # 3. 角色分工奖励 - 在开采和递送之间平衡角色
    miners_count = sum(1 for i in range(num_agents) 
                       if self.agent_energy[i] > 0 and 
                       self.agent_inventories[i] < self.prev_agent_inventories[i])
    
    transporters_count = sum(1 for i in range(num_agents)
                           if self.agent_energy[i] > 0 and
                           np.linalg.norm(self.agent_positions[i] - self.mothership_pos) < 
                           self.prev_distances_to_mothership[i] and
                           self.agent_inventories[i] > 5.0)
    
    # 如果存在明确的角色分工，奖励所有智能体
    if miners_count > 0 and transporters_count > 0:
        role_division_reward = 0.5
        for i in range(num_agents):
            if self.agent_energy[i] > 0:
                collaboration_rewards[i] += role_division_reward
    
    return collaboration_rewards

def _compute_energy_efficiency_reward(self, actions, observations, info):
    """计算能量效率奖励组件"""
    num_agents = len(actions)
    energy_rewards = np.zeros(num_agents, dtype=np.float32)
    
    for i in range(num_agents):
        if self.agent_energy[i] <= 0:
            continue
            
        # 计算能量消耗
        energy_used = self.prev_agent_energy[i] - self.agent_energy[i]
        if energy_used <= 0:
            continue  # 如果能量增加（在母舰处充电），跳过
        
        # 能量效率 = 资源收益 / 能量消耗
        resources_gained = max(0, self.agent_inventories[i] - self.prev_agent_inventories[i])
        
        if energy_used > 0 and resources_gained > 0:
            efficiency = resources_gained / energy_used
            energy_rewards[i] += efficiency * 2.0  # 缩放系数
        
        # 低能量时返回母舰的激励
        distance_to_mothership = np.linalg.norm(self.agent_positions[i] - self.mothership_pos)
        prev_distance = self.prev_distances_to_mothership[i]
        
        # 如果能量低且正在接近母舰，给予奖励
        if self.agent_energy[i] < 30.0 and distance_to_mothership < prev_distance:
            # 奖励与能量剩余成反比，与距离减少成正比
            energy_factor = (30.0 - self.agent_energy[i]) / 30.0
            distance_factor = (prev_distance - distance_to_mothership) / prev_distance
            energy_rewards[i] += 1.0 * energy_factor * distance_factor
        
        # 惩罚不必要的能量消耗（高能量状态下的过度推力）
        if self.agent_energy[i] > 70.0 and "thrust" in actions[i]:
            thrust_magnitude = np.linalg.norm(actions[i]["thrust"])
            if thrust_magnitude > 0.8:  # 过大的推力
                energy_rewards[i] -= 0.2 * thrust_magnitude
    
    return energy_rewards

def _compute_exploration_reward(self, actions, observations, info):
    """计算探索奖励组件"""
    num_agents = len(actions)
    exploration_rewards = np.zeros(num_agents, dtype=np.float32)
    
    # 初始化小行星距离矩阵（如果不存在）
    if not hasattr(self, 'prev_distances_to_asteroids'):
        self.prev_distances_to_asteroids = np.ones((num_agents, len(self.asteroid_positions))) * float('inf')
        for i in range(num_agents):
            for j, asteroid_pos in enumerate(self.asteroid_positions):
                self.prev_distances_to_asteroids[i, j] = np.linalg.norm(self.agent_positions[i] - asteroid_pos)
    
    # 更新当前距离矩阵
    current_distances = np.ones((num_agents, len(self.asteroid_positions))) * float('inf')
    for i in range(num_agents):
        if self.agent_energy[i] <= 0:
            continue
            
        # 计算到每个小行星的距离
        for j, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[j] <= 0:
                continue
                
            distance = np.linalg.norm(self.agent_positions[i] - asteroid_pos)
            current_distances[i, j] = distance
            
            # 如果发现新小行星（首次接近到观察范围内），给予奖励
            if distance <= self.observation_radius and self.prev_distances_to_asteroids[i, j] > self.observation_radius:
                asteroid_id = (asteroid_pos[0], asteroid_pos[1], asteroid_pos[2])  # 小行星唯一标识
                if asteroid_id not in self.discovered_asteroids[i]:
                    self.discovered_asteroids[i].add(asteroid_id)
                    exploration_rewards[i] += 1.0
            
            # 如果接近未知小行星，给予小幅奖励
            elif self.prev_distances_to_asteroids[i, j] > distance:
                exploration_rewards[i] += 0.05
    
    # 更新小行星距离历史
    self.prev_distances_to_asteroids = current_distances
    
    return exploration_rewards

def _compute_communication_reward(self, actions, observations, info):
    """计算通信奖励/惩罚组件"""
    num_agents = len(actions)
    communication_rewards = np.zeros(num_agents, dtype=np.float32)
    
    for i in range(num_agents):
        if self.agent_energy[i] <= 0 or "communicate" not in actions[i]:
            continue
            
        # 获取通信内容
        comm_message = actions[i]["communicate"]
        
        # 惩罚过度通信（过高的通信带宽使用）
        if np.linalg.norm(comm_message) > 0:
            communication_cost = -0.1 * np.linalg.norm(comm_message)
            communication_rewards[i] += communication_cost
        
        # 奖励变化的、有意义的通信（避免发送相同的消息）
        if hasattr(self, 'prev_communications'):
            prev_message = self.prev_communications[i]
            message_change = np.linalg.norm(comm_message - prev_message)
            
            # 只有当通信消息发生变化且智能体处于特殊状态时才奖励（如发现新资源）
            if message_change > 0.5:
                # 检查是否发现了新资源
                for j, asteroid_pos in enumerate(self.asteroid_positions):
                    if self.asteroid_resources[j] <= 0:
                        continue
                        
                    distance = np.linalg.norm(self.agent_positions[i] - asteroid_pos)
                    # 如果智能体刚刚发现新资源，那么通信变化可能是有意义的
                    if distance <= self.observation_radius and self.prev_distances_to_asteroids[i, j] > self.observation_radius:
                        communication_rewards[i] += 0.5  # 奖励有意义的通信
                        break
    
    return communication_rewards

def _compute_path_planning_reward(self, actions, observations, info):
    """计算路径规划和策略奖励组件"""
    num_agents = len(actions)
    path_rewards = np.zeros(num_agents, dtype=np.float32)
    
    for i in range(num_agents):
        if self.agent_energy[i] <= 0:
            continue
            
        # 获取智能体状态
        inventory = self.agent_inventories[i]
        energy = self.agent_energy[i]
        pos = self.agent_positions[i]
        
        # 1. 满载后直接返回母舰的奖励
        if inventory > 20.0:  # 库存较满
            distance_to_mothership = np.linalg.norm(pos - self.mothership_pos)
            prev_distance = self.prev_distances_to_mothership[i]
            
            # 如果库存满且正在接近母舰，给予奖励
            if distance_to_mothership < prev_distance:
                path_rewards[i] += 0.5 * (inventory / 50.0)  # 库存越满，奖励越高
        
        # 2. 持续高效采矿的奖励
        elif "mine" in actions[i] and actions[i]["mine"] == 1:
            # 检查是否有小行星在附近
            mining_successful = False
            for j, asteroid_pos in enumerate(self.asteroid_positions):
                if self.asteroid_resources[j] <= 0:
                    continue
                    
                distance = np.linalg.norm(pos - asteroid_pos)
                if distance < 3.0:  # 采矿范围
                    mining_successful = True
                    break
            
            if mining_successful:
                # 连续采矿次数越多，奖励越高（鼓励高效采矿）
                self.consecutive_mining[i] += 1
                path_rewards[i] += min(2.0, 0.2 * self.consecutive_mining[i])
            else:
                self.consecutive_mining[i] = 0
        else:
            self.consecutive_mining[i] = 0
        
        # 3. 智能路径选择的奖励
        if "thrust" in actions[i]:
            thrust = actions[i]["thrust"]
            
            # 如果能量低，奖励朝向母舰的移动
            if energy < 30.0:
                to_mothership = self.mothership_pos - pos
                if np.linalg.norm(to_mothership) > 0:
                    to_mothership = to_mothership / np.linalg.norm(to_mothership)
                
                if np.linalg.norm(thrust) > 0:
                    thrust_dir = thrust / np.linalg.norm(thrust)
                    alignment = np.dot(to_mothership, thrust_dir)
                    
                    if alignment > 0.7:  # 运动方向与母舰方向一致
                        path_rewards[i] += 0.3
            
            # 如果库存低，奖励朝向资源丰富的小行星移动
            elif inventory < 10.0:
                # 找到最丰富的小行星
                best_asteroid_idx = -1
                best_value = -float('inf')
                
                for j, asteroid_pos in enumerate(self.asteroid_positions):
                    if self.asteroid_resources[j] <= 0:
                        continue
                        
                    distance = np.linalg.norm(pos - asteroid_pos)
                    # 价值 = 资源量 / 距离
                    value = self.asteroid_resources[j] / max(1.0, distance)
                    
                    if value > best_value:
                        best_value = value
                        best_asteroid_idx = j
                
                if best_asteroid_idx >= 0:
                    to_asteroid = self.asteroid_positions[best_asteroid_idx] - pos
                    if np.linalg.norm(to_asteroid) > 0:
                        to_asteroid = to_asteroid / np.linalg.norm(to_asteroid)
                    
                    if np.linalg.norm(thrust) > 0:
                        thrust_dir = thrust / np.linalg.norm(thrust)
                        alignment = np.dot(to_asteroid, thrust_dir)
                        
                        if alignment > 0.7:  # 运动方向与最佳小行星方向一致
                            path_rewards[i] += 0.3
    
    return path_rewards

def sample_compute_fitness_score(self, actions, observations, info):
    """
    计算整体健康度分数（示例实现），用于评估多智能体系统的整体表现。
    这个分数注重团队绩效而非个体奖励。
    
    Returns:
        float: 团队整体健康度分数
    """
    # 基本分数组件
    total_resources_collected = np.sum(self.agent_inventories)
    active_agents = np.sum(self.agent_energy > 0)
    remaining_resources = np.sum(self.asteroid_resources)
    
    # 加权组合评估指标
    fitness = (
        # 资源收集是主要目标
        total_resources_collected * 10.0 +
        
        # 保持智能体活跃很重要
        active_agents * 20.0 +
        
        # 高能量水平表示良好的能量管理
        np.sum(self.agent_energy) * 0.5 +
        
        # 如果大部分资源已被开采，这是好事
        (1.0 - remaining_resources / np.sum(self.asteroid_resources + 0.0001)) * 50.0
    )
    
    return fitness 