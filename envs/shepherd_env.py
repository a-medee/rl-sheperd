import gym
from gym import spaces
import numpy as np
import pygame

class Shepherd:
    def __init__(self, pos, wheel_base=15.0):
        self.pos = np.array(pos, dtype=float)
        self.theta = np.random.uniform(0, 2*np.pi) # Heading in radians
        self.wheel_base = wheel_base

class Sheep:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.is_safe = False

class ShepherdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level=1, n_sheep=5, screen_size=600, goal_radius=60, max_steps=1000):
        super().__init__()
        self.level = level
        self.n_sheep = n_sheep # Must be fixed for RL
        self.n_shepherds = 1 if level <= 2 else 2
        self.screen_size = screen_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.max_wheel_velocity = 5.0

        # Define Observation Space
        # Sheep (2*N) + Shepherds (3*N: x, y, theta) + Goals (2*N_goals)
        n_goals = 1 if level < 4 else 2
        obs_dim = (2 * self.n_sheep) + (3 * self.n_shepherds) + (2 * n_goals)
        
        self.observation_space = spaces.Box(low=0, high=screen_size, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: [Left Wheel Vel, Right Wheel Vel] for each shepherd
        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.n_shepherds,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.obstacle_size = 120
        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False
        
        # Initialize Sheeps (ensure they aren't in the goal immediately)
        self.sheep = [Sheep(np.random.rand(2) * (self.screen_size * 0.8) + 50) for _ in range(self.n_sheep)]
        
        # Initialize Shepherds
        self.shepherds = [Shepherd(np.random.rand(2) * self.screen_size) for _ in range(self.n_shepherds)]

        # Goals
        if self.level < 4:
            self.goals = [np.array([self.screen_size/2, self.screen_size/2])] # Central goal for Level 1-3
        else:
            self.goals = [np.array([100, 100]), np.array([500, 500])]

        self.obstacles = []
        if self.level >= 2:
            self.obstacles.append(np.array([self.screen_size/2 - 50, self.screen_size/2 + 50]))

        return self._get_obs()

    def _get_obs(self):
        sheep_pos = np.array([s.pos for s in self.sheep]).flatten() / self.screen_size
        shep_data = []
        for sh in self.shepherds:
            shep_data.extend([sh.pos[0]/self.screen_size, sh.pos[1]/self.screen_size, sh.theta/(2*np.pi)])
        goals_pos = np.array(self.goals).flatten() / self.screen_size
        
        return np.concatenate([sheep_pos, shep_data, goals_pos]).astype(np.float32)

    def _update_sheep(self):
        for s in self.sheep:
            if s.is_safe: continue
            
            move = np.zeros(2)
            
            # 1. Check if safe
            for g in self.goals:
                if np.linalg.norm(s.pos - g) < self.goal_radius:
                    s.is_safe = True
            
            # 2. React to shepherds (Repulsion)
            for sh in self.shepherds:
                vec = s.pos - sh.pos
                dist = np.linalg.norm(vec)
                if dist < 80:
                    move += (vec / (dist + 1e-6)) * 4
            
            # 3. Ambient movement
            if np.linalg.norm(move) < 0.1:
                move += np.random.uniform(-1, 1, 2)

            # Obstacle Collision (Improved)
            new_pos = s.pos + move
            for obs in self.obstacles:
                if (obs[0] < new_pos[0] < obs[0] + self.obstacle_size) and \
                   (obs[1] < new_pos[1] < obs[1] + self.obstacle_size):
                    move = -move # Bounce
            
            s.pos = np.clip(s.pos + move, 10, self.screen_size - 10)

    def _update_shepherds(self, actions):
        # actions is a flat array [sh1_left, sh1_right, sh2_left, sh2_right...]
        for i, sh in enumerate(self.shepherds):
            v_l = actions[i*2] * self.max_wheel_velocity
            v_r = actions[i*2 + 1] * self.max_wheel_velocity
            
            # Differential Drive Kinematics
            v = (v_l + v_r) / 2.0
            omega = (v_r - v_l) / sh.wheel_base
            
            sh.theta += omega
            sh.pos[0] += v * np.cos(sh.theta)
            sh.pos[1] += v * np.sin(sh.theta)
            sh.pos = np.clip(sh.pos, 0, self.screen_size)

    def step(self, actions):
        self.steps += 1
        
        self._update_shepherds(actions)
        self._update_sheep()
        
        # Reward Logic (Shaping)
        reward = 0
        num_safe = sum([1 for s in self.sheep if s.is_safe])
        
        # Reward for sheep being safe
        reward += num_safe * 10.0 
        
        # Small penalty for time to encourage speed
        reward -= 0.1 
        
        # Reward Shaping: Distance from sheep to goal
        for s in self.sheep:
            if not s.is_safe:
                dist_to_goal = np.linalg.norm(s.pos - self.goals[0])
                reward -= dist_to_goal * 0.001 # Move sheep closer
        
        # End conditions
        if num_safe == self.n_sheep or self.steps >= self.max_steps:
            self.done = True
            
        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 240, 240))
        
        # Draw Goals
        for g in self.goals:
            pygame.draw.circle(self.screen, (200, 255, 200), g.astype(int), self.goal_radius)
        
        # Draw Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (*obs, self.obstacle_size, self.obstacle_size))

        # Draw Sheep
        for s in self.sheep:
            color = (0, 150, 0) if s.is_safe else (50, 50, 50)
            pygame.draw.circle(self.screen, color, s.pos.astype(int), 5)

        # Draw Shepherds (with heading indicator)
        for sh in self.shepherds:
            pygame.draw.circle(self.screen, (200, 0, 0), sh.pos.astype(int), 10)
            # Draw line to show direction
            end_line = sh.pos + [15 * np.cos(sh.theta), 15 * np.sin(sh.theta)]
            pygame.draw.line(self.screen, (0,0,0), sh.pos, end_line, 3)

        pygame.display.flip()
        self.clock.tick(60)