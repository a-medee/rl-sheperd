import gym
from gym import spaces
import numpy as np
import pygame

class Shepherd:
    def __init__(self, pos, wheel_base=10.0, buffer_distance=50):
        self.pos = np.array(pos, dtype=float)
        self.theta = 0.0
        self.wheel_base = wheel_base
        self.buffer_distance = buffer_distance
        self.target_sheep = None

class Sheep:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)

class ShepherdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level=1, n_sheep=0, screen_size=600, sheep_radius=5, goal_radius=100, max_steps=2000, dt=2e-1):
        super().__init__()
        self.level = level
        if n_sheep == 0:
            self.n_sheep  = np.random.randint(1, 10)
        else:
            self.n_sheep = n_sheep
        self.n_shepherds = 1 if level <= 2 else 2
        self.screen_size = screen_size
        self.sheep_radius = sheep_radius
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.dt = dt
        self.max_wheel_velocity = 10.0

        self.bounds = np.array([screen_size, screen_size])

        # Gym spaces
        obs_dim = 2*(n_sheep + self.n_shepherds + self.n_shepherds)
        self.observation_space = spaces.Box(low=0, high=screen_size, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2*self.n_shepherds,), dtype=np.float32)

        # Environment objects
        self.sheep = []
        self.shepherds = []
        self.goals = []
        self.obstacles = []
        self.obstacle_size = 200

        # Pygame
        self.screen = None
        self.clock = None

        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False

        # Sheep
        self.sheep = [Sheep(np.random.rand(2)*self.screen_size) for _ in range(self.n_sheep)]

        # Shepherds
        self.shepherds = [Shepherd(np.random.rand(2)*self.screen_size) for _ in range(self.n_shepherds)]

        # Goals
        if self.level < 4:
            self.goals = [self.goal_radius+np.random.rand(2)*(self.screen_size-2*self.goal_radius) for _ in range(self.n_shepherds)]
        else:
            # Level 4: fixed corners
            self.goals = [np.array([50,50]), np.array([self.screen_size-50, self.screen_size-50])]

        # Obstacles (Level 2+)
        self.obstacles = []
        if self.level >= 2:
            x = np.random.randint(0, self.screen_size - self.obstacle_size)
            y = np.random.randint(0, self.screen_size - self.obstacle_size)
            if self.level > 3:
                while any(np.linalg.norm(np.array([x + self.obstacle_size / 2, y + self.obstacle_size / 2]) - g) < self.goal_radius for g in self.goals):
                    x = np.random.randint(0, self.screen_size - self.obstacle_size)
                    y = np.random.randint(0, self.screen_size - self.obstacle_size)
            self.obstacles.append(np.array([x, y]))

        return self._get_obs()

    def _get_obs(self):
        sheep_pos = np.array([s.pos for s in self.sheep]).flatten()
        shepherd_pos = np.array([sh.pos for sh in self.shepherds]).flatten()
        goals_pos = np.array(self.goals).flatten()
        return np.concatenate([sheep_pos, shepherd_pos, goals_pos]).astype(np.float32)

    def _update_sheep(self):
        repulsion_dist = 50
        for s in self.sheep:
            move = np.zeros(2)
            freeze_move = False

            for g in self.goals:
                if np.linalg.norm(s.pos - g) < self.goal_radius:
                    freeze_move = True
                # Attraction to goal
            if freeze_move:
                continue
            # Repulsion from shepherds
            for sh in self.shepherds:
                vec = sh.pos - s.pos
                dist = np.linalg.norm(vec)
                if dist < repulsion_dist:
                    move -= (vec / (dist+1e-6)) * 8

            if np.linalg.norm(move) < 1e-6:
                # Not near shepherds
                if self.level == 4:
                    # Move toward nearest goal
                    dists = [np.linalg.norm(g - s.pos) for g in self.goals]
                    goal = self.goals[np.argmin(dists)]
                    direction = goal - s.pos
                    move += 0.02 * direction / (np.linalg.norm(direction)+1e-6)+(np.random.rand(2)-0.5)
                else:
                    # Random movement
                    move += 2 * (np.random.rand(2)-0.5)

            

            # Obstacle avoidance
            for obs in self.obstacles:
                if (obs[0] <= s.pos[0]+move[0] <= obs[0]+self.obstacle_size) and \
                   (obs[1] <= s.pos[1]+move[0] <= obs[1]+self.obstacle_size):
                    s.pos += move * -2  # Simple bounce back

            s.pos += move
            s.pos = np.clip(s.pos, [10,10], self.bounds-10)

    def _check_goal(self):
        reached = np.zeros(len(self.sheep), dtype=bool)
        for i, s in enumerate(self.sheep):
            for g in self.goals:
                if np.linalg.norm(s.pos - g) < self.goal_radius:
                    reached[i] = True
        return reached

    def update_wheels(self, actions):
        for i, (sh, a) in enumerate(zip(self.shepherds, actions)):
            # Update position
            sh.pos += a * self.max_wheel_velocity
            # Clip to bounds
            sh.pos = np.clip(sh.pos, [0,0], self.bounds)
            # print(f"Shepherd {i} action: {a}, new pos: {sh.pos} with {a}")

    def reward_function(self):

        reached = self._check_goal()
        if self.level < 4:
            rewards = np.zeros(self.n_shepherds)
            reward = reached.sum()
        else:
            # Level 4 competitive: difference of sheep in own goal
            rewards = np.zeros(self.n_shepherds)
            for i in range(self.n_shepherds):
                goal = self.goals[i]
                rewards[i] = sum(np.linalg.norm(s.pos - goal)<self.goal_radius for s in self.sheep)
            reward = rewards[0] - rewards[1]

        # Done if all sheep reached goal or max steps
        if reached.all() or self.steps >= self.max_steps:
            self.done = True

        return reward,rewards
    
    def step(self, actions):
        self.steps += 1
        if self.done:
            return self._get_obs(), 0, self.done, {}

        # Update shepherds
        self.update_wheels(actions)

        # Update sheep
        self._update_sheep()

        # Compute reward
        reward,_ = self.reward_function()

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255,255,255))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100,100,100), (*obs, self.obstacle_size, self.obstacle_size))

        # Draw goals
        colors_g = [(255,0,0),(0,255,0)]
        for i, g in enumerate(self.goals):
            pygame.draw.circle(self.screen, colors_g[i%len(colors_g)], g.astype(int), self.goal_radius)

        # Draw sheep
        for s in self.sheep:
            pygame.draw.circle(self.screen, (0,0,0), s.pos.astype(int), self.sheep_radius)

        # Draw shepherds
        colors_s = [(255,0,0),(0,255,0)]
        for i, sh in enumerate(self.shepherds):
            pygame.draw.circle(self.screen, colors_s[i%len(colors_s)], sh.pos.astype(int), self.sheep_radius*2)

        # Display reward score for shepherds
        font = pygame.font.Font(None, 24)
        if self.level < 4:
            score_text = font.render(f"Level: {self.level} (Step:{int(self.steps)}), Score: {self.reward_function()[0]}", True, (0, 0, 0))
            self.screen.blit(score_text, (10, 10))
        else:
            score_text  = font.render(f"Level: {self.level}  (Step:{int(self.steps)}) ", True, (0,0,0))
            self.screen.blit(score_text, (10,10))
            for i, sh in enumerate(self.shepherds):
                score_text  = font.render(f"Shepherd {i+1} : {self.reward_function()[1][i]}", True, colors_s[i%len(colors_s)])
                self.screen.blit(score_text, (self.screen_size- 150 - i * 150,10))


        pygame.display.flip()
        self.clock.tick(30)