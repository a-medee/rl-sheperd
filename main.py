import time
from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd
from stable_baselines3 import PPO,A2C,TD3
import numpy as np

# --- Configuration ---
LEVEL = 1         # 1,2,3,4

# Initialize environment and agent
env = ShepherdEnv(level=LEVEL)
agent = RuleBasedShepherd(env_init=env,drive_distance=5)

# agent = PPO.load(f"shepherd_level{LEVEL}_ppo_mlp", env=env)

obs = env.reset()
done = False

while not done:
    actions = []
    for i in range(env.n_shepherds):
        # Rule-based action
        a = agent.act(obs, env.n_sheep, env.n_shepherds, i)
        # a, _ = agent.predict(obs, deterministic=True)
        actions.extend([a])
    actions = np.array(actions)

    # Step environment
    obs, reward, done, _ = env.step(actions)

    # Render
    env.render()

print(f"Simulation finished! Reward: {reward}")
  
env.close()
