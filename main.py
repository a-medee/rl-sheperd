import time
from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd
from stable_baselines3 import PPO,A2C,TD3
import numpy as np
import argparse

# --- Configuration ---
LEVEL = 1         # 1,2,3,4
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Shepherd Environment Simulation")
parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=1, help="Level of the environment (1-4)")
parser.add_argument("--model", type=str, choices=["ruleBase", "PPO", "A2C", "TD3"], default="ruleBase", help="Model to use (ruleBase, PPO, A2C, TD3)")
args = parser.parse_args()

LEVEL = args.level
MODEL = args.model

# Initialize environment and agent
env = ShepherdEnv(level=LEVEL,n_sheep=4)
print(f"Environment initialized with {env.n_sheep} sheep and {env.n_shepherds} shepherd(s).")
if MODEL == "ruleBase":
    agent = RuleBasedShepherd(env_init=env,drive_distance=5)
    print("Using Rule-Based Shepherd Agent.")
elif MODEL == "PPO":
    agent = PPO.load(f"models/shepherd_level{LEVEL}_ppo_mlp", env=env)
    print("Using PPO Agent.")
elif MODEL == "A2C":
    agent = A2C.load(f"models/shepherd_level{LEVEL}_a2c_mlp", env=env)
    print("Using A2C Agent.")
elif MODEL == "TD3":
    agent = TD3.load(f"models/shepherd_level{LEVEL}_td3_mlp", env=env)
    print("Using TD3 Agent.")
else:
    raise ValueError("Invalid model type specified.")

obs = env.reset()
done = False

while not done:
    actions = []
    for i in range(env.n_shepherds):
        if MODEL != "ruleBase":
            # RL-based action
            a = agent.act(obs, env.n_sheep, env.n_shepherds, i)
        else:
            a, _ = agent.predict(obs, deterministic=True)
        actions.extend([a])
    actions = np.array(actions)

    # Step environment
    obs, reward, done, _ = env.step(actions)

    # Render
    env.render()

print(f"Simulation finished! Reward: {reward}")
  
env.close()
