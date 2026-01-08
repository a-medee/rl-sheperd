from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp,train_rl_agent_a2c_mlp,train_rl_agent_td3_mlp
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL agents for ShepherdEnv.")
parser.add_argument("-a", "--algorithm", type=str, choices=["td3", "a2c", "ppo", "all"], default="all",
                    help="Choose the algorithm to train: 'td3', 'a2c', 'ppo', or 'all'. Default is 'all'.")
args = parser.parse_args()

for level in [1, 2, 3, 4]:
    print(f"\n\n ----- LEVEL {level} -----")

    if args.algorithm in ["td3", "all"]:
        try:
            env = ShepherdEnv(level=level, n_sheep=5)
            eval_env = ShepherdEnv(level=level, n_sheep=5)
            print(f"\t ----- LEVEL {level} / TD3  (#sheep:{env.n_sheep})-----")
            model = train_rl_agent_td3_mlp(env, eval_env, timesteps=1000000)
            model.save(f"models/shepherd_level{level}_td3_mlp")
        except:
            print(f"!!! LEVEL {level} / TD3  Failed")

    if args.algorithm in ["a2c", "all"]:
        try:
            env = ShepherdEnv(level=level, n_sheep=5)
            eval_env = ShepherdEnv(level=level, n_sheep=5)
            print(f"\t ----- LEVEL {level} / A2C (#sheep:{env.n_sheep})-----")
            model = train_rl_agent_a2c_mlp(env, eval_env, timesteps=1000000)
            model.save(f"models/shepherd_level{level}_a2c_mlp")
        except:
            print(f"!!! LEVEL {level} / A2C  Failed")

    if args.algorithm in ["ppo", "all"]:
        try:
            env = ShepherdEnv(level=level, n_sheep=5)
            eval_env = ShepherdEnv(level=level, n_sheep=5)
            print(f"\t ----- LEVEL {level} / PPO (#sheep:{env.n_sheep})-----")
            model = train_rl_agent_ppo_mlp(env, eval_env, timesteps=1000000)
            model.save(f"models/shepherd_level{level}_ppo_mlp")
        except:
            print(f"!!! LEVEL {level} / PPO  Failed")


