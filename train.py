from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp,train_rl_agent_a2c_mlp,train_rl_agent_td3_mlp
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL agents for ShepherdEnv.")
parser.add_argument("-a", "--algorithm", type=str,
                    choices=["td3", "a2c", "ppo", "all"], default="ppo",
                    help="Choose the algorithm to train: 'td3', 'a2c', 'ppo', or 'all'. Default is 'ppo'.")

parser.add_argument(
    "-s", "--num_sheep",
    type=int,
    default=1,
    help="Number of sheep in the environment.",
)

parser.add_argument(
    "-m", "--max_steps",
    type=int,
    default=500,
    help="Maximum number of steps per episode.",
)

parser.add_argument(
    "-r", "--obstacle_radius",
    type=float,
    default=0.0,
    help="Radius of obstacles in the environment.",
)

parser.add_argument(
    "-c", "--checkpoint_dir",
    type=str,
    default=None,
    help="Directory to load a checkpoint from. If not provided, training starts from scratch.",
)

parser.add_argument(
    "-cl", "--criculam_learning",
    type=bool,
    default=True,
    help="Enable or disable curriculum learning. Default is True.",
)
args = parser.parse_args()


env = ShepherdEnv(n_sheep=args.num_sheep,
                max_steps=args.max_steps,
                obstacle_radius=args.obstacle_radius)
eval_env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius)

if args.algorithm in ["td3", "all"]:
    try:
        print(f"Training with TD3 algorithm (#sheep: {env.n_sheep})...")
        model = train_rl_agent_td3_mlp(env, eval_env, timesteps=2000000,
                                       checkpoint_dir=args.checkpoint_dir,
                                       criculam_learning=args.criculam_learning)
        model.save(f"models/td3_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
    except Exception as e:
        print(f"TD3 training failed: {e}")

if args.algorithm in ["a2c", "all"]:
    try:
        print(f"Training with A2C algorithm (#sheep: {env.n_sheep})...")
        model = train_rl_agent_a2c_mlp(env, eval_env, timesteps=2000000)
        model.save(f"models/a2c_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
    except Exception as e:
        print(f"A2C training failed: {e}")

if args.algorithm in ["ppo", "all"]:
    try:
        print(f"Training with PPO algorithm (#sheep: {env.n_sheep})...")
        model = train_rl_agent_ppo_mlp(env, eval_env, timesteps=2000000)
        model.save(f"models/ppo_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
    except Exception as e:
        print(f"PPO training failed: {e}")
