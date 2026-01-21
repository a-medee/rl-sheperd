import argparse
import numpy as np

from stable_baselines3 import PPO, A2C, TD3

from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd,TipsyShepherd,LazyShepherd


def load_agent(model_name: str, env: ShepherdEnv):
    """Load the selected agent."""
    if model_name == "ruleBase":
        print("A Rule-Based Shepherd Agent ...")
        return RuleBasedShepherd()
    if model_name == "tipsy":
        print("A Tipsy Shepherd Agent ...")
        return TipsyShepherd()
    if model_name == "lazy":
        print("A Lazy Shepherd Agent ...")
        return LazyShepherd()

    model_paths = {
        "PPO": f"logs/ppo/best_model.zip",
        "A2C": f"models/a2c_mlp",
        "TD3": f"models/td3_mlp",
    }

    if model_name not in model_paths:
        raise ValueError(f"Unsupported model type: {model_name}")

    print(f"Using {model_name} Agent.")
    return globals()[model_name].load(
        model_paths[model_name],
        env=env,
        device="cpu",
    )


def run_episode(env: ShepherdEnv, agent, model_name: str,display_flag=False) -> float:
    obs = env.reset()
    done = False
    reward = 0.0

    while not done:
        if model_name in ["ruleBase","lazy","tipsy"]:
            actions = agent.act(obs)
        else:
            actions, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(actions)
        if display_flag :
            # print(f"Step {env.steps}: Action:{actions}, Reward: {reward:.2f}")
            env.render()

    return reward


def main():
    parser = argparse.ArgumentParser(description="Shepherd Environment Test Runner")

    parser.add_argument(
        "-a", "--agent",
        type=str,
        choices=["ruleBase", "lazy", "tipsy", "PPO", "A2C", "TD3"],
        default="ruleBase",
        help="Agent model type",
    )

    parser.add_argument(
        "-n", "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run. If greater than 1, renders are disabled and statistics are reported.",
    )

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

    args = parser.parse_args()

    rewards = []

    for eps in range(1, args.num_episodes + 1):
        env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius)
        print(
            f"Environment initialized with "
            f"{env.n_sheep} sheep for episode {eps}"
        )

        agent = load_agent(args.agent, env)
        final_reward = run_episode(env, agent, args.agent,display_flag=(args.num_episodes==1))

        rewards.append(final_reward)
        if args.num_episodes == 1:
            print(f"Episode {eps} finished. Final reward: {final_reward:.2f}")
        env.close()
        
    if args.num_episodes > 1:
        print(f"\nRan {args.num_episodes} episodes.")
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ci = 1.96 * (std_reward / np.sqrt(len(rewards)))  # 95% confidence interval

        print(f"Average reward: {mean_reward:.2f} ± {ci:.2f} (95% CI)")


if __name__ == "__main__":
    main()
