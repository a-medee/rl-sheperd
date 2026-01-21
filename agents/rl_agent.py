from stable_baselines3 import PPO,A2C,TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_ppo_mlp(env,eval_env, timesteps=2000000):
    # model = PPO("MlpPolicy", env, verbose=1,device='cpu', tensorboard_log="./ppo_shepherd_logs/")
    log_dir = f"./logs/ppo/"
    model = PPO(
                "MlpPolicy",
                env,
                n_steps=2048,
                ent_coef=0.003,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=1,
                tensorboard_log=log_dir
                )
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000,n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_a2c_mlp(env, eval_env, timesteps=5000000):
    log_dir = f"./logs/a2c/"
    model = A2C("MlpPolicy", env, verbose=1,tensorboard_log=log_dir)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000,n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=5000000):
    log_dir = f"./logs/td3/"
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, verbose=1,tensorboard_log=log_dir, action_noise=action_noise)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000,n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

