from stable_baselines3 import PPO,A2C,TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_ppo_mlp(env,eval_env, timesteps=500000):
    # model = PPO("MlpPolicy", env, verbose=1,device='cpu', tensorboard_log="./ppo_shepherd_logs/")
    log_dir = f"./logs/ppo/level_{env.level}/"
    model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir,
            n_steps=5000,  # Smaller buffer = more frequent log updates
            device='cpu'
        )
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_a2c_mlp(env, eval_env, timesteps=500000):
    log_dir = f"./logs/a2c/level_{env.level}/"
    model = A2C("MlpPolicy", env, verbose=1,tensorboard_log=log_dir)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=500000):
    log_dir = f"./logs/td3/level_{env.level}/"
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, verbose=1,tensorboard_log=log_dir, action_noise=action_noise)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

