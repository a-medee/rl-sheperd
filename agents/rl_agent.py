from stable_baselines3 import PPO,A2C,TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_ppo_mlp(env,eval_env, timesteps=500000):
    # model = PPO("MlpPolicy", env, verbose=1,device='cpu', tensorboard_log="./ppo_shepherd_logs/")
    model = PPO("MlpPolicy", env, verbose=1,device='cpu')
    model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="logs/ppo",
            n_steps=5000,  # Smaller buffer = more frequent log updates
            device='cpu'
        )
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/ppo/',
                                log_path='./logs/ppo/', eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_a2c_mlp(env, eval_env, timesteps=500000):
    model = A2C("MlpPolicy", env, verbose=1,tensorboard_log="logs/a2c")
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/a2c/',
                                log_path='./logs/a2c/', eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=500000):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, verbose=1,tensorboard_log="logs/td3", action_noise=action_noise)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/td3/',
                                log_path='./logs/td3/', eval_freq=5000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

