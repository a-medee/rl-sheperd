from stable_baselines3 import PPO,A2C,TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np

def train_rl_agent_ppo_mlp(env, timesteps=500000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps,log_interval=500)
    return model

def train_rl_agent_a2c_mlp(env, timesteps=500000):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps,log_interval=500)
    return model

def train_rl_agent_td3_mlp(env, timesteps=500000):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env,action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps,log_interval=500)
    return model
