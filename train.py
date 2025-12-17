from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp,train_rl_agent_a2c_mlp,train_rl_agent_td3_mlp



for level in [1,2,3,4]:

    env = ShepherdEnv(level=level)
    model = train_rl_agent_td3_mlp(env, timesteps=1000000)
    model.save(f"models/shepherd_level{level}_td3_mlp")

    env = ShepherdEnv(level=level)
    model = train_rl_agent_a2c_mlp(env, timesteps=1000000)
    model.save("models/shepherd_level{level}_a2c_mlp")

    env = ShepherdEnv(level=level)
    model = train_rl_agent_ppo_mlp(env, timesteps=1000000)
    model.save("models/shepherd_level{level}_ppo_mlp")

