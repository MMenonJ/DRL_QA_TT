from mestrado.Environment_fixed_length import *
from stable_baselines3 import PPO

env = OTT_QA_GYM()
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=25000)