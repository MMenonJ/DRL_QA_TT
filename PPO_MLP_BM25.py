from Environment_group_fixed_length_BM25 import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor 

class MLP(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.linear1 = nn.Sequential(nn.Linear(768*11, 512), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

    def forward(self, x):
        x =self.linear1(x)
        x =self.linear2(x)
        return x

if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/PPO_MLP_BM25/"
    save_path = "./model_checkpoints/"
    model_name = "PPO_MLP_BM25"
    total_timesteps = 200000
    #num_proc = 4  # Number of processes to use
    #h = [lambda: OTT_QA_GYM() for i in range(num_proc)]
    #env = SubprocVecEnv(h, start_method = 'forkserver')
    env = OTT_QA_GYM()
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    policy_kwargs = dict(
        features_extractor_class=MLP,
        net_arch=dict(pi=[64,32], vf=[64,32]),
        features_extractor_kwargs={},
    )
    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, 
                n_steps= 128, batch_size=32, n_epochs=60,tensorboard_log= tensor_log_dir)

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)

    model.save("PPO_MLP_BM25_"+str(total_timesteps))