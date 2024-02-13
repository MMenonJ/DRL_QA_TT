from Environment_group_fixed_length import *
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class transformer(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        self.transformers = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
    def forward(self, x):
        x = x.reshape(-1, 11, 512)
        x = self.transformers(x)
        x = self.linear(x[:,0,:])
        return x

if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/DQN_transformer/"
    save_path = "./model_checkpoints/"
    model_name = "DQN_transformer"
    total_timesteps = 1000000
    #num_proc = 1  # Number of processes to use
    #h = [lambda: OTT_QA_GYM() for i in range(num_proc)]
    #env = SubprocVecEnv(h, start_method = 'forkserver')

    env = OTT_QA_GYM()
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_path,
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    policy_kwargs = dict(
        features_extractor_class=transformer,
        net_arch=[64],
        features_extractor_kwargs={},
    )

    model = DQN("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, 
                gradient_steps=-1,tensorboard_log= tensor_log_dir, buffer_size=500000)#, learning_starts=50000, target_update_interval=10000

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)

    model.save("DQN_transformer_"+str(total_timesteps))