from Environment_group_fixed_length_BM25 import *
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRU(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.gru = nn.GRU(input_size=768, hidden_size=768, num_layers=2, dropout=0.1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(768, 128), nn.ReLU())
    def forward(self, x):
        x = x.reshape(-1, 11, 768)
        x, _ = self.gru(x)
        x = self.linear(x[:, -1, :])
        return x

if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/DQN_GRU_BM25/"
    save_path = "./model_checkpoints/"
    model_name = "DQN_GRU_BM25"
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
        features_extractor_class=GRU,
        net_arch=[64],
        features_extractor_kwargs={},
    )

    model = DQN("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, 
                gradient_steps=-1,tensorboard_log= tensor_log_dir, buffer_size=100000, learning_starts=10000)#, learning_starts=50000, target_update_interval=10000

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)

    model.save("DQN_GRU_BM25_"+str(total_timesteps))