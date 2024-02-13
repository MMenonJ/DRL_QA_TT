from Environment_group_fixed_length_BM25 import *
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, dropout=0.1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(768, 128), nn.ReLU())
    def forward(self, x):
        x = x.reshape(-1, 11, 768)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/DQN_LSTM_BM25/"
    save_path = "./model_checkpoints/"
    model_name = "DQN_LSTM_BM25"
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
        features_extractor_class=LSTM,
        net_arch=[64],
        features_extractor_kwargs={},
    )

    model = DQN.load("model_checkpoints/DQN_LSTM_BM25_80000_steps", env=env, tensorboard_log = tensor_log_dir)#, learning_starts=50000, target_update_interval=10000
    model.load_replay_buffer("model_checkpoints/DQN_LSTM_BM25_replay_buffer_80000_steps")

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback, reset_num_timesteps = False)

    model.save("DQN_LSTM_BM25_"+str(total_timesteps))