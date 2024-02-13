from Environment_group_fixed_length import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, dropout=0.1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
    def forward(self, x):
        x = x.reshape(-1, 11, 512)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/PPO_LSTM/"
    save_path = "./model_checkpoints/"
    model_name = "PPO_LSTM"
    total_timesteps = 600000
    num_proc = 4  # Number of processes to use
    h = [lambda: OTT_QA_GYM() for i in range(num_proc)]
    env = SubprocVecEnv(h, start_method = 'forkserver')

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
        features_extractor_class=LSTM,
        net_arch=dict(pi=[64,32], vf=[64,32]),
        features_extractor_kwargs={},
    )

    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, 
                n_steps= 128, batch_size=32, n_epochs=60,tensorboard_log= tensor_log_dir)

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)

    model.save("PPO_LSTM_"+str(total_timesteps))