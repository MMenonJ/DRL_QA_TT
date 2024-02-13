from Environment_group_fixed_length_BM25 import *
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import  SubprocVecEnv
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == '__main__':  
    tensor_log_dir = "./tensorboard_logs/DQN_MLP_BM25/"
    save_path = "./model_checkpoints/"
    model_name = "DQN_MLP_BM25"
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
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                        net_arch=[512, 128, 64])
    model = DQN("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, 
                gradient_steps=-1,tensorboard_log= tensor_log_dir, buffer_size=100000, learning_starts=10000)#, learning_starts=50000, target_update_interval=10000

    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)

    model.save("DQN_MLP_BM25_"+str(total_timesteps))