from Environment_group_fixed_length import *
import numpy as np
from stable_baselines3 import DQN, PPO


PPO_model_paths = ["model_checkpoints/PPO_MLP_100000_steps"]

models = []
#for i in range(len(DQN_model_paths)):
#    models.append([DQN.load(DQN_model_paths[i]),DQN_model_paths[i]])
for i in range(len(PPO_model_paths)):
    models.append([PPO.load(PPO_model_paths[i]),PPO_model_paths[i]])

for model,model_name in models:
    print('starting with model: ' + str(model_name))
    env = OTT_QA_GYM(dataset = "dev") # initialize environment
    obs, _ = env.reset() # reset environment
    end_validation = False
    cont = 0
    while  True:
        done = False
        while not done: 
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action) 
        if end_validation: 
            break
        if done:
            time.sleep(1) 
            obs, _ = env.reset()
            end_validation = env.end_validation
        cont += 1
    print('F1-score for ' + str(model_name) + ' rules: ' + str(np.mean(env.f1_scores)))
    print('EM for ' + str(model_name) + ' rules: ' + str(np.mean(env.em_scores)))
    with open('test_models.txt', 'a', encoding='utf-8') as my_file:
        my_file.write('F1-score for ' + str(model_name) + ' rules: ' + str(np.mean(env.f1_scores)) + '\n')
        my_file.write('EM for ' + str(model_name) + ' rules: ' + str(np.mean(env.em_scores)) + '\n')
    del env
    time.sleep(60) 
