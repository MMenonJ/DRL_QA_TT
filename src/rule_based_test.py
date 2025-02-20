from Environment_group_fixed_length import *
import numpy as np



rules = [[0,2],[1,2],[0,0,2],[0,1,2],[1,0,2],[1,1,2],[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]] 

for rule in rules:
    print('starting with rule: ' + str(rule))
    env = OTT_QA_GYM(dataset = "dev") # initialize environment
    env.reset() # reset environment
    end_validation = False
    while  True:
    #for i in range(10): 
        for action in rule: 
            obs, reward, done, _, info = env.step(action) 
        if end_validation: 
            break
        if done:
            time.sleep(1) 
            env.reset()
            end_validation = env.end_validation
    print('F1-score for ' + str(rule) + ' rules: ' + str(np.mean(env.f1_scores)))
    print('EM for ' + str(rule) + ' rules: ' + str(np.mean(env.em_scores)))
    with open('rules_results.txt', 'a', encoding='utf-8') as my_file:
        my_file.write('F1-score for ' + str(rule) + ' rules: ' + str(np.mean(env.f1_scores)) + '\n')
        my_file.write('EM for ' + str(rule) + ' rules: ' + str(np.mean(env.em_scores)) + '\n')
    del env
    time.sleep(60) 
    #print("action: ", action, "obs: ", obs, "reward: ", reward, "done: ", done, "info: ", info)
