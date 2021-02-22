import os
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import time

#from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from quadcoptV4 import QuadcoptEnvV4

# Definition of Hyperparameters
LearningRate = 1.0e-4
LearningTimeSteps = 10**6

if __name__ == '__main__':
    #log_dir = "Tensorflow_logs/"
    #os.makedirs(log_dir, exist_ok=True)

    cpu = 4

    # Creating the environment parallelized to use all 4 threads
    env = SubprocVecEnv([lambda : QuadcoptEnvV4() for num in range(cpu)], start_method='spawn')

    model = PPO2(MlpPolicy, env, verbose=1, learning_rate=LearningRate, tensorboard_log="./tensorboardLogs/")
    
    ################################################
    # Train the agent and take the time for learning
    t = time.localtime()
    Learning_time_start_sec= t[5]+ 60*t[4] + 3600*t[3] # take the time
    del t

    print("Learning process start...")

    model.learn(total_timesteps=LearningTimeSteps)

    t = time.localtime()
    Learning_time_finish_sec= t[5]+ 60*t[4] + 3600*t[3]
    del t
    Time_for_learning = Learning_time_finish_sec - Learning_time_start_sec

    print("Learning process for ", LearningTimeSteps, "time steps",\
        "completed in ", Time_for_learning, "seconds!")
    ##### LEARNING PROCESS end
    ################################################

    print("Model saving...")
    ## MODEL SAVING
    for i in range(1, 100): ## policies name format "PPO_Quad_<numberOfAttempt>.zip"

        # check for file existance
        filename_check = "/home/giorgio/Scrivania/Python/ReinforcementLearning/Stable-Baselines2_Frame/QuadEnv4/Policies/PPO_Quad_" + str(i) + ".zip"
        print("file number ", i, " == ", os.path.exists(filename_check))

        if os.path.exists(filename_check) == False:
            ## checks for the first number available, creates the file with this name and exits for cycle
            filename_toSave = "Policies/PPO_Quad_" + str(i)

            model.save(filename_toSave)
            print("New policy ", filename_toSave, " correctly saved!")
            break

    
        
    

    







