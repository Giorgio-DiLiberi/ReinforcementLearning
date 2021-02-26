import os
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import time

#from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import EvalCallback

## Importing linear function to define a variable cliprange and learning rate
from custom_modules.learning_schedules import linear_schedule
from quadcoptV4 import QuadcoptEnvV4

# Definition of Hyperparameters
## clip_range and learning rates are now variable, linear with learning progress:
# see custom_modules or common  
LearningTimeSteps = 10**6 ## Time step size for policy evaluation and deployment is 0.1 s

LearningRate_ini = 1.0e-4 # LR initial value for linear interpolation
#LearningRate_fin = 1.0e-8 # LR final value for linear interpolation
LearningRate = linear_schedule(LearningRate_ini)

clipRange_ini = 0.2 # Clip initial value for linear interpolation
#clipRange_fin = 1.e-4 # LR final value for linear interpolation
cliprange = linear_schedule(clipRange_ini)

if __name__ == '__main__':
    #log_dir = "Tensorflow_logs/"
    #os.makedirs(log_dir, exist_ok=True)

    ### CREATION OF VECTORIZED ENVIRONMENT

    cpu = 4

    # Creating the environment parallelized to use all 4 threads
    env = SubprocVecEnv([lambda : QuadcoptEnvV4() for num in range(cpu)], start_method='spawn')

    ### AGENT MODEL AND CALLBACK DEFINITION

    eval_env = DummyVecEnv([lambda : QuadcoptEnvV4()]) # Definition of one evaluation environment
    eval_callback = EvalCallback(eval_env, best_model_save_path='./EvalClbkLogs/',
                             log_path='./EvalClbkLogs/npyEvals/', n_eval_episodes=1, eval_freq= 6000,
                             deterministic=True, render=False)

    model = PPO2(MlpPolicy, env, verbose=1, learning_rate=LearningRate,
            cliprange=cliprange, tensorboard_log="./tensorboardLogs/", nminibatches=2,
            noptepochs=2, n_steps=500)
    

    ################################################
    # Train the agent and take the time for learning
    ################################################

    t = time.localtime()
    Learning_time_start_sec= t[5]+ 60*t[4] + 3600*t[3] # take the time
    del t

    print("Learning process start...")

    model.learn(total_timesteps=LearningTimeSteps, callback=eval_callback)

    t = time.localtime()
    Learning_time_finish_sec= t[5]+ 60*t[4] + 3600*t[3]
    del t
    Time_for_learning = Learning_time_finish_sec - Learning_time_start_sec

    print("Learning process for ", LearningTimeSteps, "time steps\n",\
        "completed in ", Time_for_learning, "seconds!")

    ################################################
    #####        LEARNING PROCESS END      #########
    ################################################

    ### MODEL SAVING

    print("Model saving...")

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

    
        
    

    







