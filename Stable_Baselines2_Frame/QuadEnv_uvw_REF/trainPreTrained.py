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

## Code to train a pretrained model

## Importing linear function to define a variable cliprange and learning rate
from custom_modules.learning_schedules import linear_schedule
from quadcopt_6DOF import QuadcoptEnv_6DOF

# Definition of Hyperparameters
## clip_range and learning rates are now variable, linear with learning progress:
# see custom_modules or common  
LearningTimeSteps = 10 * (10**5) ## Time step size for policy evaluation and deployment is 0.1 s

LearningRate_ini = 5e-4 # LR initial value for linear interpolation
#LearningRate_fin = 1.0e-8 # LR final value for linear interpolation
LearningRate = linear_schedule(LearningRate_ini)

cliprange_ini = 0.2 # Clip initial value for linear interpolation
#clipRange_fin = 1.e-4 # LR final value for linear interpolation
cliprange = linear_schedule(cliprange_ini)

if __name__ == '__main__':
    #log_dir = "Tensorflow_logs/"
    #os.makedirs(log_dir, exist_ok=True)

    ### CREATION OF VECTORIZED ENVIRONMENT

    cpu = 4

    # Creating the environment parallelized to use all 4 threads
    env = SubprocVecEnv([lambda : QuadcoptEnv_6DOF(Random_reset=True, Process_perturbations=True) for num in range(cpu)], start_method='spawn')

    ### AGENT MODEL AND CALLBACK DEFINITION

    eval_env = DummyVecEnv([lambda : QuadcoptEnv_6DOF(Random_reset=True, Process_perturbations=True)]) # Definition of one evaluation environment
    eval_callback = EvalCallback(eval_env, best_model_save_path='./EvalClbkLogs/',
                             log_path='./EvalClbkLogs/npyEvals/', n_eval_episodes=1, eval_freq= 8156,
                             deterministic=True, render=False)

    
    ## Model loading section: asks to decide if continue to train a last policy, a specific policy
    # or the best saved policy
    ######################################
    ##      POLICY LOADING SECTION      ##
    ######################################

    # use of os.path.exists() to check and load the last policy evaluated by training
    # function.
    print("Policy loading...")

    Policy_loading_mode = input("Insert loading mode\nlast: loads last policy saved\nbest: loads best policy saved\nsel: loads a specified policy\n-----> ")

    if Policy_loading_mode == "last":
        for i in range(100, 0, -1): ## function look for the last policy evaluated.
            fileName_toFind = "/home/giorgio/Scrivania/Python/ReinforcementLearning/Stable_Baselines2_Frame/Trivial_problems/QuadEnvTest_6DOF/Policies/PPO_Quad_" + str(i) + ".zip"

            if os.path.exists(fileName_toFind):
                print("last policy found is PPO_Quad_", i)

                Policy2Load = "Policies/PPO_Quad_" + str(i)
                
                break

    

    elif Policy_loading_mode == "best":

        Policy2Load = "EvalClbkLogs/best_model.zip" # best policy name

    else:

        Policy2Load  = input("enter the relative path of policy to load (check before if exists): ")
  
    
    model = PPO2.load(Policy2Load, env, verbose=1, learning_rate=LearningRate, ent_coef=5e-8, lam=0.99,
            cliprange=cliprange, nminibatches=4, gamma=0.9999, noptepochs=16, n_steps=8156, n_cpu_tf_sess=4)

    model.tensorboard_log="./tensorboardLogs/"

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
        filename_check = "/home/giorgio/Scrivania/Python/ReinforcementLearning/Stable_Baselines2_Frame/Trivial_problems/QuadEnvTest_6DOF/Policies/PPO_Quad_" + str(i) + ".zip"
        print("file number ", i, " == ", os.path.exists(filename_check))

        if os.path.exists(filename_check) == False:
            ## checks for the first number available, creates the file with this name and exits for cycle
            filename_toSave = "Policies/PPO_Quad_" + str(i)

            model.save(filename_toSave)
            print("New policy ", filename_toSave, " correctly saved!")
            break