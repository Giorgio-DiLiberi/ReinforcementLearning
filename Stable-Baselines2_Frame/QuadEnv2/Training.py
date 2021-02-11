import os

import gym
import numpy as np

#from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from quadcoptV2 import QuadcoptEnvV2


if __name__ == '__main__':
    #log_dir = "Tensorflow_logs/"
    #os.makedirs(log_dir, exist_ok=True)

    cpu = 4

    # Creating the environment parallelized to use all 4 threads
    env = SubprocVecEnv([lambda : QuadcoptEnvV2() for num in range(cpu)], start_method='spawn')

    model = PPO2(MlpPolicy, env, verbose=1, learning_rate=1.0e-4, tensorboard_log="./tensorboardLogs/")
    # Train the agent
    model.learn(total_timesteps=50000)
    model.save("ppo_Quad_1Attempt")





