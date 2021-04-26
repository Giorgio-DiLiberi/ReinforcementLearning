# Code to simulate the environment when trained
import os
#ignore tensorflow warnings
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from Trajectory_gen import Navigation


env = Navigation(Random_reset=False, Process_perturbations=True)

tieme_steps_to_simulate = env.max_Episode_time_steps + 1 ## define the number of timesteps to simulate

######################################
##      POLICY LOADING SECTION      ##
######################################

# use of os.path.exists() to check and load the last policy evaluated by training
# function.
print("Policy loading...")

Policy_loading_mode = input("Insert loading mode\nlast: loads last policy saved\nbest: loads best policy saved\nsel: loads a specified policy\n-----> ")

if Policy_loading_mode == "last":
  for i in range(100, 0, -1): ## function look for the last policy evaluated.
    fileName_toFind = "/home/ghost/giorgio_diliberi/ReinforcementLearning/SBs_2_on_remote/Trajectory_gen/Policies/PPO_Quad_" + str(i) + ".zip"

    if os.path.exists(fileName_toFind):
      print("last policy found is PPO_Quad_" + str(i))

      Policy2Load = "Policies/PPO_Quad_" + str(i)
      
      break

  

elif Policy_loading_mode == "best":

  Policy2Load = "EvalClbkLogs/best_model.zip" # best policy name

else:

  Policy2Load = input("enter the number of policy to load (check before if exists): ")

model = PPO2.load(Policy2Load)
print("Policy ", Policy2Load, " loaded!")

######################################
##     SIMULATION SECTION           ##
######################################
 
#model = PPO2.load("Policies/PPO_Quad_1")  # uncomment this line to load a specific policy instead of the last one

obs = env.reset()

# info vectors initialization for simulation history
info_u = [env.state[0]]
info_v = [env.state[1]]
info_X = [env.state[2]]
info_Y = [env.state[3]]
action_memory = np.array([0., 0.]) ## vector to store actions during the simulation
#Throttle_memory = [env.dTt]
episode_reward = [env.getReward()]

time=0.
info_time=[time] # elapsed time vector

# SIMULATION

for i in range(tieme_steps_to_simulate): #last number is excluded
    
    action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    
    obs, reward, done, info = env.step(action) 

    info_u.append(info["u"])
    info_v.append(info["v"])
    info_X.append(info["X"])
    info_Y.append(info["Y"])
    action_memory = np.vstack([action_memory, action])
    episode_reward.append(reward) # save the reward for all the episode

    time=time + env.timeStep # elapsed time since simulation start
    info_time.append(time)

    #env.render()
    if done:
      # obs = env.reset()
      break

## PLOT AND DISPLAY SECTION

plt.figure(1)
plt.plot(info_time, info_u)
plt.plot(info_time, info_v)
plt.xlabel('time')
plt.ylabel('Velocity [m/s]')
plt.title('u,v')
plt.legend(['u', 'v'])
plt.savefig('SimulationResults/Velocity.jpg')

plt.figure(2)
plt.plot(info_time, info_X)
plt.plot(info_time, info_Y)
plt.xlabel('time')
plt.ylabel('Position NE [m]')
plt.title('X,Y')
plt.legend(['X', 'Y'])
plt.savefig('SimulationResults/Position.jpg')

plt.figure(3)
plt.plot(info_time, episode_reward)
plt.xlabel('time')
plt.ylabel('Reward')
plt.title('Episode Reward')
plt.savefig('SimulationResults/reward.jpg')

plt.figure(4)
plt.plot(info_time, action_memory[:, 0])
plt.plot(info_time, action_memory[:, 1])
plt.xlabel('time')
plt.ylabel('Actions')
plt.title('Actions in episode [-1, 1]')
plt.legend(['tg_theta', 'tg_phi'])
plt.savefig('SimulationResults/action.jpg')

plt.figure(5)
plt.plot(info_X, info_Y)
plt.xlabel('X_Pos [m]')
plt.ylabel('Y_Pos [m]')
plt.title('Trajectory')
plt.savefig('SimulationResults/Trajectory.jpg')