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
from quadcoptV4 import QuadcoptEnvV4




env = QuadcoptEnvV4()

tieme_steps_to_simulate = env.max_Episode_time_steps ## define the number of timesteps to simulate

######################################
##      POLICY LOADING SECTION      ##
######################################

# use of os.path.exists() to check and load the last policy evaluated by training
# function.
print("Policy loading...")

Policy_loading_mode = input("Insert loading mode\nlast: loads last policy saved\nbest: loads best policy saved\nsel: loads a specified policy\n-----> ")

if Policy_loading_mode == "last":
  for i in range(100, 0, -1): ## function look for the last policy evaluated.
    fileName_toFind = "/home/giorgio/Scrivania/Python/ReinforcementLearning/Stable-Baselines2_Frame/QuadEnv4/Policies/PPO_Quad_" + str(i) + ".zip"

    if os.path.exists(fileName_toFind):
      print("last policy found is PPO_Quad_", i)

      Policy2Load = "Policies/PPO_Quad_" + str(i)
      
      break

  del i

elif Policy_loading_mode == "best":

  Policy2Load = "EvalClbkLogs/best_model.zip" # best policy name

else:

  Policy_number = input("enter the number of policy to load (check before if exists): ")
  Policy2Load = "Policies/PPO_Quad_" + str(Policy_number)
  

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
info_w = [env.state[2]]
info_p = [env.state[3]]
info_q = [env.state[4]]
info_r = [env.state[5]]
info_quaternion = np.array([env.state[6:10]]) # quaternion stored in a np.array matrix
info_X = [env.state[10]]
info_Y = [env.state[11]]
info_Z = [env.state[12]]
episode_reward = [env.getReward()]

time=0.
info_time=[time] # elapsed time vector

# SIMULATION

for i in range(tieme_steps_to_simulate): #last number is excluded
    
    action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    
    obs, reward, done, info = env.step(action) 

    info_u.append(info["u"])
    info_v.append(info["v"])
    info_w.append(info["w"])
    info_p.append(info["p"])
    info_q.append(info["q"])
    info_r.append(info["r"])
    info_quaternion = np.vstack([info_quaternion, [info["q0"], info["q1"], info["q2"], info["q3"]]])
    info_X.append(info["X"])
    info_Y.append(info["Y"])
    info_Z.append(info["Z"])
    episode_reward.append(reward) # save the reward for all the episode

    time=time + env.timeStep # elapsed time since simulation start
    info_time.append(time)

    #env.render()
    if done:
      # obs = env.reset()
      break

## PLOT AND DISPLAY SECTION

plt.figure(1)
plt.plot(info_time, info_p)
plt.plot(info_time, info_q)
plt.plot(info_time, info_r)
plt.xlabel('time')
plt.ylabel('Angular velocity [rad/s]')
plt.title('p,q and r')
plt.legend(['p', 'q', 'r'])
plt.savefig('SimulationResults/Angular_velocity.jpg')

plt.figure(2)
plt.plot(info_time, info_u)
plt.plot(info_time, info_v)
plt.plot(info_time, info_w)
plt.xlabel('time')
plt.ylabel('Velocity [m/s]')
plt.title('u,v and w')
plt.legend(['u', 'v', 'w'])
plt.savefig('SimulationResults/Velocity.jpg')

plt.figure(3)
plt.plot(info_time, info_X)
plt.plot(info_time, info_Y)
plt.plot(info_time, info_Z)
plt.xlabel('time')
plt.ylabel('Position NED [m]')
plt.title('X,Y and Z')
plt.legend(['X', 'Y', 'Z'])
plt.savefig('SimulationResults/Position.jpg')

## CONVERSION OF THE QUATERNION INTO EULER ANGLES
Euler_angles = np.zeros([np.size(info_quaternion, 0), 3])

for row in range(np.size(Euler_angles, 0)):
  q0 = info_quaternion[row, 0]
  q1 = info_quaternion[row, 1]
  q2 = info_quaternion[row, 2]
  q3 = info_quaternion[row, 3]

  Euler_angles[row, 0] = np.arctan2(2*(q0*q1 + q2*q3), 1-2*(q1**2+q2**2))
  Euler_angles[row, 1] = np.arcsin(2*(q0*q2-q3*q1))
  Euler_angles[row, 2] = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

#Conversion to degrees from radians
Euler_angles = Euler_angles * (180 / np.pi)

plt.figure(4)
plt.plot(info_time, Euler_angles[:, 0])
plt.plot(info_time, Euler_angles[:, 1])
plt.plot(info_time, Euler_angles[:, 2])
plt.xlabel('time')
plt.ylabel('Angles [deg]')
plt.title('Euler Angles')
plt.legend(['Phi', 'Theta', 'Psi'])
plt.savefig('SimulationResults/Euler.jpg')

plt.figure(5)
plt.plot(info_time, episode_reward)
plt.xlabel('time')
plt.ylabel('Reward')
plt.title('Episode Reward')
plt.savefig('SimulationResults/reward.jpg')