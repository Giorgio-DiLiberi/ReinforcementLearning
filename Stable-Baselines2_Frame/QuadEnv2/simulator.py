# Code to simulate the environment when trained
import gym
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from quadcoptV2 import QuadcoptEnvV2

env = QuadcoptEnvV2()

model = PPO2.load("ppo_Quad_1Attempt")

obs = env.reset()

# info vectors initialization for simulation history
info_u=[env.state[0]]
info_v=[env.state[1]]
info_w=[env.state[2]]
info_p=[env.state[3]]
info_q=[env.state[4]]
info_r=[env.state[5]]
info_X=[env.state[10]]
info_Y=[env.state[11]]
info_Z=[env.state[12]]

time=0.
info_time=[time] # elased time vector

# SIMULATION

for i in range(1000):

    # uncomment the correct statement to test trim or a policy
    
    action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    
    obs, reward, done, info = env.step(action) 

    info_u.append(info["u"])
    info_v.append(info["v"])
    info_w.append(info["w"])
    info_p.append(info["p"])
    info_q.append(info["q"])
    info_r.append(info["r"])
    info_X.append(info["X"])
    info_Y.append(info["Y"])
    info_Z.append(info["Z"])

    time=time+0.1 # elapsed time since simulation start
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
plt.ylabel('data')
plt.title('p,q and r')
plt.legend(['p', 'q', 'r'])

plt.figure(2)
plt.plot(info_time, info_u)
plt.plot(info_time, info_v)
plt.plot(info_time, info_w)
plt.xlabel('time')
plt.ylabel('data')
plt.title('u,v and w')
plt.legend(['u', 'v', 'w'])

plt.figure(3)
plt.plot(info_time, info_X)
plt.plot(info_time, info_Y)
plt.plot(info_time, info_Z)
plt.xlabel('time')
plt.ylabel('data')
plt.title('X,Y and Z')
plt.legend(['X', 'Y', 'Z'])

plt.show()