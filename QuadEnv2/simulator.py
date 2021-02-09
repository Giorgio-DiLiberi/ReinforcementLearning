# Code to simulate the environment when trained
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from quadcoptV2 import QuadcoptEnvV2

env = QuadcoptEnvV2()

# model = PPO.load("ppo_Quad_1Attempt")

obs = env.reset()

# info vectors initialization for simulation history
info_u=[env.state[0]]
info_v=[env.state[1]]
info_w=[env.state[2]]
info_p=[env.state[3]]
info_q=[env.state[4]]
info_r=[env.state[5]]
info_Z=[env.state[12]]

time=0.
info_time=[time] # elased time vector

# SIMULATION

for i in range(1000):

    # uncomment the correct statement to test trim or a policy
    
    # action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    action = np.array([0, 0, 0, 0]) # Trim thrust test now actions are varations on trim value yet implemented in the environment

    obs, reward, done, info = env.step(action) 

    info_u.append(info["u"])
    info_v.append(info["v"])
    info_w.append(info["w"])
    info_p.append(info["p"])
    info_q.append(info["q"])
    info_r.append(info["r"])
    info_Z.append(info["Z"])

    time=time+0.1 # elapsed time since simulation start
    info_time.append(time)

    #env.render()
    if done:
      # obs = env.reset()
      break

## PLOT AND DISPLAY SECTION
plt.plot(info_time, info_Z)
plt.plot(info_time, info_w)
plt.xlabel('time')
plt.ylabel('data')
plt.title('w and Z')
plt.legend(['Z', 'w'])
plt.show()