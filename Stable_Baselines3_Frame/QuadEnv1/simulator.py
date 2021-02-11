# Code to simulate the environment when trained
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from quadcoptV1 import QuadcoptEnvV1

env = QuadcoptEnvV1()

model = PPO.load("ppo_Quad_1Attempt")

obs = env.reset()
info_u=[0]
info_w=[0]
info_Z=[0]
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    
    obs, reward, done, info = env.step(action) 
    info_u.append(info["u"])
    info_w.append(info["w"])
    info_Z.append(info["Z"])

    #env.render()
    if done:
      # obs = env.reset()
      break

#print(info_u)
print(info_w)
print(info_w.__len__())