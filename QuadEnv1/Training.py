import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from quadcoptV1 import QuadcoptEnvV1

env = QuadcoptEnvV1()

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1)
model.save("ppo_Quad_1Attempt")

del model, env



