import gym

from stable_baselines3 import A2C
from customenv1 import CustomEnv

env = CustomEnv()

model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=10000)
#model.save('A2C_cart')
model.load('A2C_cart.zip')

obs = env.reset()
info1=[0]
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action) #info contiene la storia delle osservazioni
    info1.append(info)

    #env.render()
    if done:
      obs = env.reset()

print(info1)