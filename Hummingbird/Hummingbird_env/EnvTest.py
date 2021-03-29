# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt

from Humming_env import Hummingbird_6DOF

env = Hummingbird_6DOF()

print("Kf= ", env.Prop_Kf)
print("Kq= ", env.Prop_Kq)
print("Max_thrustper motor= ", (env.Prop_Kf*(env.nMax_motor**2)))

print("Trim_thr= ", env.dTt)

RPS = 4500 / 60
wm = -20.
vi = wm/2 + np.sqrt(((0.5*wm)**2) + (env.vh**2))

Delta_Thrust = env.rho * np.pi * (2*np.pi*RPS) * env.prop_mean_chord * ((env.D_prop**2)/4) * (- vi + wm + env.vh)
      
print("Delta Thrust = ", Delta_Thrust)

obs = env.reset()
print(obs)