# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt

from quadcopt_6DOF import QuadcoptEnv_6DOF

env = QuadcoptEnv_6DOF()

print("Kf= ", env.Prop_Kf)
print("Kq= ", env.Prop_Kq)
print("Max_thrust= ", (env.Prop_Kf*(env.nMax_motor**2)))

print("Trim_thr= ", env.dTt)



obs = env.reset()
print(obs)