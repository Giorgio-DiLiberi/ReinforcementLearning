# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt

from quadcoptV5 import QuadcoptEnvV5

env = QuadcoptEnvV5()

print("Trim_thr= ", env.dTt)

dT1, dT2, dT3, dT4 = env.getThrsFromControls(np.array([0.1, 0., 1., 0.]))

print("action[0.1, 0, 0, 0]= ", dT1, dT2, dT3, dT4)

print("average throttle", 0.25 * (dT1+dT2+dT3+dT4))