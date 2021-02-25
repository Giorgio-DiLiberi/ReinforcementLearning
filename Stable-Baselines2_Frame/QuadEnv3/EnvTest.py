# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt

from quadcoptV3 import QuadcoptEnvV3

env = QuadcoptEnvV3()

print("Trim_thr= ", env.dTt)

dT1, dT2, dT3, dT4 = env.act2ThrotMap([-1., -1., -1., -1.])

print("action[-1]= ", dT1, dT2, dT3, dT4)

dT1, dT2, dT3, dT4 = env.act2ThrotMap([0., 0., 0., 0.])

print("action[0]= ", dT1, dT2, dT3, dT4)

dT1, dT2, dT3, dT4 = env.act2ThrotMap([1., 1., 1., 1.])

print("action[0.1]= ", dT1, dT2, dT3, dT4)


env.reset()
print("test output for q0= ", env.state[6])
print("test output for POS= ", env.state[10:13])