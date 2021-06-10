# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos as cos
from numpy import sin as sin

from Humming_env import Hummingbird_6DOF

env = Hummingbird_6DOF()

print("Kf= ", env.Prop_Kf)
print("Kq= ", env.Prop_Kq)
print("Max_thrustper motor= ", (env.Prop_Kf*(env.nMax_motor**2)))
print("vH = ", env.vh)

print("Trim_thr= ", env.dTt)

RPS = 4500 / 60
wm = -10.
vi = wm/2 + np.sqrt(((0.5*wm)**2) + (env.vh**2))

Delta_Thrust = env.rho * np.pi * (2*np.pi*RPS) * env.prop_mean_chord * ((env.D_prop**2)/4) * (- vi + wm + env.vh)
      
print("Delta Thrust = ", Delta_Thrust)

dT1 = 0.38
Vb = np.array([0., 0., 0.])
Omega = np.array([0., 0., 1.75])
Vm1 = Vb + np.cross(Omega, env.rM1)
M1_Thrust, M1_Torque, M1_a1, M1_b1 = env.Motor(dT1, Vm1) # scalar values for M1
F1 = np.array([-M1_Thrust * sin(M1_a1), -M1_Thrust * sin(M1_b1), -M1_Thrust])

print("Motor Force = ", F1)


obs = env.reset()
print(obs)