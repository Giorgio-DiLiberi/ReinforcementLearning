# Code to test the environment constants and methods

import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos as cos
from numpy import sin as sin
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt

from Humming_env import Hummingbird_6DOF

env = Hummingbird_6DOF()

print("Kf= ", env.Prop_Kf)
print("Kq= ", env.Prop_Kq)
print("Max_thrustper motor= ", (env.Prop_Kf*(env.nMax_motor**2)))
print("vH = ", env.vh)

print("Trim_thr= ", env.dTt)

RPS = np.sqrt(env.dTt)*env.nMax_motor
print (RPS)


wm_arr = np.linspace(-2., 10., 120)

vi_arr = []
Delta_T_arr = []

for i in range(120):

    wm = -wm_arr[i]
    vi = wm/2 + np.sqrt(((0.5*wm)**2) + (env.vh**2))
    Delta_Thrust = env.rho * np.pi * (2*np.pi*RPS) * env.prop_mean_chord * ((env.D_prop**2)/4) * (- vi + wm + env.vh)
    
    vi_arr.append(vi)
    Delta_T_arr.append(Delta_Thrust)

plt.figure(1)
plt.plot(wm_arr, vi_arr)
plt.xlabel('Vc [m/s]')
plt.ylabel('induced vel. [m/s]')
plt.title('Induced velocity')
plt.savefig('SimulationResults/Induced_vel.jpg')

plt.figure(2)
plt.plot(wm_arr, Delta_T_arr)
plt.xlabel('Vc [m/s]')
plt.ylabel('Delta Thrust N]')
plt.title('Thrust variation')
plt.savefig('SimulationResults/Delta_Thrust.jpg')


