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

xvec=np.arange(-1,1,0.01)

M1T = []
M1Q = []
M2T = []
M2Q = []
M3T = []
M3Q = []
M4T = []
M4Q = []

for x in xvec:

    dT1, dT2, dT3, dT4 = env.getThrsFromControls(np.array([x, 0., 0., 0.]))

    M1_Thrust, M1_Torque = env.Motor(dT1) # scalar values for M1
    M2_Thrust, M2_Torque = env.Motor(dT2) # scalar values for M2
    M3_Thrust, M3_Torque = env.Motor(dT3) # scalar values for M3
    M4_Thrust, M4_Torque = env.Motor(dT4) # scalar values for M4

    M1T.append(M1_Thrust)
    M1Q.append(M1_Torque)
    M2T.append(M2_Thrust)
    M2Q.append(M2_Torque)
    M3T.append(M3_Thrust)
    M3Q.append(M3_Torque)
    M4T.append(M4_Thrust)
    M4Q.append(M4_Torque)


plt.figure(1)
plt.plot(xvec, M1T)
plt.plot(xvec, M2T)
plt.plot(xvec, M3T)
plt.plot(xvec, M4T)
plt.xlabel('aileron')
plt.ylabel('motor thrust N')
plt.title('Thrust')
plt.legend(['T 1', 'T 2', 'T 3', 'T 4'])

plt.figure(2)
plt.plot(xvec, M1Q)
plt.plot(xvec, M2Q)
plt.plot(xvec, M3Q)
plt.plot(xvec, M4Q)
plt.xlabel('aileron')
plt.ylabel('motor Torque N m')
plt.title('Torque')
plt.legend(['Q 1', 'Q 2', 'Q 3', 'Q 4'])

plt.show()

print("action[0., 1, 1, 0]= ", dT1, dT2, dT3, dT4)

print("average throttle", 0.25 * (dT1+dT2+dT3+dT4))

obs = env.reset()
print(obs)