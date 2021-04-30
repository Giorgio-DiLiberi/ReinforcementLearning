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

# file2Open = "buildParamsFiles/D_8_in.txt"

# with open(file2Open, "r") as input_file: # with open context
#     input_file_all = input_file.readlines() # crate an array of strings containing all the file lines
#     for line in input_file_all: # read line
#         line = line.split() # splits lines into 2 strings and set left = right, this suggest how to format the file
#         globals()[line[0]] = line[1]

# print("Lx = ", Lx)
# print("Ly = ", Ly) 
# print("motor_M = ", motor_mass)
# print(float(body_mass))
# print(batt_payload_mass)
# print(Motor_KV)
# print(Batt_V_nom)
# print(prop_D)
# print(Prop_Ct)
# print(Prop_Cp)
# env.nMax_motor = env.Motor_Kv * env.V_batt_nom / 60 #[RPS]
# env.Prop_Kf = env.Ct * env.rho * (env.D_prop**4) #[kg m]==[N/RPS^2]
# env.Prop_Kq = env.Cp * env.rho * (env.D_prop**5)/(2*np.pi) #[kg m^2]
# env.dTt = (env.mass * env.g0 / (4*env.Prop_Kf)) / (env.nMax_motor**2) 
# print("Trim_thr= ", env.dTt)

print(env.Lx_a)
print(env.Ly_a)
print(env.motor_mass_a)
print(env.body_mass_a)
print(env.batt_payload_mass_a)
print(env.Max_prop_RPM_a)
print(env.prop_D_a)
print(env.Prop_Ct_a)
print(env.Prop_Cp_a)

env.reset()
env.reset()
env.reset()
print(env.Lx)
print(env.Ly)
print(env.motor_mass)
print(env.body_mass)
print(env.battery_mass)
print(env.nMax_motor)
print(env.D_prop)
print(env.Ct)
print(env.Cp)

