# Code to simulate the environment when trained
import os
#ignore tensorflow warnings
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from Humming_env import Hummingbird_6DOF


env = Hummingbird_6DOF(Random_reset=False, Process_perturbations=True)

tieme_steps_to_simulate = env.max_Episode_time_steps + 1 ## define the number of timesteps to simulate

######################################
##      POLICY LOADING SECTION      ##
######################################

# use of os.path.exists() to check and load the last policy evaluated by training
# function.
print("Policy loading...")

Policy_loading_mode = input("Insert loading mode\nlast: loads last policy saved\nbest: loads best policy saved\nsel: loads a specified policy\n-----> ")

if Policy_loading_mode == "last":
  for i in range(100, 0, -1): ## function look for the last policy evaluated.
    fileName_toFind = "/home/ghost/giorgio_diliberi/ReinforcementLearning/Hummingbird/Hummingbird_env_Vref_Psiref/Policies/PPO_Quad_" + str(i) + ".zip"

    if os.path.exists(fileName_toFind):
      print("last policy found is PPO_Quad_", i)

      Policy2Load = "Policies/PPO_Quad_" + str(i)
      
      break

  

elif Policy_loading_mode == "best":

  Policy2Load = "EvalClbkLogs/best_model.zip" # best policy name

elif Policy_loading_mode == "lastbest":

  Policy2Load = "EvalClbkLogs/best_model9.zip"

else:

  Policy2Load  = input("enter the relative path of policy to load (check before if exists): ")
  
  

model = PPO2.load(Policy2Load)
print("Policy ", Policy2Load, " loaded!")

######################################
##     SIMULATION SECTION           ##
######################################
 
#model = PPO2.load("Policies/PPO_Quad_1")  # uncomment this line to load a specific policy instead of the last one

obs = env.reset()

# info vectors initialization for simulation history
info_u = [env.state[0]]
info_v = [env.state[1]]
info_w = [env.state[2]]
info_p = [env.state[3]]
info_q = [env.state[4]]
info_r = [env.state[5]]
info_quaternion = np.array([env.state[6:10]]) # quaternion stored in a np.array matrix
info_X = [env.state[10]]
info_Y = [env.state[11]]
info_Z = [env.state[12]]
info_V_N = [0.]
info_V_E = [0.]
info_V_D = [0.]
action_memory = np.array([0., 0., 0., 0.]) ## vector to store actions during the simulation
#Throttle_memory = [env.dTt]
episode_reward = [env.getReward()]

X_ref = [env.X_ref]
Y_ref = [env.Y_ref]
Z_ref = [env.Z_ref]
VN_ref = [env.V_NED_ref[0]]
VE_ref = [env.V_NED_ref[1]]
VD_ref = [env.V_NED_ref[2]]

time=0.
info_time=[time] # elapsed time vector

env.Position_reference = True

# SIMULATION

for i in range(tieme_steps_to_simulate): #last number is excluded

    # # Waypoint navigation section (uncomment to realize wp nav)
    if i==350:
      env.X_ref = 0.
      env.Y_ref = 0.
      env.Z_ref = -20.

    if i==700:
      env.X_ref = 15.
      env.Y_ref = 0.
      env.Z_ref = -20.

    if i==1150:
      env.X_ref = 15.
      env.Y_ref = 15.
      env.Z_ref = -20.
      #env.NewWP = True
      #env.psi_ref_mem = 0.  #90.*0.0175

    if i==1500:
      env.X_ref = 0.
      env.Y_ref = 15.
      env.Z_ref = -20.
      #.NewWP = True
      #env.psi_ref_mem = 0.  #175.*0.0175

    if i==1850:
      env.X_ref = 0.
      env.Y_ref = 0.
      env.Z_ref = -20.
      #env.NewWP = True
      #env.psi_ref_mem = -135. * 0.0175   #-90.*0.0175

    if i==2512:
      env.X_ref = 0.
      env.Y_ref = 0.
      env.Z_ref = -5.
    #   #env.NewWP = True
    #   #env.psi_ref_mem = -90. * 0.0175

    # if i==2756:
    #   env.X_ref = 0.
    #   env.Y_ref = 0.
    #   env.Z_ref = -2.

    # # Vectorial navigation--> spiral movement each step references are updated with sin, cos and linear z
    # env.VNord_ref = 2 * np.cos(0.5 * env.elapsed_time_steps * 0.04)
    # env.VEst_ref = 2 * np.sin(0.5 * env.elapsed_time_steps * 0.04)
    # env.VDown_ref = - 1.4 * 10. / 40

    # moving waypoint

    # if i==32:
    #   env.Z_ref = -17.

    # if i>=256 and i<1750:
    #   if i%32==0:
    #     env.X_ref = 7.5 * np.sin(0.25 * (env.elapsed_time_steps-256) * 0.04)
    #     env.Y_ref = 9.2 * env.elapsed_time_steps * 0.04 / 40

    # if i==1750:
    #   env.X_ref = 0.0
    #   env.Y_ref = 0.0

    # if i==2125:
    #   env.Z_ref = -2.0

    
    action, _state = model.predict(obs, deterministic=True) # Add deterministic true for PPO to achieve better performane
    
    obs, reward, done, info = env.step(action) 

    # state matrices
    info_u.append(info["u"])
    info_v.append(info["v"])
    info_w.append(info["w"])
    info_p.append(info["p"])
    info_q.append(info["q"])
    info_r.append(info["r"])
    info_quaternion = np.vstack([info_quaternion, [info["q0"], info["q1"], info["q2"], info["q3"]]])
    info_X.append(info["X"])
    info_Y.append(info["Y"])
    info_Z.append(info["Z"])
    info_V_N.append(info["V_Nord"])
    info_V_E.append(info["V_Est"])
    info_V_D.append(info["V_Down"])
    action_memory = np.vstack([action_memory, action])
    #Throttle_memory.append(env.linearAct2ThrMap(action[0]))
    episode_reward.append(reward) # save the reward for all the episode

    # references matrices
    X_ref.append(env.X_ref)
    Y_ref.append(env.Y_ref)
    Z_ref.append(env.Z_ref)
    VN_ref.append(env.V_NED_ref[0])
    VE_ref.append(env.V_NED_ref[1])
    VD_ref.append(env.V_NED_ref[2])

    time=time + env.timeStep # elapsed time since simulation start
    info_time.append(time)

    #env.render()
    if done:
      # obs = env.reset()
      break

    if i==450:
      print("Mid V_NED [N, E, D]= ", [info["V_Nord"], info["V_Est"], info["V_Down"]])

print("final V_NED [N, E, D]= ", [info["V_Nord"], info["V_Est"], info["V_Down"]])

## PLOT AND DISPLAY SECTION

plt.figure(1)
plt.plot(info_time, info_p)
plt.plot(info_time, info_q)
plt.plot(info_time, info_r)
plt.xlabel('time')
plt.ylabel('Angular velocity [rad/s]')
plt.title('p,q and r')
plt.legend(['p', 'q', 'r'])
plt.savefig('SimulationResults/Angular_velocity.jpg')

plt.figure(2)
plt.plot(info_time, info_u)
plt.plot(info_time, info_v)
plt.plot(info_time, info_w)
plt.xlabel('time')
plt.ylabel('Velocity [m/s]')
plt.title('u,v and w')
plt.legend(['u', 'v', 'w'])
plt.savefig('SimulationResults/Velocity.jpg')

plt.figure(3)
plt.plot(info_time, info_X)
plt.plot(info_time, info_Y)
plt.plot(info_time, info_Z)
plt.xlabel('time')
plt.ylabel('Position NED [m]')
plt.title('X,Y and Z')
plt.legend(['X', 'Y', 'Z'])
plt.savefig('SimulationResults/Position.jpg')

## CONVERSION OF THE QUATERNION INTO EULER ANGLES
Euler_angles_rad = np.zeros([np.size(info_quaternion, 0), 3]) 

for row in range(np.size(Euler_angles_rad, 0)):
  q0 = info_quaternion[row, 0]
  q1 = info_quaternion[row, 1]
  q2 = info_quaternion[row, 2]
  q3 = info_quaternion[row, 3]

  Euler_angles_rad[row, 0] = np.arctan2(2*(q0*q1 + q2*q3), 1-2*(q1**2+q2**2))
  Euler_angles_rad[row, 1] = np.arcsin(2*(q0*q2-q3*q1))
  Euler_angles_rad[row, 2] = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

#Conversion to degrees from radians
Euler_angles = Euler_angles_rad * (180 / np.pi)

plt.figure(4)
plt.plot(info_time, Euler_angles[:, 0])
plt.plot(info_time, Euler_angles[:, 1])
plt.plot(info_time, Euler_angles[:, 2])
plt.xlabel('time')
plt.ylabel('Angles [deg]')
plt.title('Euler Angles')
plt.legend(['Phi', 'Theta', 'Psi'])
plt.savefig('SimulationResults/Euler.jpg')

plt.figure(5)
plt.plot(info_time, episode_reward)
plt.xlabel('time')
plt.ylabel('Reward')
plt.title('Episode Reward')
plt.savefig('SimulationResults/reward.jpg')

plt.figure(6)
plt.plot(info_time, action_memory[:, 0])
plt.plot(info_time, action_memory[:, 1])
plt.plot(info_time, action_memory[:, 2])
plt.plot(info_time, action_memory[:, 3])
plt.xlabel('time')
plt.ylabel('Actions')
plt.title('Actions in episode [-1, 1]')
plt.legend(['Avg_thr', 'Ail', 'Ele', 'Rud'])
plt.savefig('SimulationResults/action.jpg')

plt.figure(7)
plt.plot(info_time, info_V_N)
plt.plot(info_time, info_V_E)
plt.plot(info_time, info_V_D)
plt.xlabel('time')
plt.ylabel('V_NED')
plt.title('Earth frame velocity')
plt.legend(['V_Nord', 'V_Est', 'V_Down'])
plt.savefig('SimulationResults/V_NED.jpg')

info_H = -1 * np.array([info_Z])

#ax.xlabel('X')
#ax.ylabel('Y')
#ax.ylabel('H==-Z')
#ax.title('Trajectory')

for count in range(int(env.elapsed_time_steps/8)):

  figCount = 8+count

  fig = plt.figure(figCount)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_wireframe(np.array([info_X]), np.array([info_Y]), np.array([info_Z]))
  ax.invert_xaxis()
  ax.invert_zaxis()

  step_n = count * 8

  Phi, Theta, Psi = Euler_angles_rad[step_n, :]

  u_Xb = np.cos(Theta) * np.cos(Psi)
  v_Xb = np.cos(Theta) * np.sin(Psi)
  w_Xb = -np.sin(Theta)

  u_Yb = -np.cos(Phi) * np.sin(Psi) + np.sin(Phi) * np.sin(Theta) * np.cos(Psi)
  v_Yb = np.cos(Phi) * np.cos(Psi) + np.sin(Phi) * np.sin(Theta) * np.sin(Psi)
  w_Yb = np.sin(Phi) * np.cos(Theta) 

  u_Zb = np.sin(Phi) * np.sin(Psi) + np.sin(Phi) * np.sin(Theta) * np.cos(Psi)
  v_Zb = -np.sin(Phi) * np.cos(Psi) + np.cos(Phi) * np.sin(Theta) * np.sin(Psi)
  w_Zb = np.cos(Phi) * np.cos(Theta)

  x = info_X[step_n]
  y = info_Y[step_n]
  z = info_Z[step_n]

  ax.quiver(x, y, z, u_Xb, v_Xb, w_Xb, length=5., normalize=False, color="red") # X_b
  ax.quiver(x, y, z, u_Yb, v_Yb, w_Yb, length=5., normalize=False, color="blue") #Y_b
  ax.quiver(x, y, z, u_Zb, v_Zb, w_Zb, length=5., normalize=False, color="green") #Z_b

  #ax.set_xlim3d(7.5, -7.5)

  ax.set_xlabel("North")
  ax.set_ylabel("East")
  ax.set_zlabel("Down")

  ax.scatter(0, 0, -5, c="black", s=1.)
  ax.scatter(15, 0, 0, c="black", s=1.)
  ax.scatter(0, 15, 0, c="black", s=1.)
  ax.scatter(0, 0, -20, c="black", s=1.)

  #plot the waypoint
  ax.scatter(X_ref[step_n], Y_ref[step_n], Z_ref[step_n], c="red", s=100.)

  fig2save = 'SimulationResults/Orientation/trajectory' + str(count) + '.jpg'

  plt.savefig(fig2save)
  n_fig = count


## Sction to plot orientation with arrows to represent the miniquad figure
# so for arrow to represent arms and a dot to represent body

for count in range(int(env.elapsed_time_steps/8)):

  figCount = 9 + n_fig + count

  fig = plt.figure(figCount)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_wireframe(np.array([info_X]), np.array([info_Y]), np.array([info_Z]))
  ax.invert_xaxis()
  ax.invert_zaxis()

  step_n = count * 8

  Phi, Theta, Psi = Euler_angles_rad[step_n, :]

  ## Components of arm 1
  u_M1 = np.cos(Theta) * np.cos(Psi)
  v_M1 = np.cos(Theta) * np.sin(Psi)
  w_M1 = -np.sin(Theta)

  ## Components of arm 2
  u_M2 = -np.cos(Phi) * np.sin(Psi) + np.sin(Phi) * np.sin(Theta) * np.cos(Psi)
  v_M2 = np.cos(Phi) * np.cos(Psi) + np.sin(Phi) * np.sin(Theta) * np.sin(Psi)
  w_M2 = np.sin(Phi) * np.cos(Theta) 

  ## Components of arm 3 = -1*arm1
  u_M3 = -u_M1
  v_M3 = -v_M1
  w_M3 = -w_M1

  ## Components of arm 4 = -1*arm2
  u_M4 = -u_M2
  v_M4 = -v_M2
  w_M4 = -w_M2

  x = info_X[step_n]
  y = info_Y[step_n]
  z = info_Z[step_n]

  ax.quiver(x, y, z, u_M1, v_M1, w_M1, length=5., normalize=False, color="green") # r_M1
  ax.quiver(x, y, z, u_M2, v_M2, w_M2, length=5., normalize=False, color="blue") # r_M1
  ax.quiver(x, y, z, u_M3, v_M3, w_M3, length=5., normalize=False, color="red") # r_M1
  ax.quiver(x, y, z, u_M4, v_M4, w_M4, length=5., normalize=False, color="blue") # r_M1
  ax.scatter(x, y, z, c="blue", s=65.) # dot for the body

  #ax.set_xlim3d(7.5, -7.5)

  ax.set_xlabel("North")
  ax.set_ylabel("East")
  ax.set_zlabel("Down")

  #plot the waypoint
  ax.scatter(X_ref[step_n], Y_ref[step_n], Z_ref[step_n], c="red", s=100.)

  fig2save = 'SimulationResults/Orient_quad/trajectory' + str(count) + '.jpg'

  plt.savefig(fig2save)

simout_array = np.stack([info_u, info_v, info_w, info_p, info_q, info_r, Euler_angles[:, 0], Euler_angles[:, 1], Euler_angles[:, 2], info_X, info_Y, info_Z, info_V_N, info_V_E, info_V_D], axis=1)

np.savetxt("simout.txt", simout_array)

ref_array = np.stack([X_ref, Y_ref, Z_ref, VN_ref, VE_ref, VD_ref], axis=1)

np.savetxt("references.txt", ref_array)

np.savetxt("time.txt", np.array(info_time))

