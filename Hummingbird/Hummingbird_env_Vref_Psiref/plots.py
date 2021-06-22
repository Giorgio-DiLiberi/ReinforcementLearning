# Code to plot some useful graphs from simulazion data

import numpy as np
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


state = np.loadtxt("simout.txt")
X = state[:, 9]
Y = state[:, 10]
Z = state[:, 11]
VN = state[:, 12]
VE = state[:, 13]
VD = state[:, 14]

references = np.loadtxt("references.txt")
X_ref = references[:, 0]
Y_ref = references[:, 1]
Z_ref = references[:, 2]
VN_ref = references[:, 3]
VE_ref = references[:, 4]
VD_ref = references[:, 5]


time = np.loadtxt("time.txt")

distance = []

for i in range(3073):
    distance.append(np.sqrt((X[i] - X_ref[i])**2 + (Y[i] - Y_ref[i])**2 + (Z[i] - Z_ref[i])**2))

plt.figure(1)
plt.plot(time, distance)
plt.scatter(0.04 * 0.00, 0, c="red", s=5.)
plt.scatter(0.04 * 750, 0, c="red", s=5.)
plt.scatter(0.04 * 1200, 0, c="red", s=5.)
plt.scatter(0.04 * 1600, 0, c="red", s=5.)
plt.scatter(0.04 * 2000, 0, c="red", s=5.)
plt.scatter(0.04 * 2512, 0, c="red", s=5.)
plt.xlabel('time [s]')
plt.ylabel('Distance [m]')
plt.title('Distance in sim')
plt.savefig('SimulationResults/Dist1sim.jpg')

plt.figure(2)
plt.plot(time, X)
plt.plot(time, X_ref)
plt.xlabel('time [s]')
plt.ylabel('Position [m]')
plt.title('X - X reference')
plt.legend(["X", "X_ref"])
plt.savefig('SimulationResults/X_Xref.jpg')

plt.figure(3)
plt.plot(time, Y)
plt.plot(time, Y_ref)
plt.xlabel('time [s]')
plt.ylabel('Position [m]')
plt.title('Y - Y reference')
plt.legend(["Y", "Y_ref"])
plt.savefig('SimulationResults/Y_Yref.jpg')

plt.figure(4)
plt.plot(time, Z)
plt.plot(time, Z_ref)
plt.xlabel('time [s]')
plt.ylabel('Position [m]')
plt.title('Z - Z reference')
plt.legend(["Z", "Z_ref"])
plt.savefig('SimulationResults/Z_Zref.jpg')

plt.figure(5)
plt.plot(time, VN)
plt.plot(time, VN_ref)
plt.xlabel('time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('VN - VN reference')
plt.legend(["VN", "VN_ref"])
plt.savefig('SimulationResults/VN_VNref.jpg')

plt.figure(6)
plt.plot(time, VE)
plt.plot(time, VE_ref)
plt.xlabel('time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('VE - VE reference')
plt.legend(["VE", "VE_ref"])
plt.savefig('SimulationResults/VE_VEref.jpg')

plt.figure(7)
plt.plot(time, VD)
plt.plot(time, VD_ref)
plt.xlabel('time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('VD - VD reference')
plt.legend(["VD", "VD_ref"])
plt.savefig('SimulationResults/VD_VDref.jpg')

## PLOT of trajectory and waypoints
fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
ax.invert_xaxis()
ax.invert_zaxis()

ax.scatter(0., 0., -5., c="red", s=80.)
ax.scatter(0., 0., -20., c="red", s=80.)
ax.scatter(15., 0., -20., c="red", s=80.)
ax.scatter(15., 15., -20., c="red", s=80.)
ax.scatter(0., 15., -20., c="red", s=80.)

# plot arrows to show waypoints sequence
ax.quiver(0, 0, -5, 0, 0, -1, length=7.5, normalize=False, color="green") # first waypoint
ax.quiver(0, 0, -12.5, 0, 0, -1, length=7.5, normalize=False, color="green") # first waypoint

ax.quiver(0, 0, -20, 1, 0, 0, length=7.5, normalize=False, color="green") # first waypoint
ax.quiver(7.5, 0, -20, 1, 0, 0, length=7.5, normalize=False, color="green") # first waypoint

ax.quiver(15, 0, -20, 0, 1, 0, length=7.5, normalize=False, color="green") # first waypoint
ax.quiver(15, 7.5, -20, 0, 1, 0, length=7.5, normalize=False, color="green") # first waypoint

ax.quiver(15, 15, -20, -1, 0, 0, length=7.5, normalize=False, color="green") # first waypoint
ax.quiver(7.5, 15, -20, -1, 0, 0, length=7.5, normalize=False, color="green") # first waypoint

ax.quiver(0, 15, -20, 0, -1, 0, length=7.5, normalize=False, color="green") # first waypoint
ax.quiver(0, 7.5, -20, 0, -1, 0, length=7.5, normalize=False, color="green") # first waypoint

ax.quiver(0, 0.2, -20, 0, 0, 1, length=7.5, normalize=False, color="blue") # first waypoint
ax.quiver(0, 0.2, -12.5, 0, 0, 1, length=7.5, normalize=False, color="blue") # first waypoint


ax.set_xlabel("North")
ax.set_ylabel("East")
ax.set_zlabel("Down")
plt.savefig("SimulationResults/Waypoints.jpg")


