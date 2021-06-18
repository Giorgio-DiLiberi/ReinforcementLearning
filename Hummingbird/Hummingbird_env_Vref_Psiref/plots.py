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

references = np.loadtxt("references.txt")
X_ref = references[:, 0]
Y_ref = references[:, 1]
Z_ref = references[:, 2]

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

