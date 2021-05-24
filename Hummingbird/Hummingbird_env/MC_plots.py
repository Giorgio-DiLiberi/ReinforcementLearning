## code to plot results from saved arrays and matrices
import numpy as np
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

info_time = np.loadtxt("time.txt")

SS_dist = np.loadtxt("SS_distance_array.txt")

distance_mem_matrix = np.loadtxt("Distance_mem_matrix.txt")

X_mem = np.loadtxt("X_mem.txt")
print("X shape: ", X_mem.shape)

Y_mem = np.loadtxt("Y.mem.txt")

Z_mem = np.loadtxt("Z.mem.txt")




  
fig = plt.figure(1)
# to print all on the same graph
ax = fig.add_subplot(111, projection='3d')

for count2 in range(25):

    X = X_mem[int(count2+1), :]
    Y = Y_mem[int(count2+1), :]
    Z = Z_mem[int(count2+1), :]
    ax.plot(X, Y, Z)

    #plot the waypoint
    X_ini = X_mem[int(count2+1), 0]
    Y_ini = Y_mem[int(count2+1), 0]
    Z_ini = -2.
    ax.scatter(X_ini, Y_ini, Z_ini, c="blue", s=20.)

ax.invert_xaxis()
ax.invert_zaxis()

ax.set_xlabel("North")
ax.set_ylabel("East")
ax.set_zlabel("Down")

ax.scatter(0., 0., -10., c="red", s=60.)

plt.savefig("SimulationResults/Trajectories.jpg")