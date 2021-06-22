import numpy as np
from numpy.random import normal as np_normal

mass = 0.71
I=np.array([0.0037, 0.0037, 0.0099])

np.random.seed(10)

time=[0]

F = mass * np_normal(0, 0.005, 3)

M = I * np_normal(0, 5*0.00175, 3)

for i in range(256):

    time.append(0.04 * i)

    F = np.vstack([F, mass * np_normal(0, 0.005, 3)])
    M = np.vstack([M, I * np_normal(0, 40*0.00175, 3)])

np.savetxt("F.txt", F)
np.savetxt("M.txt", M)
