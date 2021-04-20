import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6])

y = np.concatenate((x[0:3], [1, 2, 3]))

print (y)