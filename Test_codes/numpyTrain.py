import numpy as np
import matplotlib
matplotlib.use('pdf') # To avoid plt.show issues in virtualenv
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5, 6])

y = np.concatenate((x[0:3], [1, 2, 3]))

print (y)

Obs_normalization_vector = np.array([30., 30., 30., 50., 50., 50., 1., 1., 1., 1., 50., 50., 50.]) 

obs = Obs_normalization_vector

obs = np.stack([obs, np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13)], axis=0)

print(obs.shape)

print(np.arctan2(0.,-0.))




def arguments():
    arg1 = np.array([0, 1, 2])
    arg2 = np.array([3, 4, 5])

    return arg1, arg2

primo, secondo = arguments()

print(primo)
print(secondo)

LEB = np.array([[1, 2] , [3, 4]])
print(LEB[1])

WP_list = np.array([[0, 0, -20], [15, 0, -20], [15, 15, -20], [0, 15, -20], [0, 0, -20], [0, 0, -5]])

print(WP_list[1, :])

print(WP_list.size)
