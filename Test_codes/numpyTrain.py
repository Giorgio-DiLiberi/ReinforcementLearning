import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6])

y = np.concatenate((x[0:3], [1, 2, 3]))

print (y)

Obs_normalization_vector = np.array([30., 30., 30., 50., 50., 50., 1., 1., 1., 1., 50., 50., 50.]) 

obs = Obs_normalization_vector

obs = np.stack([obs, np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13)], axis=0)

print(obs.shape)

print(np.arctan2(0.,-0.))