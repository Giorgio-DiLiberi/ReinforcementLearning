# RL for UAV guidance and control

This Repository contains all the codes developed to train a Neural Network using Reinforcement Learning algorithms to control a Multirotor environment.

The Model of the multirotor is encoded in a gym_env format; there are some different folders containing different models of the Multirotor components.

## Required Packages 

Dirctory root/Stable_Baselines2_Frame contains codes to train and simulate RL agents on Quadcopter environment using [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) frame work. Please follow the instructions on [this page](https://stable-baselines.readthedocs.io/en/master/guide/install.html) to install all the required packages.
n particular I used python 3.7.9 and Tensorflow 1.15.

The Directory root/Stable_Baselines3_Frame contains codes to train and deploy RL agent on a Quadrotor environment using S-Bs 3 frame work, please visit [this page](https://stable-baselines3.readthedocs.io/en/master/) for all the docmentation and installation instructions.
