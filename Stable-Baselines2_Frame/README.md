# ReinforcementLearning
This directory contains the files and environments created to write my master degree thesis: the project involves 
controlling a quadcopter drone using a Neural network. 
The projects is developed using Stable-Baselines framework and gym.

"QuadEnv2" contains a second assumption of the model with all written in vector form.

Those codes are written for version 2 of stable baselines using tensorflow 1.15 so they need to be run on the conda virtual environment "python37_Env"

Directory "QuadEnv2" contains the simple environment with the equations and the code to execute a simulation with given actions, also it can be used to load a trained policy and test it on the environment

Directory "QuadEnv3" contains the environment modified with equations in vector form.
Those previous environments with all code to simulate, train a policy or test simple actions, are with simple initial condition of leveled drone stopped at 50 meters of height.

Directory "QuadEnv4" contains a modified version of the environment which implements a more accurate model of the motors ut is the same as quadcopt v3 for other instances

Dir "Quadenv5" contains a model which implements a PID on p, q and r. the only change is that now ations are average overall thr and reference for p,q,r while the throttle for each motor is computed by a PID function




