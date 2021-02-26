# RL for UAV guidance and control

This Repository contains all the codes developed to train a Neural Network using Reinforcement Learning algorithms to control a Multirotor environment.

The Model of the multirotor is encoded in a gym_env format; there are some different folders containing different models of the Multirotor components.

## Required Packages

Dirctory root/Stable_Baselines2_Frame contains codes to train and simulate RL agents on Quadcopter environment using [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) frame work. Please follow the instructions on [this page](https://stable-baselines.readthedocs.io/en/master/guide/install.html) to install all the required packages.
I used python 3.7.9 and Tensorflow 1.15.

The Directory root/Stable_Baselines3_Frame contains codes to train and deploy RL agent on a Quadrotor environment using S-Bs 3 frame work, please visit [this page](https://stable-baselines3.readthedocs.io/en/master/) for all the docmentation and installation instructions.
The stable Baselines 3 codes are currently not updated.

## Usage of Stable baselines 2 Frame

In root/Stable_Baselines2_Frame is possible to find some sub directory such as .../QuadEnv3, each of these contains the same codes and the difference is in the assumption in Quadcopter models.

* Directory QuadEnv3: The Quadcopter  motor thrust and torques are modeled as:

    ```python
    Thrust = self.Max_motor_Thrust * Throttle #[N]
    Torque = self.K_Q * Thrust #[N m]
    ```

    where Max_motor_Thrust and K_Q are constants determined by the props performance data. In tis Env the observations are the 13 states of the quadrotor and the actions are the 4 motors throttle values.

* Directory QuadEnv4: The motor thrust and torque are modeled as:

    ```python
    N_prop = self.N_max_motor * Throttle
    Thrust = self.Prop_Kf * (N_prop**2) #[N]
    Torque = self.Prop_Kq * (N_prop**2) #[N m]

    ```

    where N_max_motor is a constant evaluated considering the type of motor and battery used,
    Prop_kf and kq are constants based on the propeller performance data and calculated as:

    ```python
    self.Prop_Kf = self.Ct * self.rho * (self.D_prop**4) #[kg m]
    self.Prop_Kq = self.Cp * self.rho * (self.D_prop**5) / (2*np.pi) #[kg m^2]
    ```

    where Ct and Cp are thrust and power non dimensional constants, D is the prop diameter (in m if IS units are required) and rho is air density, considered constant as height is not supposed to be greater than 50 m ASL for this simulations. In tis Env the observations are the 13 states of the quadrotor and the actions are the 4 motors throttle values.

* Directory QuadEnv5:

## Usage of a QuadEnv directory

The directory can be used for:

* train a policy launching the code Training.py, the code save the best policy from EvalCallback() in the directory /EvalClbkLogs, the last policy obtained in /Policies and the tensorboard logs in /tensorboardLogs, to show the logs execute the command:

    ```bash
    tensorboard --logdir ./tensorboardLogs/
    ```

* simulate a saved policy using the code simulator.py which ask for simulating the best policy, the last policy or to select a specific policy

* test some user defined actions with the code actionTest.py

* make some debugging on the environment constants and methods with the code EnvTest.py.

The directory /SimulationResults can be used to store some graphs and useful data resulting from simulation of trained Policies; the directory /custom_modules contains some useful functions.
