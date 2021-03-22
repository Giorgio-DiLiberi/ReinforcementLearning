# RL for UAV guidance and control

This Repository contains all the codes developed to train a Neural Network using Reinforcement Learning algorithms to control a Multirotor environment.

The Model of the multirotor is encoded in a gym_env format; there are some different folders containing different models of the Multirotor components.

## Required Packages

Dirctory root/Stable_Baselines2_Frame contains codes to train and simulate RL agents on Quadcopter environment using [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) frame work. Please follow the instructions on [this page](https://stable-baselines.readthedocs.io/en/master/guide/install.html) to install all the required packages.
I used python 3.7.9 and Tensorflow 1.15.

The Directory root/Stable_Baselines3_Frame contains codes to train and deploy RL agent on a Quadrotor environment using S-Bs 3 frame work, please visit [this page](https://stable-baselines3.readthedocs.io/en/master/) for all the docmentation and installation instructions.
The stable Baselines 3 codes are currently not updated.

The directory SBs_2_on_remote contains codes to be deployed on the Sapienza remote machine. the are no particular issues in running those codes in a PC but the names of check files for policy to save or load.

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

* Directory QuadEnv5: in this model the actions represents the commands on external torques in order: Averge throttle, Aileron command, Elevator command, Rudder command, those commands are mixed in appropriate method.

* Directory QuadEnvTest_6DOF: it is a complete model with observability on all the states and control over all the 4 pseudo commands: Average throttle, Aileron, Elevator and Rudder.

* Directory Trivial_problems contains some test models in which the commands are unpacked EG.: the policy can control only the average thrust or the pitching torque and other controls can be decided directly by the programmer; the models are equals to the others except for this change on action space.

* Directory Trajectory_gen is a simple 2D environment to train a simple policy for obstacle avoidance and waypoint navigation, this model does not include dynamics and rotation.

## Usage of a QuadEnv directory

The directory can be used for:

* train a policy launching the code Training.py, the code save the best policy from EvalCallback() in the directory /EvalClbkLogs, the last policy obtained in /Policies and the tensorboard logs in /tensorboardLogs, to show the logs execute the command:

    ```bash
    tensorboard --logdir ./tensorboardLogs/
    ```

* simulate a saved policy using the code simulator.py which ask for simulating the best policy, the last policy or to select a specific policy, the simulators in SBs2 use pdf to save plots due to matplolib issues in virtualenvironments

* test some user defined actions with the code actionTest.py

* make some debugging on the environment constants and methods with the code EnvTest.py.

The directory /SimulationResults can be used to store some graphs and useful data resulting from simulation of trained Policies; the directory /custom_modules contains some useful functions.

## Create a virtualenv to use old packages

Here is a how-to create a virtualenv on pc and run it, after run is it possible to install packages on the environment and run them without affecting the main pc packages.
In this example I use python 3.7:

* Install the venv package for the python version:

* To create a virtualenv:

    ```bash
    python3.7 -m venv /home/user/virtualEnvsFolder/virtualEnvsName
    ```
    
* To activate the environment (from home/user dir):

    ```bash
    source /home/user/virtualEnvsFolder/virtualEnvsName/bin/activate
    ```
    
* To deactivate the environment while it is still active:

    ```bash
    deactivate
    ```
    
See the [venv webpage](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) for further information on virtualenvs.
