# This code implements a 2D environment of material point to train a model for trajectory 
# generation. the model is in X,Y plane and the states are V, psi, X, Y.
# Controls from action u1, u2 can control psi dot ancd Vdot
 
import numpy as np
from numpy.random import normal as np_normal
import gym
from gym import spaces

class Navigation(gym.Env):
  """2D Navigation environment"""
  metadata = {'render.modes': ['human']}

  def __init__(self, Random_reset = False, Process_perturbations = False):
    super(Navigation, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Definition of action space with 1 control action representing V dot and
    # the other control variable is psi dot
    highActionSpace = np.array([1., 1.])
    lowActionSpace = -highActionSpace
    self.action_space = spaces.Box(lowActionSpace, highActionSpace, dtype=np.float32)

    # Creation of observation space: an array for maximum values is created 
    # obs are evaluated as distance from goal and distance from obstacle 

    highObsSpace = np.array([1.1 , 1.1, 1.1 , 1.1, 1.1]) 
    lowObsSpace = -highObsSpace
    # the order is: [u, v, X_error_wp, Y_error_wp, Obs_dist] X_error_wp is error from wp from waypoint
    # Obs_dist is distance from obstacle

    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32) # boundary is set 0.1 over to avoid AssertionError

    # A vector with max value for each state is defined to perform normalization of obs
    # so to have obs vector components between -1,1. The max values are taken acording to 
    # previous comment
    self.Obs_normalization_vector = np.array([50., 50., 50., 50., 50.])
                                        
    self.state = None

    # Random utils
    self.Random_reset = Random_reset # true to have random reset
    self.Process_perturbations = Process_perturbations # to have random accelerations due to wind
    
    # atmosphere and gravity definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 #[kg/m^3] Standard day at 0 m ASL
    self.g0 = 9.815  #[m/s^2] gravity acceleration

    # mass of components
    self.mass = 0.71   #[kg] mass is 600 grams can be changed
    
    self.CdA = np.array([0.1, 0.1]) / self.mass #[1/s] drag constant on linear aerodynamical drag model
    # linear aerodynamics considered normalized by mass  

    # integration parameters: The dynamics simulation time is different from the time step size
    # for policy deployment and evaluation: 
    # For the dynamics a dynamics_timeStep is used as 0.01 s 
    # The policy time steps is 0.05 (this step is also the one taken outside)
    self.dynamics_timeStep = 0.1 #[s] time step for Runge Kutta 
    self.timeStep = 0.4 #[s] time step for policy
    self.max_Episode_time_steps = int(12*10.24/self.timeStep) # maximum number of timesteps in an episode (=20s) here counts the policy step
    self.elapsed_time_steps = 0 # time steps elapsed since the beginning of an episode, to be updated each step
    self.Tot_Traj_len = None # variable to store total trajectory len
    self.max_tot_traj_len = 200. # maximum possible length of trajectory

    self.command_scale = 0.5 # scale factor for commands

    # Setting up a goal to reach affecting reward (it seems to work better with humans 
    # rather than forcing them to stay in their original position, and humans are
    # biological neural networks)
    self.X_Pos_Goal = 15. #[m] goal x position
    self.Y_Pos_Goal = 0. #[m] goal y position

    self.desired_min_obs_dist = 7.5 #minimum obs dist desired

    # obstacle position
    self.Obs_X = 10.
    self.Obs_Y = 0.
    
  def step(self, action):

      # State-action variables assignment
      State_curr_step = self.state # self.state is initialized as np.array, this temporary variable is used than in next step computation 

      h = self.dynamics_timeStep # Used only for RK
      
      ###### INTEGRATION OF DYNAMICS EQUATIONS ###
      for _RK_Counter in range(int(self.timeStep/self.dynamics_timeStep)): 
        # to separate the time steps the integration is performed in a for loop which runs
        # into the step function for nTimes = policy_timeStep / dynamics_timeStep

        # Integration of the equation of motion with Runge-Kutta 4 order method
        ## The state derivatives funcion xVec_dot = fvec(x,u) is implemented in a separate function
        k1vec = h * self.eqnsOfMotion(State_curr_step, action) # Action stays unchanged for this loop
        k2vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k1vec), action) # K2 from state+K1/2
        k3vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k2vec), action) # K3 from state+k2/2
        k4vec = h * self.eqnsOfMotion(np.add(State_curr_step, k3vec), action) # K4 from state+K3
        # Final step of integration 
        State_curr_step = State_curr_step + (k1vec/6) + (k2vec/3) + (k3vec/3) + (k4vec/6)

      ######### COMPOSITION OF STEP OUTPUT
      # self.state variable assignment with next step values (step n+1)

      ## Update of total length percurred
      X_dist = State_curr_step[2] - self.state[2] 
      Y_dist = State_curr_step[3] - self.state[3] 
      dist_curr_step = np.sqrt((X_dist**2) + (Y_dist**2))

      self.Tot_Traj_len += dist_curr_step

      self.state = State_curr_step ## State_curr_step is now updated with RK4

      self.elapsed_time_steps += 1 # update for policy time steps

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division

      # as obs on Y_POsition and X_position, error instead absolut position is given as 
      # observation to the agent, purpose is to minimize (zero) this quantity.
      # error is normalized dividing by the normalization vector stated yet, sign also is given.
      # the distance from obstacle is given as vector length
      X_error = - self.state[2] + self.X_Pos_Goal
      Y_error = - self.state[3] + self.Y_Pos_Goal

      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))
      obsv_obs_dist = obs_dist - self.desired_min_obs_dist

      obs_state = np.array([self.state[0], self.state[1], X_error, Y_error, obsv_obs_dist])
      obs = obs_state / self.Obs_normalization_vector

      # REWARD evaluation and done condition definition (to be completed)
      u_1, v_1, X_1, Y_1 = State_curr_step

      reward = self.getReward()

      done = self.isDone()
    
      info = {"u": u_1, "v": v_1, "X": X_1, "Y": Y_1}

      return obs, reward, done, info

  def reset(self):

      """
      Reset state 
      """

      if self.Random_reset:
        u_reset = np_normal(0., 0.025) #[m/s]
        v_reset = np_normal(0., 0.025) #[m/s]
        X_reset = np_normal(0., 2.) #[m]
        Y_reset = np_normal(0., 2.) #[m]

      else:
        u_reset = 0. #[m/s]
        v_reset = 0. #[m/s]
        X_reset = 0. #[m]
        Y_reset = 0. #[m]

      self.state = np.array([u_reset, v_reset, X_reset, Y_reset]) # to initialize the state the object is put in x0=20 and v0=0
      
      self.elapsed_time_steps = 0 # reset for elapsed time steps
      self.Tot_Traj_len = 0.

      X_error = - self.state[2] + self.X_Pos_Goal
      Y_error = - self.state[3] + self.Y_Pos_Goal

      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))
      obsv_obs_dist = obs_dist - self.desired_min_obs_dist

      obs_state = np.array([self.state[0], self.state[1], X_error, Y_error, obsv_obs_dist])
      obs = obs_state / self.Obs_normalization_vector

      return obs  # produce an observation of the first state (xPosition) 

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      u = self.state[0]
      X_error = - self.state[2] + self.X_Pos_Goal
      v = self.state[1]
      Y_error = - self.state[3] + self.Y_Pos_Goal

      len_tot = self.Tot_Traj_len
      
      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))
      obsv_obs_dist = obs_dist - self.desired_min_obs_dist
      
      vel_error_weight = 0.3

      pos_XY_weight = 1.
      
      obs_dist_weight = 0.9

      traj_len_weight = 0.001

      #q_weight = 0.1

      R = 1. - pos_XY_weight * abs((X_error)/50.)\
        - vel_error_weight * (abs(u/50.))\
          - pos_XY_weight * (abs(Y_error)/50.) - vel_error_weight * (abs(v)/50.)\
            - obs_dist_weight * (obsv_obs_dist/50.) - traj_len_weight * len_tot/self.max_tot_traj_len
    
      if R >= 0:
        reward = R

      else:
        reward = 0

      ## Added to the reward the goals on space and height to look for zero drift on position      

      return reward

  def isDone(self):

      """
      return a bool condition True if any state falls outbound normalization vector
      components assumption. prints some indications on which state caused done.
      Dimensional unit reported in the comment in __init__()
      input: evaluates from self.state
      output: boolean done variable 
      """
    
      u_1, v_1, X_1, Y_1 = self.state

      if abs(u_1)>=50. :

        done = True
        print("u outbound---> ", u_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(v_1)>=50. :

        done = True
        print("v outbound---> ", v_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(X_1)>=50. :

        done = True
        print("X outbound---> ", X_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(Y_1)>=50. :

        done = True
        print("Y outbound---> ", Y_1, "   in ", self.elapsed_time_steps, " steps")

      elif self.Tot_Traj_len >= self.max_tot_traj_len:

        done = True

        print("LENGTH outbound in ", self.elapsed_time_steps, " steps")
      

      elif self.elapsed_time_steps >= self.max_Episode_time_steps:

        done = True

        print("Episode finished: ", self.elapsed_time_steps, " steps")
        
      else :

        done = False

      return done

## In this sections are defined functions to evaluate forces and derivatives to make the step function easy to read

  def Drag(self, V):
    
      """
      This function return an Aerodynamical drag given velocity and cd.
      Linear model is considered to take into account rotor effects on the longitudinal forces
      input: relative wind speed.
      output: drag force (vector if input is a vector)
      """
      # Evaluation of the AERODYNAMICAL DRAG: this force is modeled as 3 scalar values
      # calculated with the formula D=0.5 S Cd rho V^2 where V is the single component 
      # of velocity in body axes and S and Cd are referred to the surface normal to the 
      # selected V component: E.G. to evaluate X component of drag V = u and S and Cd are 
      # those referred to the front section.
      # Evaluation performed in vector form

      drag = - V * self.CdA  #[N/kg]

      return drag

  def eqnsOfMotion(self, State, control):

      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

      # Random process noise on linear accelerations
      if self.Process_perturbations:
        Acc_disturbance = np_normal(0, 0.01, 2) #[m/s^2]

      else:
        Acc_disturbance = np.zeros(2) #[m/s^2]

      V = State[0:2] # Subvector CG velocity [m/s]

      # DRAG [N/kg] in body axes
      DB = self.Drag(V)
      
      # TOTAL FORCES in body axes divided by mass [N/kg = m/s^2]
      Ftot_m = DB + control * (self.g0 * self.command_scale) 

      # Evaluation of LINEAR ACCELERATION components [m/s^2]
      Vb_dot = Ftot_m + Acc_disturbance
      
      # Evaluation of the cynematic linear velocities in NE axes [m/s]
      # (P_dot stands for Position_dot)
      P_dot = V

      stateTime_derivatives= np.concatenate((Vb_dot, P_dot))

      return stateTime_derivatives