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

    highObsSpace = np.array([1.1 , 1.1, 1.1 , 1.1]) 
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
    
    self.command_scale = 0.5 # scale factor for commands

    # Setting up a goal to reach affecting reward (it seems to work better with humans 
    # rather than forcing them to stay in their original position, and humans are
    # biological neural networks)
    self.X_Pos_Goal = 30. #[m] goal x position
    self.Y_Pos_Goal = 0. #[m] goal y position

    # obstacle position
    self.Obs_X = 10.
    self.Obs_Y = -2.
    
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
      self.state = State_curr_step ## State_curr_step is now updated with RK4

      self.elapsed_time_steps += 1 # update for policy time steps

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division

      # as obs on Y_POsition and X_position, error instead absolut position is given as 
      # observation to the agent, purpose is to minimize (zero) this quantity.
      # error is normalized dividing by the normalization vector stated yet, sign also is given.
      # the distance from obstacle is given as vector length
      X_error = self.state[2] - self.X_Pos_Goal
      Y_error = self.state[3] - self.Y_Pos_Goal

      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))

      obs_state = np.array([self.state[0], self.state[1], X_error, Y_error, obs_dist])
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

      X_error = self.state[2] - self.X_Pos_Goal
      Y_error = self.state[3] - self.Y_Pos_Goal

      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))

      obs_state = np.array([self.state[0], self.state[1], X_error, Y_error, obs_dist])
      obs = obs_state / self.Obs_normalization_vector

      return obs  # produce an observation of the first state (xPosition) 

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      u = self.state[0]
      X_error = self.state[2] - self.X_Pos_Goal
      v = self.state[1]
      Y_error = self.state[3] - self.Y_Pos_Goal
      
      X_obs_dist = self.state[2] - self.Obs_X
      Y_obs_dist = self.state[3] - self.Obs_Y
      obs_dist = np.sqrt((X_obs_dist**2) + (Y_obs_dist**2))

      vel_error_weight = 0.08

      pos_XY_weight = 0.8
      
      obs_dist_weight = 0.1

      #q_weight = 0.1

      if obs_dist >= 5.:

        R = 1. - pos_XY_weight * abs((X_error)/50.)\
          - vel_error_weight * (abs(u/50.))\
            - pos_XY_weight * (abs(Y_error)/50) - vel_error_weight * (abs(v)/50)

      else:

        R = 1. - pos_XY_weight * abs((X_error)/50.)\
          - vel_error_weight * (abs(u/50.))\
            - pos_XY_weight * (abs(Y_error)/50) - vel_error_weight * (abs(v)/50)\
              - obs_dist_weight (1 - obs_dist/5.)
    
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
    
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = self.state

      if Z_1>=5. or Z_1<=-200. : 

        done = True
        print("Z outbound---> ", Z_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(u_1)>=50. :

        done = True
        print("u outbound---> ", u_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(v_1)>=50. :

        done = True
        print("v outbound---> ", v_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(w_1)>=50. :

        done = True
        print("w outbound---> ", w_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(p_1)>=50. :

        done = True
        print("p outbound---> ", p_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(q_1)>=50. :

        done = True
        print("q outbound---> ", q_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(r_1)>=50. :

        done = True
        print("r outbound---> ", r_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(X_1)>=50. :

        done = True
        print("X outbound---> ", X_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(Y_1)>=50. :

        done = True
        print("Y outbound---> ", Y_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(q0_1)>=1.001 or abs(q1_1)>=1.001 or abs(q2_1)>=1.001 or abs(q3_1)>=1.001 :

        done = True
        print("Quaternion outbound...") 
        print("q0 = ", q0_1)
        print("q1 = ", q1_1)
        print("q2 = ", q2_1)
        print("q3 = ", q3_1)
        print("in ", self.elapsed_time_steps, " steps")

      elif self.elapsed_time_steps >= self.max_Episode_time_steps:

        done = True

        print("Episode finished: ", self.elapsed_time_steps, " steps")
        
      else :

        done = False

      return done

## Control section: this section contains all the methods used to get commands and throttles from 
# actions
  def linearAct2ThrMap(self, action):

      """
      Function to use alternatively to act2TrotMap(action). This function performs the mapping 
      linearly from action to throtle, obviously some part of throttle space is cut out, so
      even throttle 1 or zero is impossible to reach and the thr space is compressed to  
      [0.5*dTt, 1.5dTt] 
      input: action [-1, 1]
      output: throttle value
      """

      Thr = self.dTt * (1 + 0.65 * action)

      return Thr

  def act2ThrotMap(self, action):

      """ 
      Function that maps actions into throttle values with constraint reported on the notebook.
      Mapping follows a linear and cubic function defined by constant d1 d2 (right part of the map)
      and s1 s2 (left part). Constant are constrained by [0, 1] output and equal derivative in 
      0-eps, 0+eps input.
      input: a commnad belonging [-1, 1]
      output: mapping belonging [0, 1]
      """

      if action<=0:
        Thr = self.dTt + (self.s1*action) + (self.s2*(action**3))

      else:
        Thr = self.dTt + (self.d1*action) + (self.d2*(action**3))
      
      return Thr

  def getThrsFromControls(self, actions):

      """
      This method provides the mixer function to obtain throttle for the four motors given the external commands 
      for desired vertical thrust (average throttle) and for xb, yb and zb external torques to apply
      input: action array-> components: Average Throttle, Aileron, Elevator and Rudder
      output: array containing throttle for motor 1, 2, 3 and 4

      Note that in this model a positive elevator value is considered the one which generates 
      a positive external pitching torque according to the standard body axes reference.
      """

      ## the throttle mixer function takes the commands and convert them into throttles

      Av_Throttle = self.linearAct2ThrMap(actions[0]) # first action is average throttle,
      #mapped into [0, 1]  

      Aileron = actions[1]
      Elevator = actions[2]
      Rudder = actions[3]

      Throttle_M1 = Av_Throttle + self.Command_scaling_factor * (Elevator - Aileron + Rudder)
      Throttle_M2 = Av_Throttle + self.Command_scaling_factor * (- Elevator - Aileron - Rudder)
      Throttle_M3 = Av_Throttle + self.Command_scaling_factor * (- Elevator + Aileron + Rudder)
      Throttle_M4 = Av_Throttle + self.Command_scaling_factor * (Elevator + Aileron - Rudder)

      Throttle_array = np.array([Throttle_M1, Throttle_M2, Throttle_M3, Throttle_M4])

      ## Check if [0, 1] range is acomplished
      for thr_count in range(4):

        if Throttle_array[thr_count] >= 1.:
          Throttle_array[thr_count] = 1.

        elif Throttle_array[thr_count] <= 0.:
          Throttle_array[thr_count] = 0.

        else: 
          Throttle_array[thr_count] = Throttle_array[thr_count]

      return Throttle_array

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

      drag = - V * self.CdA  #[N]

      return drag

  def dragTorque(self, Omega):
      """
      Function which generates the resistive aerodynamic torque as reported on the notes.
      This torque is assumed to be linear to the angular velocity components and the coefficient
      is the same for each axis. This model is decoupled.
      Input is angular velocity vector and output is Torque vector.
      """

      DragTorque = - self.C_DR * Omega

      return DragTorque

  def Motor(self, Throttle):

      """
      Function which models the motors and props:
      input: Motor throttle [0,1]
      output: Motor thrust and torque.
      The model is assumed to be at V=0 so it is good for hovering but inaccurate when moving forward.
      Pay attention that output are absolute values so vector form and signs have to be adjusted
      where the method is called according to construction choises (For this reason this method 
      does not make any assumption on how to construct vectors)
      """
      ## self.nMax_motor [RPS] number of rounds per second for BL motor
      
      Thrust = self.Prop_Kf * Throttle * (self.nMax_motor**2) #[N]
      Torque = self.Prop_Kq * Throttle * (self.nMax_motor**2) #[N m]
      # The model in which throttle is linear in those formulas, the control is an alias 
      # of F = F_Max * dT so the difference is only on how to consider the command on the 
      # rounds per sec 

      return Thrust, Torque # return scalar thrust and torque

  def eqnsOfMotion(self, State, Throttles):

      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

      # Random process noise on linear accelerations
      if self.Process_perturbations:
        Acc_disturbance = np_normal(0, 0.01, 3) #[m/s^2]
        Omega_dot_dist = np_normal(0, 0.0001, 3) #[rad/s^2]

      else:
        Acc_disturbance = np.zeros(3) #[m/s^2]
        Omega_dot_dist = np.zeros(3) #[rad/s^2]

      Vb = State[0:3] # Subvector CG velocity [m/s]
      Omega = State[3:6] # Subvector angular velocity [rad/s]

      #Performing quaternion normalization
      q0, q1, q2, q3 = State[6:10] # Quaternion

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      q0 = q0/abs_Q
      q1 = q1/abs_Q
      q2 = q2/abs_Q
      q3 = q3/abs_Q

      # Motors section (vectors are evaluated later in this method)
      dT1, dT2, dT3, dT4 = Throttles

      M1_Thrust, M1_Torque = self.Motor(dT1) # scalar values for M1
      M2_Thrust, M2_Torque = self.Motor(dT2) # scalar values for M2
      M3_Thrust, M3_Torque = self.Motor(dT3) # scalar values for M3
      M4_Thrust, M4_Torque = self.Motor(dT4) # scalar values for M4

      # Evaluation of transformation matrix from Body to NED axes: LEB

      LEB = np.array([[(q0**2 + q1**2 - q2**2 - q3**2), 2.*(q1*q2 - q0*q3), 2.*(q0*q2 + q1*q3)], \
        [2.*(q1*q2 + q0*q3), (q0**2 - q1**2 + q2**2 - q3**2), 2.*(q2*q3 - q0*q1)], \
          [2.*(q1*q3 - q0*q2), 2.*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2)]])

      LBE = np.transpose(LEB) # Evaluate transpose of body to NED---> NED to body
      

      #THRUST Evaluation [N] 
      # is evaluated negative because thrust is oriented in the negative verse of Zb
      # according to how props generate the thrust.
      T1 = np.array([0, 0, - M1_Thrust])
      T2 = np.array([0, 0, - M2_Thrust])
      T3 = np.array([0, 0, - M3_Thrust])
      T4 = np.array([0, 0, - M4_Thrust])

      # TORQUES [N m]:
      # as first assumption only the thrust components of the motors combined are considered
      # as torque generator; gyroscopical effects of the props are neglected in this model. 
      # those components are NOT divided by the respective moment of Inertia.
      # Also the aerodynamic drag torque effect is added
      Mtot = np.cross(self.rM1, T1) + np.cross(self.rM2, T2)\
         + np.cross(self.rM3, T3) + np.cross(self.rM4, T4)\
            + np.array([0., 0., (M1_Torque + M3_Torque - M2_Torque - M4_Torque)])\
              + self.dragTorque(Omega)

      # WEIGHT [N] in body axes
      WB = np.dot(LBE, self.Wned) # weight in body axes

      # DRAG [N] in body axes
      DB = self.Drag(Vb)
      
      # TOTAL FORCES in body axes divided by mass [N/kg = m/s^2]
      Ftot_m = (DB + WB + T1 + T2 + T3 + T4) / self.mass 

      # Evaluation of LINEAR ACCELERATION components [m/s^2]
      Vb_dot = - np.cross(Omega, Vb) + Ftot_m + Acc_disturbance

      # Evaluation of ANGULAR ACCELERATION [rad/s^2] components in body axes
      Omega_dot = ((Mtot - np.cross(Omega, np.dot(self.InTen, Omega))) / self.InVec) + Omega_dot_dist
      
      # Evaluation of the cynematic linear velocities in NED axes [m/s]
      # The matrix LEB is written in the equivalent from quaternions components
      # (P_dot stands for Position_dot)
      P_dot = np.dot(LEB, Vb)


      # Evaluation of QUATERNION derivatives 
      p, q, r = Omega
      q0_dot = 0.5 * (-p*q1 - q*q2 - r*q3)
      q1_dot = 0.5 * (p*q0 + r*q2 - q*q3)
      q2_dot = 0.5 * (q*q0 - r*q1 + p*q3)
      q3_dot = 0.5 * (r*q0 + q*q1 - p*q2)

      Q_dot = np.array([q0_dot, q1_dot, q2_dot, q3_dot])

      stateTime_derivatives= np.concatenate((Vb_dot, Omega_dot, Q_dot, P_dot))
      return stateTime_derivatives