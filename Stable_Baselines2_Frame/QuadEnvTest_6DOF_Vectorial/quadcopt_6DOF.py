# Code to generate a Quadcopter Environment to train an RL agent with stable baselines
# this model include a mathematical rapresentation of the quadcopter in which controls are on
# AvgThr, dA, dE, dR and goal is to control Velocity given a reference on V_Nord, V_Down, V_Est
# and locking drift velocity v to 0 in the reward to mix rudder and aileron and also requesting 
# 0 error on the psi angle
import numpy as np
from numpy.random import normal as np_normal
from numpy import cos as cos
from numpy import sin as sin
import gym
from gym import spaces

class QuadcoptEnv_6DOF(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, Random_reset = False, Process_perturbations = False):
    super(QuadcoptEnv_6DOF, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Definition of action space with 1 control action representing average Throttle
    # the other control variables (wich are torques on xb, yb, zb) are given as constant zero
    # as input by the programmer 
    highActionSpace = np.array([1., 1., 1., 1.])
    lowActionSpace = np.array([-1., -1., -1., -1.])
    self.action_space = spaces.Box(lowActionSpace, highActionSpace, dtype=np.float32)

    # Creation of observation space: an array for maximum values is created using a classical Flight Dynamics states 
    # rapresentation with quaternions (state[min, max][DimensionalUnit]): 
    # u[-50,50][m/s], v[-50,50][m/s], w[-50,50][m/s], p[-50,50][rad/s], q[-50,50][rad/s], r[-50,50][rad/s],...
    #  q0[-1,1], q1[-1,1], q2[-1,1], q3[-1,1], X[-50,50][m], Y[-50,50][m], Z[-100,0][m].
    # To give normalized observations boundaries are fom -1 to 1, the problem scale 
    # is adapted to the physical world in step function.
    # The normalization is performed using limits reported above

    ##full visibility on the states as given in the previous section.
    # velocity is given as V_Nord , V_est and V_down errors and v to be set to 0
    # order is V_N_Err, V_E_Err, V_D_Err, Psi_error, p, q, r, q0, q1, q2, q3
    highObsSpace = np.array([1.1 , 1.1, 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1, 1.1, 1.1])
    lowObsSpace = -highObsSpace
    

    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32) # boundary is set 0.1 over to avoid AssertionError

    # A vector with max value for each state is defined to perform normalization of obs
    # so to have obs vector components between -1,1. The max values are taken acording to 
    # previous comment
    self.Obs_normalization_vector = np.array([20., 20., 20., 2*np.pi, 50., 50., 50., 1., 1., 1., 1.]) # normalization constants
    # Random funcs
    self.Random_reset = Random_reset # true to have random reset
    self.Process_perturbations = Process_perturbations # to have random accelerations due to wind
    

    self.Lx = 0.34   #[m] X body Length (squared x configuration)
    self.Ly = 0.34   #[m] Y body Length
    
    # Motors position vectors from CG
    self.rM1=np.array([self.Lx/2, 0., 0.]) 
    self.rM2=np.array([0., self.Ly/2, 0.])
    self.rM3=np.array([-self.Lx/2, 0., 0.])
    self.rM4=np.array([0., -self.Ly/2, 0.]) 

    # atmosphere and gravity definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 #[kg/m^3] Standard day at 0 m ASL
    self.g0 = 9.815  #[m/s^2] gravity acceleration

    # mass of components
    self.mass = 0.71   #[kg] mass is 710 grams can be changed
    self.motor_mass = 0.04 #[kg] mass of one motor+prop
    self.body_mass= 0.484 #[kg] mass of body frame + electronics (for inertia it is considered as 
    # uniformly distributed in a sphere centered in CG with radius 0.06m)
    self.battery_mass = 0.186 #[kg] mass of battery, considered at a distance of 0.06m from CG aligned with it on zb
    
    self.Wned = np.array([0, 0, self.mass * self.g0]) # Weight vector in NED axes
   
    ## Inertia tensor is considered dyagonal, null the other components
    self.Ix = 2*((self.Ly/2)**2)*self.motor_mass +\
      (0.06**2)*self.battery_mass + 0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to X axis
    
    self.Iy = 2*((self.Lx/2)**2)*self.motor_mass +\
      (0.06**2)*self.battery_mass + 0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to Y axis
    
    self.Iz = 4*((self.Lx/2)**2)*self.motor_mass +\
      0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to Z axis

    # Inertia tensor composition
    self.InTen = np.array([[self.Ix, 0., 0.],[0., self.Iy, 0.],[0., 0., self.Iz]])

    # Inertia vector: vector with 3 principal inertia useful in evaluating the Omega_dot
    self.InVec = np.diag(self.InTen)

    ## The motors model is now assumed as reported on the notebook with thrust and torques dependant on 
    # a constant multiplied by the square of prop's rounds per sec:
    # F = Kt * n**2 where n[rounds/s] = Thr * nMax and nMax is evaluated as Kv*nominal_battery_voltage/60
    # self.Motor_Kv = 1200. # [RPM/V] known for te specific motor
    # self.V_batt_nom = 11.1 # [V] nominal battery voltage 
    self.nMax_motor = 8500 / 60 #[RPS] as the hummingbird, max RPM can be limited via software 

    # Props Values
    self.D_prop = 0.2032 #[m] diameter for 7 inch prop
    self.Ct = 0.1087 # Constant of traction tabulated for V=0 
    self.Cp = 0.0477  # Constant of power tabulated for v=0
    self.Prop_Kf = self.Ct * self.rho * (self.D_prop**4) #[kg m]==[N/RPS^2]
    self.Prop_Kq = self.Cp * self.rho * (self.D_prop**5)/(2*np.pi) #[kg m^2]
    # now force and torque are evaluated as:
    # F=Kf * N_prop^2 
    # F=Kq * N_prop^2 in an appropriate method   
    # N are rounds per sec (not rad/s) 

    # Throttle constants for mapping (mapping is linear-cubic, see the act2Thr() method)
    self.dTt = (self.mass * self.g0 / (4*self.Prop_Kf)) / (self.nMax_motor**2) # trim throttle to hover
    self.d2 = 1 - self.dTt - 0.3 # Assumed value for first constant in action to throttle mapping
    self.d1 = 1 - self.d2 - self.dTt # second constant for maping (see notebook)
    self.s2 = self.d2 - 1 + 2*self.dTt # first constant for left part
    self.s1 = self.dTt - self.s2

    # Commands coefficients
    self.Command_scaling_factor = 0.3 # Coefficient to scale commands when evaluating throttles of motors
    # given the control actions    
    
    self.CdA = np.array([0.05, 0.05, 0.4]) #[kg/s] drag constant on linear aerodynamical drag model
    # linear aerodynamics considered self.Sn = np.array([0.02, 0.02, 0.05]) #[m^2] Vector of normal surfaces to main body axes to calculate drag
    # Zb normal surface is greater than othe two  

    self.C_DR = np.array([0.02, 0.02, 0.005]) # [kg m^2/s] coefficients are evaluated with aid of the 
    # Arena and apoleoni thesis
    

    # integration parameters: The dynamics simulation time is different from the time step size
    # for policy deployment and evaluation: 
    # For the dynamics a dynamics_timeStep is used as 0.01 s 
    # The policy time steps is 0.05 (this step is also the one taken outside)
    self.dynamics_timeStep = 0.01 #[s] time step for Runge Kutta 
    self.timeStep = 0.04 #[s] time step for policy
    self.max_Episode_time_steps = int(6*10.24/self.timeStep) # maximum number of timesteps in an episode (=20s) here counts the policy step
    self.elapsed_time_steps = 0 # time steps elapsed since the beginning of an episode, to be updated each step
    

    # Constants to normalize state and reward
    
    # Setting up a goal to reach affecting reward (it seems to work better with humans 
    # rather than forcing them to stay in their original position, and humans are
    # biological neural networks)
    self.V_NED_ref = np.zeros(3) #[m/s] [V_Nord_ref, V_Est_ref, V_Down_ref]

    ## Heading reference angle is evaluated separately and independently from the other parameters
    # so to let the orientation command being independent
    self.psi_ref = 0. #[rad] interval (-pi, pi]

  def step(self, action):

      # State-action variables assignment
      State_curr_step = self.state # self.state is initialized as np.array, this temporary variable is used than in next step computation 
      
      controls = action ## This variable is used to make possible the separation of actions 
      # in this example actions represent pseudo controls

      Throttles = self.getThrsFromControls(controls) # commands are the actions given by the policy

      h = self.dynamics_timeStep # Used only for RK
      
      ###### INTEGRATION OF DYNAMICS EQUATIONS ###
      for _RK_Counter in range(int(self.timeStep/self.dynamics_timeStep)): 
        # to separate the time steps the integration is performed in a for loop which runs
        # into the step function for nTimes = policy_timeStep / dynamics_timeStep

        # Integration of the equation of motion with Runge-Kutta 4 order method
        ## The state derivatives funcion xVec_dot = fvec(x,u) is implemented in a separate function
        k1vec = h * self.eqnsOfMotion(State_curr_step, Throttles) # Action stays unchanged for this loop
        k2vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k1vec), Throttles) # K2 from state+K1/2
        k3vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k2vec), Throttles) # K3 from state+k2/2
        k4vec = h * self.eqnsOfMotion(np.add(State_curr_step, k3vec), Throttles) # K4 from state+K3
        # Final step of integration 
        State_curr_step = State_curr_step + (k1vec/6) + (k2vec/3) + (k3vec/3) + (k4vec/6)

      ######### COMPOSITION OF STEP OUTPUT
      # self.state variable assignment with next step values (step n+1)
      self.state = State_curr_step ## State_curr_step is now updated with RK4

      self.elapsed_time_steps += 1 # update for policy time steps

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division

      V_NED_Err, V_NED = self.getV_NED_error()

      PHI = self.quat2Att() 

      Psi_ref = self.psi_ref # psi ref is taken at the reset
      #Psi_err = self.thnorm(Psi_ref - PHI[2])
      Psi_err = Psi_ref - PHI[2]

      if Psi_err>np.pi:
        Psi_err = Psi_err - (2 * np.pi)

      elif Psi_err<-np.pi:
        Psi_err = Psi_err + (2 * np.pi)

      obs_state = np.concatenate(([V_NED_Err[0], V_NED_Err[1], V_NED_Err[2], Psi_err], self.state[3:10])) #, self.state[1] removed visibility over v
      obs = obs_state / self.Obs_normalization_vector

      # REWARD evaluation and done condition definition (to be completed)
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = State_curr_step

      reward = self.getReward()

      done = self.isDone()
    
      info = {"u": u_1, "v": v_1, "w": w_1, "p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1, "X": X_1, "Y": Y_1, "Z": Z_1, "V_Nord": V_NED[0], "V_Est": V_NED[1], "V_Down": V_NED[2]}

      return obs, reward, done, info

  def reset(self):

      """
      Reset state 
      """

      if self.Random_reset:
        w_reset = 0. #[m/s]
        Z_reset = -30. #[m]
        u_reset = 0. #[m/s]
        X_reset = 0. #[m]
        v_reset = 0. #[m/s]
        Y_reset = 0. #[m]

        p_reset = np_normal(0., 0.0175)
        q_reset = np_normal(0., 0.0175)
        r_reset = np_normal(0., 0.0175)

        phi = np_normal(0., 0.05) #[rad]
        theta = np_normal(0., 0.05) #[rad]
        psi = np_normal(0., 175.*0.0175)    #np_normal(120. * 0.0175, 20 * 0.0175) #[rad]

        if psi >= np.pi - 0.0175:
          psi = np.pi - 0.0175

        elif psi <= -np.pi + 0.035:
          psi = -np.pi + 0.035

        q0_reset = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
        q1_reset = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
        q2_reset = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
        q3_reset = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)

        self.V_NED_ref[0] = np_normal(0., 1.) #[m/s]
        self.V_NED_ref[1] = np_normal(0., 1.) #[m/s]
        self.V_NED_ref[2] = np_normal(0., 1.5) #[m/s]

        self.psi_ref = np_normal(0., 175.*0.0175)

        if self.psi_ref >= np.pi - 0.0175:
          self.psi_ref = np.pi - 0.0175

        elif self.psi_ref <= - np.pi + 0.035:
          self.psi_ref = - np.pi + 0.035

      else:
        w_reset = 0. #[m/s]
        Z_reset = -28. #[m]
        u_reset = 0. #[m/s]
        X_reset = 0. #[m]
        v_reset = 0. #[m/s]
        Y_reset = 0. #[m]

        p_reset = 0.
        q_reset = 0.
        r_reset = 0.

        q0_reset = 1.
        q1_reset = 0.
        q2_reset = 0.
        q3_reset = 0.      

        self.VNord_ref = 0. #[m/s]
        self.VEst_ref = 0 #[m/s]
        self.VDown_ref = -1.5 #[m/s]

        self.psi_ref = 0.

      self.state = np.array([u_reset,v_reset,w_reset,p_reset,q_reset,r_reset,q0_reset,q1_reset,q2_reset,q3_reset,X_reset,Y_reset,Z_reset]) # to initialize the state the object is put in x0=20 and v0=0
      
      self.elapsed_time_steps = 0 # reset for elapsed time steps

      V_NED_Err, V_NED = self.getV_NED_error()

      PHI = self.quat2Att()  

      Psi_ref = self.psi_ref
      #Psi_err = self.thnorm(Psi_ref - PHI[2])
      Psi_err = Psi_ref - PHI[2]


      if Psi_err>np.pi:
        Psi_err = Psi_err - (2 * np.pi)

      elif Psi_err<-np.pi:
        Psi_err = Psi_err + (2 * np.pi)

      obs_state = np.concatenate(([V_NED_Err[0], V_NED_Err[1], V_NED_Err[2], Psi_err], self.state[3:10])) #, self.state[1] removed visibility over v
      obs = obs_state / self.Obs_normalization_vector

      return obs  # produce an observation of the first state (xPosition) 

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      V_NED_Err, V_NED = self.getV_NED_error()

      p, q, r = self.state[3:6]
      v = self.state[1]

      PHI = self.quat2Att()    
      Psi_ref = self.psi_ref
      #Psi_err = self.thnorm(Psi_ref - PHI[2])
      Psi_err = Psi_ref - PHI[2]

      if Psi_err>=np.pi:
        Psi_err = Psi_err - (2 * np.pi)

      elif Psi_err<-np.pi:
        Psi_err = Psi_err + (2 * np.pi)

      V_error_Weight = 0.8
      drift_weight = 0.8
      rate_weight = 0.6

      # There should be sa fourth constraint to assign values at the dR controls, this constraint is to 
      # maximize the q0 wich is maximum (=1) if the rotation between NED and body is null, since the 
      # reference velocities are small and can be achieved with small attitude changes, requesting none rotation
      # or maximum q0 is a simple way to impose that Xb stays pointed Northbound, the assumption is that the 
      # policy will learn to pitch or roll slightly to acomplish the NED velocity required while keeping the 
      # ultirotor with the node pointed North, this shows to work in the Position Reference model.

      # The second assmption about this fact is that the Xb axis stays pointed in the direction of motion 
      # in this second case the reward is higher if the psi error is 0, psi error is evaluated calculating psi ref as
      # the heading of the direction of motion (atan2(V_nord_ref / V_est_ref)), here is necessary to 
      # remove the q0 from the reward since psi error can be null also in position of maximum psi = -pi
      # where q0 = 0 (being psi and theta = 0); is good to give a visibility to the policy on all the  
      # components of the reward.
      R = (1.) - V_error_Weight * (abs(V_NED_Err[0]/20.) + abs(V_NED_Err[1]/20.) + abs(V_NED_Err[2]/20.))\
        - rate_weight * (abs(p/50.) + abs(q/50.) + abs(r/50.)) - drift_weight * (abs(v/8.) + abs(Psi_err/2*np.pi))

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

      if abs(u_1)>=20. :

        done = True
        print("u outbound---> ", u_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(v_1)>=20. :

        done = True
        print("v outbound---> ", v_1, "   in ", self.elapsed_time_steps, " steps")

      elif abs(w_1)>=20. :

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

      Thr = self.dTt * (1 + 0.9 * action)

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

      Throttle_M1 = Av_Throttle + self.Command_scaling_factor * (Elevator + Rudder)
      Throttle_M2 = Av_Throttle + self.Command_scaling_factor * (- Aileron - Rudder)
      Throttle_M3 = Av_Throttle + self.Command_scaling_factor * (- Elevator + Rudder)
      Throttle_M4 = Av_Throttle + self.Command_scaling_factor * (Aileron - Rudder)

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

## Conversion from quaternions to Phi_theta
  def quat2Att(self):

      """
      Function to convert from quaternion to attitude angles, for simplicity only phi and theta are the output
      input: Q [array_like, quaternion]
      Output: Phi, Theta 
      """

      q0, q1, q2, q3 = self.state[6:10] # Quaternion

      theta_arg = 2*(q0*q2-q3*q1)

      if theta_arg>0.999999:
        theta_arg = 0.999999

      elif theta_arg<-0.999999:
        theta_arg = -0.999999

      Phi = np.arctan2(2*(q0*q1 + q2*q3), 1-2*(q1**2+q2**2))
      Theta = np.arcsin(theta_arg)
      Psi = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

      return Phi, Theta, Psi

  def getV_NED_error(self):

      """
      Function which provides the array containing the errors on velocity in NED axis, takes self.state 
      and self.V_NED_ref as input
      output: np.array[V_Nord_err, V_Est_err, V_Down_err]
      """

      q0, q1, q2, q3 = self.state[6:10] # Quaternion
      Vb = self.state[0:3]

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      q0 = q0/abs_Q
      q1 = q1/abs_Q
      q2 = q2/abs_Q
      q3 = q3/abs_Q

      LEB = np.array([[(q0**2 + q1**2 - q2**2 - q3**2), 2.*(q1*q2 - q0*q3), 2.*(q0*q2 + q1*q3)], \
        [2.*(q1*q2 + q0*q3), (q0**2 - q1**2 + q2**2 - q3**2), 2.*(q2*q3 - q0*q1)], \
          [2.*(q1*q3 - q0*q2), 2.*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2)]])

      V_NED = np.dot(LEB, Vb)

      V_NED_Err = V_NED - self.V_NED_ref

      return V_NED_Err, V_NED

  def thnorm(self, X):

      """
      function which return a normalized value of the angle diven as argument
      input: X angle in radians
      outut: atan(sin(in), cos(in))
      """

      out = np.arctan2(np.sin(X), np.cos(X))

      return out

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

      if self.Process_perturbations: ## adds some randomness to the motor performances
        rand_thrust = np_normal(0., 0.1) 
        rand_torque = np_normal(0., 0.1)

      else:
        rand_thrust = 0.
        rand_torque = 0.
      
      Thrust = (1 + rand_thrust) * self.Prop_Kf * Throttle * (self.nMax_motor**2) #[N]
      Torque = (1 + rand_torque) * self.Prop_Kq * Throttle * (self.nMax_motor**2) #[N m]
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
        Omega_dot_dist = np_normal(0, 0.00175, 3) #[rad/s^2]

      else:
        Acc_disturbance = np.zeros(3) #[m/s^2]
        Omega_dot_dist = np.zeros(3) #[rad/s^2]

      Vb = State[0:3] # Subvector CG velocity [m/s]
      Omega = State[3:6] # Subvector angular velocity [rad/s]

      #Performing quaternion normalization and bounding
      q0, q1, q2, q3 = State[6:10] # Quaternion

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      
      q0 = q0/abs_Q
      if abs(q0)>=1.:
        q0 = 1.*np.sign(q0)

      q1 = q1/abs_Q
      if abs(q1)>=1.:
        q1 = 1.*np.sign(q1)

      q2 = q2/abs_Q
      if abs(q2)>=1.:
        q2 = 1.*np.sign(q2)

      q3 = q3/abs_Q
      if abs(q3)>=1.:
        q3 = 1.*np.sign(q3)

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