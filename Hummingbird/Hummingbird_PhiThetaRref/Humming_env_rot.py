# Code to implement an environment of a 8inch props quadcopter in "plus configuration"
# like the hummigbird by Asc. Tech. This code tries to stay more generale as possible
# so it will be good to simulate any quadcopter with 8 inches props in + config
# because it tries to simulate only the physics of the model without any stack of 
# control systems or stability augmentation systems. 
# This code is supposed to simulate and train a policy to accomodate pilot inputs 
# in terms of Phi_ref, Theta_ref, r_ref and implements only the rotation dynamics to integrate

## this model simulate only the rotation dynamics to train an agent to control the rotation angles
import numpy as np
from numpy.random import normal as np_normal
from numpy import cos as cos
from numpy import sin as sin
import gym
from gym import spaces

class Hummingbird_3DOF(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, Random_reset = False, Process_perturbations = False):
    super(Hummingbird_3DOF, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # the controls re given as pseudo commands for aileron, elevator and rudder
    # so torques are controlled in order to achieve an attitude control.
    # The motor model tries to be as accurate as possible 
    highActionSpace = np.array([1., 1.])
    lowActionSpace = np.array([-1., -1.])
    self.action_space = spaces.Box(lowActionSpace, highActionSpace, dtype=np.float32)

    # The states are composed only of the three rotation velocity and the three angles

    # visibility is gven only on 5 parameters which are pitch and roll angles and angular velocities

    # pitch and roll angles are evaluated from the quaternion with the convertion relations and than error 
    # from the request is calculated. error is given in radians and maximum possible values are 45 degs
    highObsSpace = np.array([1.1 , 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])    
    lowObsSpace = -highObsSpace
    

    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32) # boundary is set 0.1 over to avoid AssertionError

    # A vector with max value for each state is defined to perform normalization of obs
    # so to have obs vector components between -1,1. The max values are taken acording to 
    # previous comment
    self.Obs_normalization_vector = np.array([50., 50., 50., 1., 1., 1., 1.]) # normalization constants
    # maximum values for pitch and roll angles are 1 rad. those values will affect the done condition
    # Random funcs
    self.Random_reset = Random_reset # true to have random reset
    self.Process_perturbations = Process_perturbations # to have random accelerations due to wind
    

    # plus configuration
    self.Lx = 0.34   #[m] X body Length (squared x configuration)
    self.Ly = 0.34   #[m] Y body Length
    
    # Motors position vectors from CG in body frame coordinates
    # motors are in plus config so M1 is front, M2 is right, M3 is back and M4 is left 
    self.rM1=np.array([self.Lx/2, 0., 0.]) # M1 for front motor
    self.rM2=np.array([0., self.Ly/2, 0.]) # M2 for left so y > 0
    self.rM3=np.array([-self.Lx/2, 0., 0.]) # M3 is back
    self.rM4=np.array([0., -self.Ly/2, 0.]) # M4 is left

    # M1 is mounted for CounterClockwise rotation so the reaction torque applied on the frame 
    # is positive in body axes. The other motors specs are consequent in terms of reaction torque

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
    self.Motor_Kv = 900. # [RPM/V] known for te specific motor
    self.V_batt_nom = 11.1 # [V] nominal battery voltage 
    self.nMax_motor = 8500 / 60 #[RPS] set from the thesis

    # Props Values
    self.D_prop = 0.2032 #[m] diameter for 7 inch prop
    self.prop_mean_chord = 0.015 #[m] 1.5 cm as mean chord for prop profile
    self.prop_Theta0 = 0.21 #[rad] = atan(pitch / (2*pi*0.75*Rprop))
    self.Ct = 0.1087 # Constant of traction tabulated for V=0
    self.Cp = 0.0477  # Constant of power tabulated for v=0
    self.Prop_Kf = self.Ct * self.rho * (self.D_prop**4) #[kg m]==[N/RPS^2]
    self.Prop_Kq = self.Cp * self.rho * (self.D_prop**5)/(2*np.pi) #[kg m^2]
    self.Prop_KqF = self.Prop_Kq / self.Prop_Kf #[m] proportional linear term to obtain effective torque from thrust
    # now force and torque are evaluated as:
    # F=Kf * N_prop^2 
    # F=Kq * N_prop^2 in an appropriate method   
    # N are rounds per sec (not rad/s) 

    # induced velocity in hovering due to momentum theory (no profile drag accounted) is
    # vh = sqrt(T hovering / (2rho Ad))
    self.vh = np.sqrt(0.25 * self.g0 * self.mass / (0.25 * np.pi * (self.D_prop**2))) # [m/s] 

    # Throttle constants for mapping (mapping is linear, see the linearact2Thr() method)
    self.dTt = (self.mass * self.g0 / (4*self.Prop_Kf)) / (self.nMax_motor**2) # trim throttle to hover
    self.d2 = 1 - self.dTt - 0.3 # Assumed value for first constant in action to throttle mapping
    self.d1 = 1 - self.d2 - self.dTt # second constant for maping (see notebook)
    self.s2 = self.d2 - 1 + 2*self.dTt # first constant for left part
    self.s1 = self.dTt - self.s2

    # Commands coefficients
    self.Command_scaling_factor = 0.18 # Coefficient to scale commands when evaluating throttles of motors
    # given the control actions    
    
    self.CdA = np.array([0.1, 0.1, 0.4]) #[kg/s] drag constant on linear aerodynamical drag model
    # linear aerodynamics considered self.Sn = np.array([0.02, 0.02, 0.05]) #[m^2] Vector of normal surfaces to main body axes to calculate drag
    # Zb normal surface is greater than othe two  

    self.C_DR = np.array([0.01, 0.01, 0.008]) # [kg m^2/s] coefficients are evaluated with aid of the 
    # Arena and Napoleoni thesis
    

    # integration parameters: The dynamics simulation time is different from the time step size
    # for policy deployment and evaluation: 
    # For the dynamics a dynamics_timeStep is used as 0.01 s 
    # The policy time steps is 0.05 (this step is also the one taken outside)
    self.dynamics_timeStep = 0.01 #[s] time step for Runge Kutta 
    self.timeStep = 0.04 #[s] time step for policy
    self.max_Episode_time_steps = int(4*10.24/self.timeStep) # maximum number of timesteps in an episode (=20s) here counts the policy step
    self.elapsed_time_steps = 0 # time steps elapsed since the beginning of an episode, to be updated each step
    

    # Constants to normalize state and reward
    
    # Setting up a goal to reach affecting reward (it seems to work better with humans 
    # rather than forcing them to stay in their original position, and humans are
    # biological neural networks)
    self.Phi_ref = 0.
    self.Theta_ref = 0.
    
    self.Avg_Thr = self.dTt  # average throttle is updated with pitch and roll angles

  def step(self, action):

      # State-action variables assignment
      State_curr_step = self.state # self.state is initialized as np.array, this temporary variable is used than in next step computation

      controls = np.array([self.Avg_Thr, action[0], action[1], 0.]) ## This variable is used to make possible the separation of actions 
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

      # REWARD evaluation and done condition definition (to be completed)
      p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1 = State_curr_step

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division

      # as obs attitude anr r rate error is gven, evaluated on reference values.
      # error is normalized dividing by the normalization vector stated yet, sign also is given.
      Q = self.state[3:7]
      Q_ref = self.Ang2Quat(env.Phi_ref, env.Theta_ref)
      q0_error = Q[0] - Q_ref[0]
      q1_error = Q[1] - Q_ref[1]
      q2_error = Q[2] - Q_ref[2]
      q3_error = Q[3] - Q_ref[3]
      
      obs = np.array([self.state[0], self.state[1], self.state[2], q0_error, q1_error, q2_error, q3_error])/self.Obs_normalization_vector

      reward = self.getReward()

      done = self.isDone()
    
      info = {"p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1}

      return obs, reward, done, info

  def reset(self):

      """
      Reset state 
      """
      if self.Random_reset:
        p_reset = np_normal(0., 0.175)
        q_reset = np_normal(0., 0.175)
        r_reset = np_normal(0., 0.175)

        phi = np_normal(0., np.pi/18.) #[rad]
        theta = np_normal(0., np.pi/18.) #[rad]
        psi = 0.

        q0_reset = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
        q1_reset = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
        q2_reset = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
        q3_reset = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)

      else:
        p_reset = 0.
        q_reset = 0.
        r_reset = 0.

        q0_reset = 1.
        q1_reset = 0.
        q2_reset = 0.
        q3_reset = 0.     

      self.Phi_ref = np_normal(0., np.pi/18.)
      self.Theta_ref = np_normal(0., np.pi/18.)

      self.state = np.array([p_reset,q_reset,r_reset,q0_reset,q1_reset,q2_reset,q3_reset]) # to initialize the state the object is put in x0=20 and v0=0
      
      self.elapsed_time_steps = 0 # reset for elapsed time steps

      Q = self.state[3:7]
      Q_ref = self.Ang2Quat(env.Phi_ref, env.Theta_ref)
      q0_error = Q[0] - Q_ref[0]
      q1_error = Q[1] - Q_ref[1]
      q2_error = Q[2] - Q_ref[2]
      q3_error = Q[3] - Q_ref[3]
      
      obs = np.array([self.state[0], self.state[1], self.state[2], q0_error, q1_error, q2_error, q3_error])/self.Obs_normalization_vector

      return obs  # produce an observation of the first state (xPosition) 

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      Q = self.state[3:7]
      Q_ref = self.Ang2Quat(env.Phi_ref, env.Theta_ref)
      q0_error = Q[0] - Q_ref[0]
      q1_error = Q[1] - Q_ref[1]
      q2_error = Q[2] - Q_ref[2]
      q3_error = Q[3] - Q_ref[3]
      
      p = self.state[0]
      q = self.state[1]
      r = self.state[2]
      
      rate_weight = 0.4
      qaternion_error_w = 1.
      r_error_weight = 1.

      R = 1. - qaternion_error_w * (abs(q0_error) + abs(q1_error) + abs(q2_error) + abs(q3_error))\
        - r_error_weight * abs(r/50.)\
          - rate_weight * (abs(p/50.) + abs(q/50.))

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
    
      p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1 = self.state

      if abs(p_1)>=50. :

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

## Conversion from quaternions to Phi_theta
  def quat2Att(self, Q):

      """
      Function to convert from quaternion to attitude angles, for simplicity only phi and theta are the output
      input: Q [array_like, quaternion]
      Output: Phi, Theta 
      """

      q0 = Q[0]
      q1 = Q[1]
      q2 = Q[2]
      q3 = Q[3]

      Phi = np.arctan2(2*(q0*q1 + q2*q3), 1-2*(q1**2+q2**2))
      Theta = np.arcsin(2*(q0*q2-q3*q1))

      return Phi, Theta

  def Ang2Quat(self, Phi_ref, Theta_ref):

      """
      Function to convert pi and theta ref to quaternion, psi is assumed to be and stay zero
      input: phiref and theta ref
      output: reference quaternion with psi zero 
      """

      psi = 0.
      phi = Phi_ref
      theta = Theta_ref

      q0 = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
      q1 = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
      q2 = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
      q3 = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)

      Q_ref = np.array([q0, q1, q2, q3])
      
      return Q_ref

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

      Thr = self.dTt * (1 + 0.5 * action)

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

        elif Throttle_array[thr_count] <= 0.08:
          Throttle_array[thr_count] = 0.08

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
      input: Motor throttle [0,1], velocity of the motor relative to the air,
      is computed outside as Vm = Vb + Omega x Rm, Rm in body axes is constant.
      output: Motor Force and reaction torque.
      The model take into account hovering condition bu implements some analytical
      formulas to evaluate the effects of low vertical and horizontal air speed;
      this speed cause flapping motion of the rotor disk and variation in blade
      ange of attack causing variations in vertical thrust and horizontal forces.
      The only torque evaluated is the one caused by the reaction torque of the motor
      while torques caused by horizontal forces and position of the rotor from the CG 
      are computed outside this method to keep generality.
      """
      ## throttle self.nMax_motor [RPS] number of rounds per second for BL motor

      # if self.Process_perturbations: ## adds some randomness to the motor performances
      #   rand_thrust = np_normal(0., 0.1) 
      #   rand_torque = np_normal(0., 0.1)

      # else:
      #   rand_thrust = 0.
      #   rand_torque = 0.

      RPS_squared = Throttle * (self.nMax_motor**2) #[RPS**2] square of actual prop speed
      RPS = np.sqrt(RPS_squared) #[RPS]
      Omega_prop = RPS * 2 * np.pi #[rad/s] 

      Thrust_FP = self.Prop_Kf * RPS_squared #[N] thrust at fix point 
      
      Thrust = Thrust_FP  # + Delta_Thrust

      # The model in which throttle is linear in those formulas, the control is an alias 
      # of F = F_Max * dT so the difference is only on how to consider the command on the 
      # rounds per sec, this operation can be done in flight computer.

      Torque = self.Prop_KqF * Thrust

      a1 = 0.
      b1 = 0.
      # flap coefficients are evaluated without considering the flapping dynamics (which is 
      # way faster than vehicle dynamics) and are decoupled the effects of u and v.
      # remember that due to cross product and displacement between motors and CG, althought 
      # the motor axis are parallel to the body axis, the rotors linear velocity is affected 
      # also by the angular velocity of the multirotor in rigid rotation formula

      return Thrust, Torque, a1, b1 # return scalar thrust and torque

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

      #Performing quaternion normalization
      q0, q1, q2, q3 = State[3:7] # Quaternion

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      q0 = q0/abs_Q
      q1 = q1/abs_Q
      q2 = q2/abs_Q
      q3 = q3/abs_Q

      # Motors section (vectors are evaluated later in this method)
      dT1, dT2, dT3, dT4 = Throttles

      M1_Thrust, M1_Torque, M1_a1, M1_b1 = self.Motor(dT1) # scalar values for M1
      M2_Thrust, M2_Torque, M2_a1, M2_b1 = self.Motor(dT2) # scalar values for M2
      M3_Thrust, M3_Torque, M3_a1, M3_b1 = self.Motor(dT3) # scalar values for M3
      M4_Thrust, M4_Torque, M4_a1, M4_b1 = self.Motor(dT4) # scalar values for M4

      # Evaluation of transformation matrix from Body to NED axes: LEB

      LEB = np.array([[(q0**2 + q1**2 - q2**2 - q3**2), 2.*(q1*q2 - q0*q3), 2.*(q0*q2 + q1*q3)], \
        [2.*(q1*q2 + q0*q3), (q0**2 - q1**2 + q2**2 - q3**2), 2.*(q2*q3 - q0*q1)], \
          [2.*(q1*q3 - q0*q2), 2.*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2)]])

      LBE = np.transpose(LEB) # Evaluate transpose of body to NED---> NED to body
      

      #Motor FORCE vector Evaluation [N] 
      # this vector is evaluated according to values given by motor function
      # and construction specs and taking into account the assumptions and conventions 
      # used in motor method to evaluate flapping coefficients
      F1 = np.array([-M1_Thrust * sin(M1_a1), -M1_Thrust * sin(M1_b1), -M1_Thrust])
      F2 = np.array([-M2_Thrust * sin(M2_a1), -M2_Thrust * sin(M2_b1), -M2_Thrust])
      F3 = np.array([-M3_Thrust * sin(M3_a1), -M3_Thrust * sin(M3_b1), -M3_Thrust])
      F4 = np.array([-M4_Thrust * sin(M4_a1), -M4_Thrust * sin(M4_b1), -M4_Thrust])

      # TORQUES [N m]:
      # as first assumption only the thrust components of the motors combined are considered
      # as torque generator; gyroscopical effects of the props are neglected in this model. 
      # those components are NOT divided by the respective moment of Inertia.
      # Also the aerodynamic drag torque effect is added
      Mtot = np.cross(self.rM1, F1) + np.cross(self.rM2, F2)\
         + np.cross(self.rM3, F3) + np.cross(self.rM4, F4)\
            + np.array([0., 0., (M1_Torque + M3_Torque - M2_Torque - M4_Torque)])\
              + self.dragTorque(Omega)

      # Evaluation of ANGULAR ACCELERATION [rad/s^2] components in body axes
      Omega_dot = ((Mtot - np.cross(Omega, np.dot(self.InTen, Omega))) / self.InVec) + Omega_dot_dist
      
      # Evaluation of QUATERNION derivatives 
      p, q, r = Omega
      q0_dot = 0.5 * (-p*q1 - q*q2 - r*q3)
      q1_dot = 0.5 * (p*q0 + r*q2 - q*q3)
      q2_dot = 0.5 * (q*q0 - r*q1 + p*q3)
      q3_dot = 0.5 * (r*q0 + q*q1 - p*q2)

      Q_dot = np.array([q0_dot, q1_dot, q2_dot, q3_dot])

      stateTime_derivatives= np.concatenate((Omega_dot, Q_dot))
      return stateTime_derivatives