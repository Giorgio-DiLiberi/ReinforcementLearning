# Code to implement an environment of a 8inch props quadcopter in "plus configuration"
# like the hummigbird by Asc. Tech. This code tries to stay more generale as possible
# so it will be good to simulate any quadcopter with 8 inches props in + config
# because it tries to simulate only the physics of the model without any stack of 
# control systems or stability augmentation systems. 

# This model is similar to the Vref one but here the goal of the policy is to follow a 
# reference on the velocity error and also to rotate the Xb axis towards the direction of 
# motion to maintain the heading = Psi. This model is in test and does not work properly for 
# now when Psi overcome +-180°. This model can implements both a Vectorial control receiving 
# V_NED ref components and a position control which evaluates V_NED references proportionally to position error. 
import numpy as np
from numpy.random import normal as np_normal
from numpy import cos as cos
from numpy import sin as sin
import gym
from gym import spaces

class Hummingbird_6DOF(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, Random_reset = False, Process_perturbations = False, Position_reference = True):
    super(Hummingbird_6DOF, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # the controls re given as pseudo commands for average throttle, aileron, elevator and rudder
    # so torques and thrust are controlled.
    # The motor model tries to be as accurate as possible 
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
    # order is V_N_Err, V_E_Err, V_D_Err, v, p, q, r, q0, q1, q2, q3
    highObsSpace = np.array([1.1 , 1.1, 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1, 1.1, 1.1])
    lowObsSpace = -highObsSpace
    

    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32) # boundary is set 0.1 over to avoid AssertionError

    # A vector with max value for each state is defined to perform normalization of obs
    # so to have obs vector components between -1,1. The max values are taken acording to 
    # previous comment
    self.Obs_normalization_vector = np.array([20., 20., 20., 10., 2*np.pi, 50., 50., 50., 1., 1., 1., 1.]) # normalization constants
    # Random funcs
    self.Random_reset = Random_reset # true to have random reset
    self.Process_perturbations = Process_perturbations # to have random accelerations due to wind
    self.Position_reference = Position_reference
    

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
    # self.Motor_Kv = 900. # [RPM/V] known for te specific motor
    # self.V_batt_nom = 11.1 # [V] nominal battery voltage 
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
    self.Command_scaling_factor = 0.3 # Coefficient to scale commands when evaluating throttles of motors
    # given the control actions    
    
    self.CdA = np.array([0., 0., 0.]) #[kg/s] drag constant on linear aerodynamical drag model
    # linear aerodynamics considered self.Sn = np.array([0.02, 0.02, 0.05]) #[m^2] Vector of normal surfaces to main body axes to calculate drag
    # Zb normal surface is greater than othe two  

    self.C_DR = np.array([0., 0., 0.]) # [kg m^2/s] coefficients are evaluated with aid of the 
    # Arena and Napoleoni thesis
    

    # integration parameters: The dynamics simulation time is different from the time step size
    # for policy deployment and evaluation: 
    # For the dynamics a dynamics_timeStep is used as 0.01 s 
    # The policy time steps is 0.05 (this step is also the one taken outside)
    self.dynamics_timeStep = 0.01 #[s] time step for Runge Kutta 
    self.timeStep = 0.04 #[s] time step for policy
    self.max_Episode_time_steps = int(8*10.24/self.timeStep) # maximum number of timesteps in an episode (=20s) here counts the policy step
    self.elapsed_time_steps = 0 # time steps elapsed since the beginning of an episode, to be updated each step
    

    # Constants to normalize state and reward
    
    # Setting up a goal to reach affecting reward (it seems to work better with humans 
    # rather than forcing them to stay in their original position, and humans are
    # biological neural networks)
    self.X_ref = 0.
    self.Y_ref = 0.
    self.Z_ref = -2.

    # references on NED velocity are evaluated proportionally to posizion errors
    self.VNord_ref = 0. #[m] 
    self.VEst_ref = 0. #[m]
    self.VDown_ref = 0. #[m]

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

      # evaluation of NED velocity references proportionally to position errors if Position Reference ==True
      if self.Position_reference:

        self.VNord_ref = 0.4 * (self.X_ref - self.state[10])

        if abs(self.VNord_ref)<=0.000001:
          self.VNord_ref = 0.

        self.VEst_ref = 0.4 * (self.Y_ref - self.state[11])

        if abs(self.VEst_ref)<=0.000001:
          self.VEst_ref = 0.

        self.VDown_ref = 0.4 * (self.Z_ref - self.state[12])

        #if Position_reference == False the user must provide references for NED velocity manually, default values are 0

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

      VNord_error = V_NED[0] - self.VNord_ref
      VEst_error = V_NED[1] - self.VEst_ref
      VDown_error = V_NED[2] - self.VDown_ref

      Psi = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))
      Psi_ref = np.arctan2(self.VEst_ref, self.VNord_ref)

      if self.VNord_ref<=0.:
        Psi_ref = 0.

      Psi_err = Psi - Psi_ref        

      obs_state = np.concatenate(([VNord_error, VEst_error, VDown_error, self.state[1], Psi_err], self.state[3:10]))
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

        phi = np_normal(0., 0.44) #[rad]
        theta = np_normal(0., 0.44) #[rad]
        psi = np_normal(0., 0.175) #[rad]

        q0_reset = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
        q1_reset = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
        q2_reset = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
        q3_reset = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)

        self.VNord_ref = np_normal(5., 2.) #[m/s]
        self.VEst_ref = np_normal(5., 2.) #[m/s]
        self.VDown_ref = np_normal(-5., 2.) #[m/s]

      else:
        w_reset = 0. #[m/s]
        Z_reset = -2. #[m]
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

        self.VNord_ref = 3. #[m/s]
        self.VEst_ref = 4. #[m/s]
        self.VDown_ref = -3. #[m/s]

      self.state = np.array([u_reset,v_reset,w_reset,p_reset,q_reset,r_reset,q0_reset,q1_reset,q2_reset,q3_reset,X_reset,Y_reset,Z_reset]) # to initialize the state the object is put in x0=20 and v0=0
      
      self.elapsed_time_steps = 0 # reset for elapsed time steps

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

      VNord_error = V_NED[0] - self.VNord_ref
      VEst_error = V_NED[1] - self.VEst_ref
      VDown_error = V_NED[2] - self.VDown_ref

      Psi = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))
      Psi_ref = np.arctan2(self.VEst_ref, self.VNord_ref)

      Psi_err = Psi - Psi_ref

      obs_state = np.concatenate(([VNord_error, VEst_error, VDown_error, self.state[1], Psi_err], self.state[3:10]))
      obs = obs_state / self.Obs_normalization_vector

      return obs  # produce an observation of the first state (xPosition) 

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      q0, q1, q2, q3 = self.state[6:10] # Quaternion
      Vb = self.state[0:3]
      p, q, r = self.state[3:6]
      v = self.state[1]

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      q0 = q0/abs_Q
      q1 = q1/abs_Q
      q2 = q2/abs_Q
      q3 = q3/abs_Q

      LEB = np.array([[(q0**2 + q1**2 - q2**2 - q3**2), 2.*(q1*q2 - q0*q3), 2.*(q0*q2 + q1*q3)], \
        [2.*(q1*q2 + q0*q3), (q0**2 - q1**2 + q2**2 - q3**2), 2.*(q2*q3 - q0*q1)], \
          [2.*(q1*q3 - q0*q2), 2.*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2)]])

      V_NED = np.dot(LEB, Vb)

      VNord_error = V_NED[0] - self.VNord_ref
      VEst_error = V_NED[1] - self.VEst_ref
      VDown_error = V_NED[2] - self.VDown_ref

      V_error_Weight = 0.9
      drift_weight = 0.8
      rate_weight = 0.6

      R = 1. - V_error_Weight * (abs(VNord_error/20.) + abs(VEst_error/20.) + abs(VDown_error/20.))\
        - drift_weight * abs(v/30.) - rate_weight * (abs(p/50.) + abs(q/50.) + abs(r/50.))

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

        elif Throttle_array[thr_count] <= 0.00001:
          Throttle_array[thr_count] = 0.0001

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

  def Motor(self, Throttle, Vm):

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

      um, vm, wm = Vm # [m/s] velocity components

      RPS_squared = Throttle * (self.nMax_motor**2) #[RPS**2] square of actual prop speed
      RPS = np.sqrt(RPS_squared) #[RPS]
      Omega_prop = RPS * 2 * np.pi #[rad/s] 

      ## setting up a minimum value for tip velocity to avoid division by zero in evaluation of advance ratios
      # 450. [rad/s] is the hovering prop angular velocity for tis kind of drone
      if Omega_prop<=450.:
        U_tip = 450. * self.D_prop / 2 #[m/s]

      else:
        U_tip = Omega_prop * self.D_prop / 2 #[m/s]
      
      Thrust_FP = self.Prop_Kf * RPS_squared #[N] thrust at fix point 
      
      # evaluation of total induced velocity to calculate the differential thrust due to 
      # vertical velocity. the formula is the standard one for normal condition. 
      # no distribution of induced velocity on disk is considered.
      vi = (wm/2) + np.sqrt(((0.5*wm)**2) + (self.vh**2))

      # Evaluation of thrust as fix point thrust - delta which depends on effects due to vertical vel
      # 2*pi is considered as dCl/dAlpha for prop profile dT=1/4*rho*a*b*c*
      Delta_Thrust = self.rho * np.pi * self.prop_mean_chord * (2*np.pi*RPS) * ((self.D_prop**2)/4) * (- vi + wm + self.vh)
      # positive w velocity means more thrust so positive contribution
      Thrust = Thrust_FP + Delta_Thrust

      # The model in which throttle is linear in those formulas, the control is an alias 
      # of F = F_Max * dT so the difference is only on how to consider the command on the 
      # rounds per sec, this operation can be done in flight computer.

      Torque = self.Prop_KqF * Thrust

      # Evaluation of rotor disk flapping coefficients: a1 represents the x_relative coeff.
      # while b1 is the y rel coeff.

      mi_x = um / U_tip # advance ratios relative to u and v velocities
      mi_y = vm / U_tip

      lambda_i = (wm - vi) / U_tip # induced velocity coefficient

      a1 = 2 * mi_x * (((4/3)*self.prop_Theta0) + lambda_i)/(1 - ((mi_x**2)/2))
      b1 = 2 * mi_y * (((4/3)*self.prop_Theta0) + lambda_i)/(1 - ((mi_y**2)/2))
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
      q0, q1, q2, q3 = State[6:10] # Quaternion

      abs_Q = (q0**2 + q1**2 + q2**2 + q3**2)

      q0 = q0/abs_Q
      q1 = q1/abs_Q
      q2 = q2/abs_Q
      q3 = q3/abs_Q

      # Motors section (vectors are evaluated later in this method)
      dT1, dT2, dT3, dT4 = Throttles

      # Evaluation of body vel for the single motors
      Vm1 = Vb + np.cross(Omega, self.rM1)
      Vm2 = Vb + np.cross(Omega, self.rM2)
      Vm3 = Vb + np.cross(Omega, self.rM3)
      Vm4 = Vb + np.cross(Omega, self.rM4)

      M1_Thrust, M1_Torque, M1_a1, M1_b1 = self.Motor(dT1, Vm1) # scalar values for M1
      M2_Thrust, M2_Torque, M2_a1, M2_b1 = self.Motor(dT2, Vm2) # scalar values for M2
      M3_Thrust, M3_Torque, M3_a1, M3_b1 = self.Motor(dT3, Vm3) # scalar values for M3
      M4_Thrust, M4_Torque, M4_a1, M4_b1 = self.Motor(dT4, Vm4) # scalar values for M4

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

      # WEIGHT [N] in body axes
      WB = np.dot(LBE, self.Wned) # weight in body axes

      # DRAG [N] in body axes
      DB = self.Drag(Vb)
      
      # TOTAL FORCES in body axes divided by mass [N/kg = m/s^2]
      Ftot_m = (DB + WB + F1 + F2 + F3 + F4) / self.mass 

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