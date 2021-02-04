# Code to generate a Quadcopter Environment to train an RL agent with stable baselines
# this model include a mathematical rapresentation of the quadcopter and it's a first implementation.
# This model is the second version in which all the scalar operations is substitued by vector
# ones with numpy
import numpy as np
import gym
from gym import spaces


class QuadcoptEnvV2(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(QuadcoptEnvV2, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Definition of action space with 4 control actions representing 4 throttle positions in the order: 
    # M1[-1,1], M2, M3, M4. The normalization from -1 to 1 is than converted in the step method
    highActionSpace = np.array([1., 1., 1., 1.])
    lowActionSpace = np.array([-1., -1., -1., -1.])
    self.action_space = spaces.Box(lowActionSpace, highActionSpace, dtype=np.float32)

    # Creation of observation space: an array for maximum values is created using a classical Flight Dynamics states 
    # rapresentation with quaternions (state[min, max][DimensionalUnit]): 
    # u[-50,50][m/s], v[-50,50][m/s], w[-50,50][m/s], p[-20,20][rad/s], q[-20,20][rad/s], r[-20,20][rad/s],...
    #  q0[-1,1], q1[-1,1], q2[-1,1], q3[-1,1], X[-50,50][m], Y[-50,50][m], Z[-100,0][m].
    # To give normalized observations boundaries are fom -1 to 1, the problem scale 
    # is adapted to the physical world in step function.
    # The normalization is performed using limits reported above
    highObsSpace = np.array([1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1 , 1.1])
    lowObsSpace = -highObsSpace
    # lowObsSpace = np.array([-1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1])
    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32) # boundary is set 0.1 over to avoid AssertionError

    # A vector with max value for each state is defined to perform normalization of obs
    # so to have obs vector components between -1,1. The max values are taken acording to 
    # previous comment
    self.Obs_normalization_vector = np.array([50. , 50. , 50. , 20. , 20. , 20. , 1. , 1. , 1. , 1. , 50. , 50. , 100.])
                                        
    self.state = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]) # this variable is used as a memory for the specific instance created
    self.Lx = 0.2   # X body Length (squared configuration)
    self.Ly = 0.2   # Y body Length
    
    # Motors position vectors from G
    self.rM1=np.array([self.Lx/2, self.Ly/2, 0.]) 
    self.rM2=np.array([-self.Lx/2, self.Ly/2, 0.])
    self.rM3=np.array([-self.Lx/2, -self.Ly/2, 0.])
    self.rM4=np.array([self.Lx/2, -self.Ly/2, 0.])    

    self.mass = 1.   # mass is 600 grams can be changed

    ## Inertia tensor is considered dyagonal, null the other components
    self.Ix = 0.06    # rotational Inertia referred to X axis
    self.Iy = 0.06    # rotational Inertia referred to Y axis
    self.Iz = 0.06    # rotational Inertia referred to Z axis
    # Inertia tensor composition
    self.InTen = np.array([[self.Ix, 0., 0.],[0., self.Iy, 0.],[0., 0., self.Iz]])


    self.maxThrust = 9.815 # single engine maximum possible thrust taken as 1kg (9.81 N) (empirical)
    self.Kt = 0.1 ## ATTENTION, proportional constant for assumption about the torque of the motor
    self.CdX = 0.2 # drag coefficent of front section (assumed): orthogonal to Xbody axis
    self.CdY = 0.2 # drag coefficent of lateral section (assumed): orthogonal to Ybody axis
    self.CdZ = 0.2 # drag coefficent of horizontal section (assumed): orthogonal to Zbody axis
    self.SnX = 0.02 # cross sections in m^2 referred to the Cd
    self.SnY = 0.02
    self.SnZ = 0.05 # Z normal surface is bigger than other ones    

    # atmosphere definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 # kg/m^3 Standard day at 0 m ASL
    self.g0 = 9.815  # m/s^2 gravity acceleration

    # integration parameters: constant step of 0.1 s
    self.timeStep = 0.1

    # Constants to normalize state and reward
    self.VmaxSquared = 2500 # Squared by deafult to save some computation


  def step(self, action):

      # State-action variables assignment
      State_curr_step = self.state # self.state is initialized as np.array, this temporary variable is used than in next step computation 
      a1, a2, a3, a4 = action # Throttle of the 4 motors (position of the motor is given in the drawings)

      dT1 = 0.5*(a1+1)
      dT2 = 0.5*(a2+1)
      dT3 = 0.5*(a3+1)
      dT4 = 0.5*(a4+1)

      h = self.timeStep 

      # Integration of the equation of motion with Runge-Kutta 4 order method
      ## The state derivatives funcion xVec_dot = fvec(x,u) is implemented in a separate function
      # State_dot = self.eqnsOfMotion(State_curr_step, dT1, dT2, dT3, dT4)
      
      k1vec = h * self.eqnsOfMotion(State_curr_step, dT1, dT2, dT3, dT4)

      # Evaluation of constant K2 from equations of state evaluated in state+K1/2
      # State_dot2 = self.eqnsOfMotion(np.add(State_curr_step, 0.5*k1vec), dT1, dT2, dT3, dT4)
      
      k2vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k1vec), dT1, dT2, dT3, dT4)

      # Evaluation of constants K3 from state +k2/2
      # State_dot3 = self.eqnsOfMotion(np.add(State_curr_step, 0.5 * k2vec), dT1, dT2, dT3, dT4)

      k3vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5 * k2vec), dT1, dT2, dT3, dT4)
      
      #Evaluation of K4 from state+K3
      # State_dot4 = self.eqnsOfMotion(np.add(State_curr_step, k3vec), dT1, dT2, dT3, dT4)

      k4vec = h * self.eqnsOfMotion(np.add(State_curr_step, k3vec), dT1, dT2, dT3, dT4)

      # Final step of integration: each update for the state is evaluated from the Ks 
      State_next_step = State_curr_step + (k1vec/6) + (k2vec/3) + (k3vec/3) + (k4vec/6)


      # self.state variable assignment with next step values (step n+1 is indicated with _1)
      self.state = State_next_step

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division
      obs = self.state / self.Obs_normalization_vector

      # REWARD evaluation and done condition definition (to be completed)
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = State_next_step

      reward = q0_1 - ((u_1**2)/ self.VmaxSquared) - ((v_1**2)/ self.VmaxSquared) - ((w_1**2)/ self.VmaxSquared)
      done = bool(Z_1>=0)
      info = {"u": u_1, "v": v_1, "w": w_1, "p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1, "X": X_1, "Y": Y_1, "Z": Z_1}

      return obs, reward, done, info

  def reset(self):
      
      self.state = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,-50.]) # to initialize the state the object is put in x0=20 and v0=0
      
      obs = self.state / self.Obs_normalization_vector
      return obs  # produce an observation of the first state (xPosition) 

## In this sections are defined functions to evaluate forces and derivatives to make the step function easy to read

  def Drag(self, V, Sn, Cd):
      """
      This function return an Aerodynamical drag given velocity cd and normal Surface
      """
      # Evaluation of the AERODYNAMICAL DRAG: this force is modeled as 3 scalar values
      # calculated with the formula D=0.5 S Cd rho V^2 where V is the single component 
      # of velocity in body axes and S and Cd are referred to the surface normal to the 
      # selected V component: E.G. to evaluate X component of drag V = u and S and Cd are 
      # those referred to the front section.
      drag = - 0.5 * self.rho * V * abs(V) * Sn * Cd
      return drag

  def eqnsOfMotion(self, State, dT1, dT2, dT3, dT4):
      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

      u, v, w, p, q, r, q0, q1, q2, q3 = State[0:10] 
      

      #THRUST Evaluation
      T1 = - dT1 * self.maxThrust
      T2 = - dT2 * self.maxThrust
      T3 = - dT3 * self.maxThrust
      T4 = - dT4 * self.maxThrust

      # TORQUES: as first assumption only the thrust components of the motors combined are considered
      # as torque generator; gyroscopical effects of the props are neglected in this model. 
      # those components are NOT divided by the respective moment of Inertia
      MtotX = (T4 + T3 - T2 - T1) * self.Ly/2 
      MtotY = (T4 + T1 - T3 - T2) * self.Lx/2
      MtotZ = (T1 + T3 - T2 - T4) * self.Kt

      # WEIGHT in body axes
      Wx = self.mass * self.g0 * 2.* (q1*q3 - q0*q2)
      Wy = self.mass * self.g0 * 2. * (q0*q1 - q2*q3)
      Wz = self.mass * self.g0 * ((q0**2) - (q1**2) - (q2**2) + (q3**2))

      # TOTAL FORCES in body axes
      FtotX_m = (self.Drag(u, self.SnX, self.CdX) + Wx) / self.mass
      FtotY_m = (self.Drag(v, self.SnY, self.CdY) + Wy) / self.mass
      Ftotz_m = (self.Drag(w, self.SnZ, self.CdZ) + Wz + T1 + T2 + T3 + T4) / self.mass

      # Evaluation of LINEAR ACCELERATION components: first total forces are calculated, than divided by mass
      # than the equations are implemented 
      u_dot = r*v - q*w + FtotX_m
      v_dot = p*w - r*u + FtotY_m
      w_dot = q*u - p*v + Ftotz_m

      # Evaluation of ANGULAR ACCELERATION components in body axes
      p_dot = (((self.Iy-self.Iz)*q*r) + MtotX) / self.Ix
      q_dot = (((self.Iz-self.Ix)*p*r) + MtotY) / self.Iy
      r_dot = (((self.Ix-self.Iz)*q*p) + MtotZ) / self.Iz
      
      # Evaluation of the cinematics acceleration 
      # The matrix LEB is written in the equivalent from quaternions components
      X_dot = (q0**2 + q1**2 - q2**2 - q3**2) * u + 2.*(q1*q2 - q0*q3) * v + 2.*(q0*q2 + q1*q3) * w
      Y_dot = 2.*(q1*q2 + q0*q3) * u + (q0**2 - q1**2 + q2**2 - q3**2) * v + 2.*(q2*q3 - q0*q1) * w
      Z_dot = 2.*(q1*q3 - q0*q2) * u + 2.*(q0*q1 + q2*q3) * v + (q0**2 - q1**2 - q2**2 + q3**2) * w

      # Evaluation of QUATERNION derivatives
      q0_dot = 0.5 * (-p*q1 - q*q2 - r*q3)
      q1_dot = 0.5 * (p*q0 + r*q2 - q*q3)
      q2_dot = 0.5 * (q*q0 - r*q1 + p*q3)
      q3_dot = 0.5 * (r*q0 + q*q1 - p*q2)

      stateTime_derivatives= np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, q0_dot, q1_dot, q2_dot, q3_dot, X_dot, Y_dot, Z_dot])
      return stateTime_derivatives