# Code to generate a Quadcopter Environment to train an RL agent with stable baselines
# this model include a mathematical rapresentation of the quadcopter.
# This model is the third version in which all the scalar operations is substitued by vector
# ones with numpy
import numpy as np
import gym
from gym import spaces

class QuadcoptEnvV3(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(QuadcoptEnvV3, self).__init__()


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
    self.Lx = 0.2   #[m] X body Length (squared configuration)
    self.Ly = 0.2   #[m] Y body Length
    
    # Motors position vectors from G
    self.rM1=np.array([self.Lx/2, self.Ly/2, 0.]) 
    self.rM2=np.array([-self.Lx/2, self.Ly/2, 0.])
    self.rM3=np.array([-self.Lx/2, -self.Ly/2, 0.])
    self.rM4=np.array([self.Lx/2, -self.Ly/2, 0.])    

    self.mass = 1.   #[kg] mass is 600 grams can be changed

    ## Inertia tensor is considered dyagonal, null the other components
    self.Ix = 0.06    #[kg m^2] rotational Inertia referred to X axis
    self.Iy = 0.06    #[kg m^2] rotational Inertia referred to Y axis
    self.Iz = 0.06    #[kg m^2] rotational Inertia referred to Z axis
    # Inertia tensor composition
    self.InTen = np.array([[self.Ix, 0., 0.],[0., self.Iy, 0.],[0., 0., self.Iz]])

    # Inertia vector: vector with 3 principal inertia useful in evaluating the Omega_dot
    self.InVec = np.diag(self.InTen)

    # Throttle constants
    self.dTt = .25 # trim throttle to hover
    self.d2 = 0.65 # Assumed value for first constant in action to throttle mapping
    self.d1 = 1 - self.d2 - self.dTt # second constant for maping (see notebook)
    self.s2 = self.d2 - 1 + 2*self.dTt # first constant for left part
    self.s1 = self.dTt - self.s2

    self.maxThrust = 9.815 #[N] single engine maximum possible thrust taken as 1kg (9.81 N) (empirical)
    self.Kt = 0.9815 ##[N m] ATTENTION, proportional constant for assumption about the torque of the motors
    # this constant is multiplied by the throttle of the motor, this model of torque has to be validated.
    
    self.Cd = np.array([0.2, 0.2, 0.2]) # Vector of drag constants for three main body axes normal surfaces
    self.Sn = np.array([0.02, 0.02, 0.05]) #[m^2] Vector of normal surfaces to main body axes to calculate drag
    # Zb normal surface is greater than othe two  

    # atmosphere definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 #[kg/m^3] Standard day at 0 m ASL
    self.g0 = 9.815  #[m/s^2] gravity acceleration

    # integration parameters: constant step of 0.1 [s]
    self.timeStep = 0.01

    # Constants to normalize state and reward
    self.VmaxSquared = 2500 #[(m/s)^2] Squared by deafult to save some computation


  def step(self, action):

      # State-action variables assignment
      State_curr_step = self.state # self.state is initialized as np.array, this temporary variable is used than in next step computation 
      
      Throttle = self.act2ThrotMap(action) # composition of throttle vector basing on actions

      h = self.timeStep 

      # Integration of the equation of motion with Runge-Kutta 4 order method
      ## The state derivatives funcion xVec_dot = fvec(x,u) is implemented in a separate function
      k1vec = h * self.eqnsOfMotion(State_curr_step, Throttle)

      # Evaluation of K2 from state+K1/2
      k2vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5*k1vec), Throttle)

      # Evaluation of K3 from state+k2/2
      k3vec = h * self.eqnsOfMotion(np.add(State_curr_step, 0.5 * k2vec), Throttle)
      
      #E valuation of K4 from state+K3
      k4vec = h * self.eqnsOfMotion(np.add(State_curr_step, k3vec), Throttle)

      # Final step of integration 
      State_next_step = State_curr_step + (k1vec/6) + (k2vec/3) + (k3vec/3) + (k4vec/6)


      # self.state variable assignment with next step values (step n+1)
      self.state = State_next_step

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division
      obs = self.state / self.Obs_normalization_vector

      # REWARD evaluation and done condition definition (to be completed)
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = State_next_step

      reward = q0_1 - ((u_1**2)/ self.VmaxSquared) - ((v_1**2)/ self.VmaxSquared) - ((w_1**2)/ self.VmaxSquared)
      
      done = self.isDone()
    
      info = {"u": u_1, "v": v_1, "w": w_1, "p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1, "X": X_1, "Y": Y_1, "Z": Z_1}

      return obs, reward, done, info

  def reset(self):

      """
      Reset state 
      """
      
      self.state = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,-50.]) # to initialize the state the object is put in x0=20 and v0=0
      
      obs = self.state / self.Obs_normalization_vector
      return obs  # produce an observation of the first state (xPosition) 

  def act2ThrotMap(self, actions):

      """ 
      Function that maps actions into throttle values with constraint reported on the notebook.
      Mapping follows a linear and cubic function definesd by constant d1 d2 (right part of the map)
      and s1 s2 (for left part). Constant are constrained by [0, 1] output and equal derivative in 
      0-eps, 0+eps.
      """

      Thr = np.zeros(4)
      i = 0

      for a in actions:

        if a<=0:
          Thr[i]= self.dTt + (self.s1*a) + (self.s2*(a**3))

        else:
          Thr[i] = self.dTt + (self.d1*a) + (self.d2*(a**3))

        i += 1
      
      return Thr

  def isDone(self):

      """
      return a bool condition True if any state falls outbound normalization vector
      components assumption. prints some indications on which state caused done.
      Dimensional unit reported in the comment in __init__()
      """
    
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = self.state

      abs(X_1)>=50. or abs(X_1)>=50.

      if Z_1>=0. or Z_1<=-100. : 

        done = True
        print("Z outbound---> ", Z_1)

      elif abs(u_1)>=50. :

        done = True
        print("u outbound---> ", u_1)

      elif abs(v_1)>=50. :

        done = True
        print("v outbound---> ", v_1)

      elif abs(w_1)>=50. :

        done = True
        print("w outbound---> ", w_1)

      elif abs(p_1)>=20. :

        done = True
        print("p outbound---> ", p_1)

      elif abs(q_1)>=20. :

        done = True
        print("q outbound---> ", q_1)

      elif abs(r_1)>=20. :

        done = True
        print("r outbound---> ", r_1)

      elif abs(X_1)>=50. :

        done = True
        print("X outbound---> ", X_1)

      elif abs(Y_1)>=50. :

        done = True
        print("Y outbound---> ", Y_1)

      elif abs(q0_1)>=1.0000000001 or abs(q1_1)>=1.0000000001 or abs(q2_1)>=1.0000000001 or abs(q3_1)>=1.0000000001 :

        done = True
        print("Quaternion outbound...") 
        print("q0 = ", q0_1)
        print("q1 = ", q1_1)
        print("q2 = ", q2_1)
        print("q3 = ", q3_1)

      else :

        done = False

      return done

## In this sections are defined functions to evaluate forces and derivatives to make the step function easy to read

  def Drag(self, V):
    
      """
      This function return an Aerodynamical drag given velocity cd and normal Surface
      """
      # Evaluation of the AERODYNAMICAL DRAG: this force is modeled as 3 scalar values
      # calculated with the formula D=0.5 S Cd rho V^2 where V is the single component 
      # of velocity in body axes and S and Cd are referred to the surface normal to the 
      # selected V component: E.G. to evaluate X component of drag V = u and S and Cd are 
      # those referred to the front section.
      # Evaluation performed in vector form

      drag = - 0.5 * self.rho * V * abs(V) * self.Sn * self.Cd  #[N]

      return drag

  def eqnsOfMotion(self, State, Throttle):

      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

      Vb = State[0:3] # Subvector CG velocity [m/s]
      Omega = State[3:6] # Subvector angular velocity [rad/s]
      q0, q1, q2, q3 = State[6:10] # Quaternion

      dT1, dT2, dT3, dT4 = Throttle

      # Evaluation of transformation matrix from Body to NED axes: LEB

      LEB = np.array([[(q0**2 + q1**2 - q2**2 - q3**2), 2.*(q1*q2 - q0*q3), 2.*(q0*q2 + q1*q3)], \
        [2.*(q1*q2 + q0*q3), (q0**2 - q1**2 + q2**2 - q3**2), 2.*(q2*q3 - q0*q1)], \
          [2.*(q1*q3 - q0*q2), 2.*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2)]])

      LBE = np.transpose(LEB) # Evaluate transpose of body to NED---> NED to body
      

      #THRUST Evaluation [N]
      # is evaluated negative because thrust is oriented in the negative verse of Zb
      # according to how props generate the thrust.
      T1 = np.array([0, 0, - dT1 * self.maxThrust])
      T2 = np.array([0, 0, - dT2 * self.maxThrust])
      T3 = np.array([0, 0, - dT3 * self.maxThrust])
      T4 = np.array([0, 0, - dT4 * self.maxThrust])

      # TORQUES [N m]:
      # as first assumption only the thrust components of the motors combined are considered
      # as torque generator; gyroscopical effects of the props are neglected in this model. 
      # those components are NOT divided by the respective moment of Inertia
      Mtot = np.cross(self.rM1, T1) + np.cross(self.rM2, T2)\
         + np.cross(self.rM3, T3) + np.cross(self.rM4, T4)\
            + np.array([0., 0., (dT1 + dT3 - dT2 - dT4) * self.Kt])

      # WEIGHT [N] in body axes
      Wned = np.array([0, 0, self.mass * self.g0]) # weight vector in NED axes
      WB = np.dot(LBE, Wned) # weight in body axes

      # DRAG [N] in body axes
      DB = self.Drag(Vb)
      
      # TOTAL FORCES in body axes divided by mass [N/kg = m/s^2]
      Ftot_m = (DB + WB + T1 + T2 + T3 + T4) / self.mass 

      # Evaluation of LINEAR ACCELERATION components [m/s^2]
      Vb_dot = - np.cross(Omega, Vb) + Ftot_m

      # Evaluation of ANGULAR ACCELERATION [rad/s^2] components in body axes
      Omega_dot = (Mtot - np.cross(Omega, np.dot(self.InTen, Omega))) / self.InVec
      
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