# Code to generate a Quadcopter Environment to train an RL agent with stable baselines
# this model include a mathematical rapresentation of the quadcopter and it's a first implementation.
# This model 
import numpy as np
import gym
from gym import spaces


class QuadcoptEnvV1(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(QuadcoptEnvV1, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Definition of action space with 4 control actions representing 4 throttle positions in the order: 
    # M1[0,1], M2, M3, M4
    highActionSpace = np.array([1., 1., 1., 1.])
    lowActionSpace = np.array([-1., -1., -1., -1.])
    self.action_space = spaces.Box(lowActionSpace, highActionSpace, dtype=np.float32)

    # Creation of observation space: an array for maximum values is created using a classical Flight Dynamics states 
    # rapresentation with quaternions (state[min, max][DimensionalUnit]): 
    # u[-50,50][m/s], v[-50,50][m/s], w[-50,50][m/s], p[-20,20][rad/s], q[-20,20][rad/s], r[-20,20][rad/s],...
    #  q0[0,1], q1[0,1], q2[0,1], q3[0,1], X[-50,50][m], Y[-50,50][m], Z[-100,0][m].
    highObsSpace = np.array([50. , 50. , 50. , 20. , 20. , 20. , 2. , 2. , 2. , 2. , 50. , 50. , 50.])
    lowObsSpace = np.array([-50. , -50. , -50. , -20. , -20. , -20. , -2. , -2. , -2. , -2. , -50. , -50. , -100.])
    self.observation_space = spaces.Box(lowObsSpace, highObsSpace, dtype=np.float32)
                                        
    self.state = [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.] # this variable is used as a memory for the specific instance created
    self.Lx = 0.2   # X body Length (squared configuration)
    self.Ly = 0.2   # Y body Length
    

    self.mass = 1.   # mass is 600 grams can be changed
    ## Inertia tensor is considered dyagonal, null the ther components
    self.Ix = 0.06    # rotational Inertia referred to X axis
    self.Iy = 0.06    # rotational Inertia referred to Y axis
    self.Iz = 0.06    # rotational Inertia referred to Z axis
    self.maxThrust = 30.   # single engine maximum possible thrust taken as almost 3kg (30 N)
    self.Kt = 0.1 ## ATTENTION, proportional constant for assumption about the torque of the motor
    self.CdX = 0.2 # drag coefficent of front section (assumed): orthogonal to Xbody axis
    self.CdY = 0.2 # drag coefficent of lateral section (assumed): orthogonal to Ybody axis
    self.CdZ = 0.2 # drag coefficent of horizontal section (assumed): orthogonal to Zbody axis
    self.SnX = 0.02 # cross sections in m^2 referred to the Cd
    self.SnY = 0.02
    self.SnZ = 0.02

    

    # atmosphere definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 # kg/m^3 Standard day at 0 m ASL
    self.g0 = 9.815  # m/s^2 gravity acceleration

    # integration parameters: constant step of 0.001 s
    self.timeStep = 0.1 

    # Constants to normalize state and reward
    self.VmaxSquared = 7500 # Squared by deafult to save some computation


  def step(self, action):

      # State-action variables assignment
      u, v, w, p, q, r, q0, q1, q2, q3, X, Y, Z = self.state 
      a1, a2, a3, a4 = action # Throttle of the 4 motors (position of the motor is given in the drawings)

      dT1 = 0.5*(a1+1)
      dT2 = 0.5*(a2+1)
      dT3 = 0.5*(a3+1)
      dT4 = 0.5*(a4+1)

      ## The state derivatives funcion xVec_dot = f(x,u) is implemented in a separate function
      State_dot = self.eqnsOfMotion(u, v, w, p, q, r, q0, q1, q2, q3, X, Y, Z, dT1, dT2, dT3, dT4)

      u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, q0_dot, q1_dot, q2_dot, q3_dot, X_dot, Y_dot, Z_dot = State_dot

      # Integration of the equation of motion with Runge-Kutta 4 order method
      # the constant for each are written for each state variable as, for example for variable "u",
      # Ku1, Ku2, Ku3, Ku4
      h = self.timeStep 
      # K1 computation for all 13 states
      Ku1 = h * u_dot
      Kv1 = h * v_dot
      Kw1 = h * w_dot
      Kp1 = h * p_dot
      Kq1 = h * q_dot
      Kr1 = h * r_dot
      Kq0_1 = h * q0_dot
      Kq1_1 = h * q1_dot
      Kq2_1 = h * q2_dot
      Kq3_1 = h * q3_dot
      KX1 = h * X_dot
      KY1 = h * Y_dot
      KZ1 = h * Z_dot

      # Evaluation of constant K2 from equations of state evaluated in state+K1/2
      State_dot2 = self.eqnsOfMotion(u + 0.5*Ku1, v + 0.5*Kv1, w + 0.5*Kw1, p + 0.5*Kp1, q + 0.5*Kq1, r + 0.5*Kr1, q0 + 0.5*Kq0_1, q1 + 0.5*Kq1_1, q2 + 0.5*Kq2_1, q3 + 0.5*Kq3_1, X + 0.5*KX1, Y + 0.5*KY1, Z + 0.5*KZ1, dT1, dT2, dT3, dT4)

      u_dot2, v_dot2, w_dot2, p_dot2, q_dot2, r_dot2, q0_dot2, q1_dot2, q2_dot2, q3_dot2, X_dot2, Y_dot2, Z_dot2 = State_dot2
      
      Ku2 = h * u_dot2
      Kv2 = h * v_dot2
      Kw2 = h * w_dot2
      Kp2 = h * p_dot2
      Kq2 = h * q_dot2
      Kr2 = h * r_dot2
      Kq0_2 = h * q0_dot2
      Kq1_2 = h * q1_dot2
      Kq2_2 = h * q2_dot2
      Kq3_2 = h * q3_dot2
      KX2 = h * X_dot2
      KY2 = h * Y_dot2
      KZ2 = h * Z_dot2

      # Evaluation of constants K3 from state +k2/2
      State_dot3 = self.eqnsOfMotion(u + 0.5*Ku2, v + 0.5*Kv2, w + 0.5*Kw2, p + 0.5*Kp2, q + 0.5*Kq2, r + 0.5*Kr2, q0 + 0.5*Kq0_2, q1 + 0.5*Kq1_2, q2 + 0.5*Kq2_2, q3 + 0.5*Kq3_2, X + 0.5*KX2, Y + 0.5*KY2, Z + 0.5*KZ2, dT1, dT2, dT3, dT4)

      u_dot3, v_dot3, w_dot3, p_dot3, q_dot3, r_dot3, q0_dot3, q1_dot3, q2_dot3, q3_dot3, X_dot3, Y_dot3, Z_dot3 = State_dot3
      
      Ku3 = h * u_dot3
      Kv3 = h * v_dot3
      Kw3 = h * w_dot3
      Kp3 = h * p_dot3
      Kq3 = h * q_dot3
      Kr3 = h * r_dot3
      Kq0_3 = h * q0_dot3
      Kq1_3 = h * q1_dot3
      Kq2_3 = h * q2_dot3
      Kq3_3 = h * q3_dot3
      KX3 = h * X_dot3
      KY3 = h * Y_dot3
      KZ3 = h * Z_dot3

      #Evaluation of K4 from state+K3
      State_dot4 = self.eqnsOfMotion(u + Ku3, v + Kv3, w + Kw3, p + Kp3, q + Kq3, r + Kr3, q0 + Kq0_3, q1 + Kq1_3, q2 + Kq2_3, q3 + Kq3_3, X + KX3, Y + KY3, Z + KZ3, dT1, dT2, dT3, dT4)

      u_dot4, v_dot4, w_dot4, p_dot4, q_dot4, r_dot4, q0_dot4, q1_dot4, q2_dot4, q3_dot4, X_dot4, Y_dot4, Z_dot4 = State_dot4
      
      Ku4 = h * u_dot4
      Kv4 = h * v_dot4
      Kw4 = h * w_dot4
      Kp4 = h * p_dot4
      Kq4 = h * q_dot4
      Kr4 = h * r_dot4
      Kq0_4 = h * q0_dot4
      Kq1_4 = h * q1_dot4
      Kq2_4 = h * q2_dot4
      Kq3_4 = h * q3_dot4
      KX4 = h * X_dot4
      KY4 = h * Y_dot4
      KZ4 = h * Z_dot4

      # Final step of integration: each update for the state is evaluated from the Ks 
      u_1 = u + (Ku1/6) + (Ku2/3) + (Ku3/3) + (Ku4/6)
      v_1 = v + (Kv1/6) + (Kv2/3) + (Kv3/3) + (Kv4/6)
      w_1 = w + (Kw1/6) + (Kw2/3) + (Kw3/3) + (Kw4/6)
      p_1 = p + (Kp1/6) + (Kp2/3) + (Kp3/3) + (Kp4/6)
      q_1 = q + (Kq1/6) + (Kq2/3) + (Kq3/3) + (Kq4/6)
      r_1 = r + (Kr1/6) + (Kr2/3) + (Kr3/3) + (Kr4/6)
      q0_1 = q0 + (Kq0_1/6) + (Kq0_2/3) + (Kq0_3/3) + (Kq0_4/6)
      q1_1 = q1 + (Kq1_1/6) + (Kq1_2/3) + (Kq1_3/3) + (Kq1_4/6)
      q2_1 = q2 + (Kq2_1/6) + (Kq2_2/3) + (Kq2_3/3) + (Kq2_4/6)
      q3_1 = q3 + (Kq3_1/6) + (Kq3_2/3) + (Kq3_3/3) + (Kq3_4/6)
      X_1 = X + (KX1/6) + (KX2/3) + (KX3/3) + (KX4/6)
      Y_1 = Y + (KY1/6) + (KY2/3) + (KY3/3) + (KY4/6)
      Z_1 = Z + (KZ1/6) + (KZ2/3) + (KZ3/3) + (KZ4/6)


      # self.state variable assignment with next step values (step n+1 is indicated with _1)
      self.state = [u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1]
      obs = np.array(self.state)

      # REWARD evaluation and done condition definition (to be completed)
      reward = (q0_1 - (u_1**2)/ self.VmaxSquared + (v_1**2)/ self.VmaxSquared + (w_1**2)/ self.VmaxSquared) 
      done = bool(Z>=0)
      info = {"u": u_1, "v": v_1, "w": w_1, "p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1, "X": X_1, "Y": Y_1, "Z": Z_1}

      print(obs)

      return obs, reward, done, info

  def reset(self):
      
      self.state = [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,-50.] # to initialize the state the object is put in x0=20 and v0=0
      obs = np.array(self.state)

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

  def eqnsOfMotion(self, u, v, w, p, q, r, q0, q1, q2, q3, X, Y, Z, dT1, dT2, dT3, dT4):
      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

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

      stateTime_derivatives= (u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, q0_dot, q1_dot, q2_dot, q3_dot, X_dot, Y_dot, Z_dot)
      return stateTime_derivatives