# Code to generate a Quadcopter Environment to train an RL agent with stable baselines
# this model include a mathematical rapresentation of the quadcopter and a PID which7
# takes as input average thr, p ref, q ref and r ref and outputs torques requests, they 
# have to be mixed and added (or subtracted) to average throttle to obtain the throttle 
# value for each motor wich remains the input for the equations of motion.
# This model is the third version in which all the scalar operations is substitued by vector
# ones with numpy
import numpy as np
import gym
from gym import spaces

class QuadcoptEnvV4(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(QuadcoptEnvV4, self).__init__()


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
    self.Lx = 0.2   #[m] X body Length (squared x configuration)
    self.Ly = 0.2   #[m] Y body Length
    
    # Motors position vectors from CG
    self.rM1=np.array([self.Lx/2, self.Ly/2, 0.]) 
    self.rM2=np.array([-self.Lx/2, self.Ly/2, 0.])
    self.rM3=np.array([-self.Lx/2, -self.Ly/2, 0.])
    self.rM4=np.array([self.Lx/2, -self.Ly/2, 0.]) 

    # atmosphere and gravity definition (assumed constant but a standard atmosphere model can be included)
    self.rho = 1.225 #[kg/m^3] Standard day at 0 m ASL
    self.g0 = 9.815  #[m/s^2] gravity acceleration

    # mass of components
    self.mass = 1.   #[kg] mass is 600 grams can be changed
    self.motor_mass = 0.04 #[kg] mass of one motor+prop
    self.body_mass= 0.54 #[kg] mass of body frame + electronics (for inertia it is considered as 
    # uniformly distributed in a sphere centered in CG with radius 0.06m)
    self.battery_mass = 0.3 #[kg] mass of battery, considered at a distance of 0.06m from CG aligned with it on zb
    
    self.Wned = np.array([0, 0, self.mass * self.g0]) # Weight vector in NED axes
   
    ## Inertia tensor is considered dyagonal, null the other components
    self.Ix = 4*((self.Ly/2)**2)*self.motor_mass +\
      (0.06**2)*self.battery_mass + 0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to X axis
    
    self.Iy = 4*((self.Lx/2)**2)*self.motor_mass +\
      (0.06**2)*self.battery_mass + 0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to Y axis
    
    self.Iz = 4*(((self.Lx/2)**2)+((self.Ly/2)**2))*self.motor_mass +\
      0.4*(0.06**2)*self.body_mass #[kg m^2] rotational Inertia referred to Z axis

    # Inertia tensor composition
    self.InTen = np.array([[self.Ix, 0., 0.],[0., self.Iy, 0.],[0., 0., self.Iz]])

    # Inertia vector: vector with 3 principal inertia useful in evaluating the Omega_dot
    self.InVec = np.diag(self.InTen)

    ## The motors model is now assumed as reported on the notebook with thrust and torques dependant on 
    # a constant multiplied by the square of prop's rounds per sec:
    # F = Kt * n**2 where n[rounds/s] = Thr * nMax and nMax is evaluated as Kv*nominal_battery_voltage/60
    self.Motor_Kv = 2500 # [RPM/V] known for te specific motor
    self.V_batt_nom = 14.8 # [V] nominal battery voltage 
    self.nMax_motor = self.Motor_Kv * self.V_batt_nom / 60 #[RPS]

    # Props Values
    self.D_prop = 0.1778 #[m] diameter for 7 inch prop
    self.Ct = 0.1164 # Constant of traction tabulated for V=0
    self.Cp = 0.064  # Constant of power tabulated for v=0
    self.Prop_Kf = self.Ct * self.rho * (self.D_prop**4) #[kg m]
    self.Prop_Kq = self.Cp * self.rho * (self.D_prop**5)/(2*np.pi) #[kg m^2]
    # now force and torque are evaluated as:
    # F=Kf * N_prop^2 
    # F=Kq * N_prop^2 in an appropriate method   
    # N are rounds per sec (not rad/s) 

    # Throttle constants for mapping (mapping is linear-cubic, see the act2Thr() method)
    self.dTt = np.sqrt(self.mass * self.g0 / (4*self.Prop_Kf)) / self.nMax_motor # trim throttle to hover
    self.d2 = 0.65 # Assumed value for first constant in action to throttle mapping
    self.d1 = 1 - self.d2 - self.dTt # second constant for maping (see notebook)
    self.s2 = self.d2 - 1 + 2*self.dTt # first constant for left part
    self.s1 = self.dTt - self.s2
    
    self.Cd = np.array([0.2, 0.2, 0.2]) # Vector of drag constants for three main body axes normal surfaces
    self.Sn = np.array([0.02, 0.02, 0.05]) #[m^2] Vector of normal surfaces to main body axes to calculate drag
    # Zb normal surface is greater than othe two  

    self.C_DR = 0.01 # [kg m^2/s] constant to evaluate the aerodynamic torque which model a drag
    # for angular motion, coefficient is assumed

    # integration parameters: constant step of 0.01 [s]
    self.timeStep = 0.01
    self.max_Episode_time_steps = 2000 # maximum number of timesteps in an episode (=20s)
    self.elapsed_time_steps = 0 # time steps elapsed since the beginning of an episode, to be updated each step
    

    # useful Constants to normalize state and evaluate reward
    self.VmaxSquared = 2500 #[(m/s)^2] Squared by deafult to save some computation

    self.Goal_Altitude = -10 #[m] altitude to achieve is 10 m


    # PID constants
    self.p_P = 0.1 # proportional gain p
    self.p_I = 0.1 # integral gain p

    self.q_p = 0.1 # proportional gain q
    self.q_I = 0.1 # integral gain q

    self.r_p = 0.1 # proportional gain q
    self.r_I = 0.1 # integral gain q


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

      self.elapsed_time_steps += 1 # update for time steps

      # obs normalization is performed dividing state_next_step array by normalization vector
      # with elementwise division
      obs = self.state / self.Obs_normalization_vector

      # REWARD evaluation and done condition definition (to be completed)
      u_1, v_1, w_1, p_1, q_1, r_1, q0_1, q1_1, q2_1, q3_1, X_1, Y_1, Z_1 = State_next_step

      reward = self.getReward()

      done = self.isDone()
    
      info = {"u": u_1, "v": v_1, "w": w_1, "p": p_1, "q": q_1, "r": r_1, "q0": q0_1, "q1": q1_1, "q2": q2_1, "q3": q3_1, "X": X_1, "Y": Y_1, "Z": Z_1}

      return obs, reward, done, info

  def reset(self):

      """
      Reset state 
      """
      
      self.state = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,-50.]) # to initialize the state the object is put in x0=20 and v0=0
      
      self.elapsed_time_steps = 0 # reset for elapsed time steps

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

  def getReward(self):

      """
      Function which given a certain state evaluates the reward, to be called in step method.
      input: none, take self.state
      output: reward, scalar value.
      """

      u, v, w = self.state[0:3]
      q0 = self.state[6]
      X, Y, Z = self.state[10:13] 
      
      reward = q0 - ((u**2)/ self.VmaxSquared) - ((v**2)/ self.VmaxSquared) - \
        ((w**2)/ self.VmaxSquared) - (((Z - self.Goal_Altitude)**2)/ self.Obs_normalization_vector[12])\
          - ((X**2)/ self.Obs_normalization_vector[11]) - ((Y**2)/ self.Obs_normalization_vector[11])

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

      elif self.elapsed_time_steps >= self.max_Episode_time_steps:

        done = True
        
      else :

        done = False

      return done

## In this sections are defined functions to evaluate forces and derivatives to make the step function easy to read

  def Drag(self, V):
    
      """
      This function return an Aerodynamical drag given velocity cd and normal Surface
      input: relative wind speed.
      output: drag force (vector if input is a vector)
      """
      # Evaluation of the AERODYNAMICAL DRAG: this force is modeled as 3 scalar values
      # calculated with the formula D=0.5 S Cd rho V^2 where V is the single component 
      # of velocity in body axes and S and Cd are referred to the surface normal to the 
      # selected V component: E.G. to evaluate X component of drag V = u and S and Cd are 
      # those referred to the front section.
      # Evaluation performed in vector form

      drag = - 0.5 * self.rho * V * abs(V) * self.Sn * self.Cd  #[N]

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
      N_prop = Throttle * self.nMax_motor #[RPS] number of rounds per second for BL motor
      
      Thrust = self.Prop_Kf * (N_prop**2) #[N]
      Torque = self.Prop_Kq * (N_prop**2) #[N m]

      return Thrust, Torque # return scalar thrust and torque



  def eqnsOfMotion(self, State, Throttle):

      """
      This function evaluates the xVec_dot=fVec(x,u) given the states and controls in current step
      """
      # This function is implemented separately to make the code more easily readable

      Vb = State[0:3] # Subvector CG velocity [m/s]
      Omega = State[3:6] # Subvector angular velocity [rad/s]
      q0, q1, q2, q3 = State[6:10] # Quaternion

      # Motors section (vectors are evaluated later in this method)
      dT1, dT2, dT3, dT4 = Throttle

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