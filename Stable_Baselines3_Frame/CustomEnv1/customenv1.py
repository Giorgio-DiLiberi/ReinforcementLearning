# Code to generate a custom environment to train an RL agent with stable baselines
import numpy as np
import gym
from gym import spaces


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(3)
    # Example for using image as input (can be channel-first or channel-last):
    highObsSpace = np.array([500.])
    self.observation_space = spaces.Box(-highObsSpace,highObsSpace)
                                        # this should mean that observazion space is countinous 
                                        # and goes from -50 to 50 meters. Only x is observed
    self.state = None
    self.viewer = None

  def step(self, action):
      if action == 0:
        force = -10.
      elif action == 1:
        force = 0.
      elif action ==2:
        force = 10.

      x, v = self.state
      drag = 0. ######-0.1*force if v==0 else -0.1*v # drag definition: see the notebook

      v = v + (force - drag) * 0.01 # time step is assumed to be 0.01 seconds and simple euler is used
      x = x + v * 0.01

      self.state = (x,v)

      reward = -abs(x)/300
      done = bool(
            x < -300
            or x > 300
        )

      return np.array([self.state[0]]), reward, done, self.state[0]

  def reset(self):
      
      self.state = [-150., -1.] # to initialize the state the object is put in x0=20 and v0=0
      return np.array([self.state[0]])  # produce an observation of the first state (xPosition) 

  def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 1000.0
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            
            
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            
        if self.state is None:
            return None

        # Edit the pole polygon vertex
        

        x = self.state[0]
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close (self):
    if self.viewer:
            self.viewer.close()
            self.viewer = None

## Le funzioni di render e close non è necessario definirle, servono solo ad avere il visualizzatore grafico
## ma se non si può fare il rendering per il quadrotor possono anche essere lasciate in banco
