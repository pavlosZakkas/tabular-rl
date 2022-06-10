import gym
from gym import spaces
from gym.utils import seeding

"""
The RandomWalk envrionment code was taken from 
https://github.com/NishanthVAnand/new_env_gym/blob/master/random_walk/envs/random_walk.py
"""
class RandomWalk(gym.Env):
  """
    This environment imports random walk environment.

    Two actions are possible:
    0: Moves up the chain (Right)
    1: Moves down the chain (Left)

    There are two terminal states location at both the extremes
    of the chain of states.
    1. The extreme right of the walk has a reward of +1
    2. The extreme left of the walk has a reward of 0

    The agent starts in a state that is located in between both
    these terminal states.
    """

  def __init__(self, n=21, slip=0.1, small=-1, large=1, name='random_walk'):
    self.n = n
    self.n_states = n
    self.n_actions = 2
    self.slip = slip  # probability of 'slipping' an action
    self.small = small  # payout for 'backwards' action
    self.large = large  # payout at end of chain for 'forwards' action
    self.state = 10  # Start at beginning of the chain
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Discrete(self.n)
    self.name = name
    self.type = 'gym'
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    done = False
    reward = 0
    assert self.action_space.contains(action)

    if self.np_random.rand() < self.slip:
      action = not action  # agent slipped, reverse action taken

    if self.state != 0 and self.state != 20:
      if action:
        self.state -= 1
      else:
        self.state += 1

    if self.state == 0:
      reward = self.small
      done = True

    elif self.state == 20:
      reward = self.large
      done = True

    return self.state, reward, done

  def reset(self):
    self.state = 10
    return self.state