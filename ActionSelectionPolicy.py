import numpy as np

from Helper import argmax, softmax, linear_anneal

class ActionSelectionPolicy:

  def select_action_from(self, state, Q_sa, n_actions):
    pass

class EGreedyPolicy(ActionSelectionPolicy):

  def __init__(self, epsilon):
    if epsilon is None:
      raise KeyError("Provide an epsilon")

    self.epsilon = epsilon

  def select_action_from(self, state, Q_sa, n_actions):
    return self.random_action_from(n_actions) \
      if self.random_action_to_be_selected() \
      else self.highest_valued_action_from(state, Q_sa)

  def random_action_to_be_selected(self):
    return False if np.random.uniform(0, 1) > self.epsilon else True

  def random_action_from(self, n_actions):
    return np.random.randint(0, n_actions)

  def highest_valued_action_from(self, state, Q_sa):
    return argmax(Q_sa[state])

  def label(self, timesteps=None):
    epsilon = r'$\epsilon$'
    return f"{epsilon}-greedy {epsilon}={self.epsilon}"

class SoftMaxPolicy(ActionSelectionPolicy):

  def __init__(self, temperature):
    if temperature is None:
      raise KeyError("Provide a temperature")

    self.temperature = temperature

  def select_action_from(self, state, Q_sa, n_actions):
    action_probs = softmax(Q_sa[state], self.temperature)
    return np.random.choice(range(n_actions), 1, p=action_probs)[0]

  def label(self, timesteps=None):
    tau = r'$\tau$'
    return f"softmax {tau}={self.temperature}"

class AnnealingEGreedyPolicy(EGreedyPolicy):

  def __init__(self, timesteps, initial_epsilon, final_epsilon, steps_percentage):
    super().__init__(initial_epsilon)
    self.timesteps = timesteps
    self.initial_epsilon = initial_epsilon
    self.final_epsilon = final_epsilon
    self.steps_percentage = steps_percentage
    self.current_timestep = -1

  def select_action_from(self, state, Q_sa, n_actions):
    self.current_timestep += 1
    self.epsilon = linear_anneal(
      self.current_timestep,
      self.timesteps,
      self.initial_epsilon,
      self.final_epsilon,
      self.steps_percentage,
    )

    return super().select_action_from(state, Q_sa, n_actions)

  def label(self, timesteps=None):
    epsilon = r'$\epsilon$'
    return f"{epsilon}-greedy {epsilon} anneal {self.initial_epsilon}->{self.final_epsilon} in {int(self.steps_percentage * 100)}% steps"

class AnnealingSoftMaxPolicy(SoftMaxPolicy):

  def __init__(self, timesteps, initial_temperature, final_temperature, steps_percentage):
    super().__init__(initial_temperature)
    self.timesteps = timesteps
    self.initial_temperature = initial_temperature
    self.final_temperature = final_temperature
    self.steps_percentage = steps_percentage
    self.current_timestep = -1

  def select_action_from(self, state, Q_sa, n_actions):
    self.current_timestep += 1
    self.temperature = linear_anneal(
      self.current_timestep,
      self.timesteps,
      self.initial_temperature,
      self.final_temperature,
      self.steps_percentage,
    )

    return super().select_action_from(state, Q_sa, n_actions)

  def label(self, timesteps=None):
    tau = r'$\tau$'
    return f"softmax {tau} anneal {self.initial_temperature}->{self.final_temperature} in {int(self.steps_percentage * 100)}% steps"
