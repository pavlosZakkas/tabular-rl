import numpy as np

from Helper import argmax, softmax

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

class SoftMaxPolicy(ActionSelectionPolicy):

  def __init__(self, temperature):
    if temperature is None:
      raise KeyError("Provide a temperature")

    self.temperature = temperature

  def select_action_from(self, state, Q_sa, n_actions):
    action_probs = softmax(Q_sa[state], self.temperature)
    return np.random.choice(range(n_actions), 1, p=action_probs)[0]

