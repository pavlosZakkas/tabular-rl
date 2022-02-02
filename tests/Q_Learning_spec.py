import unittest
from parameterized import parameterized
import numpy as np
import mock

from Q_learning import QLearningAgent, ActionSelection

class QLearningSpec(unittest.TestCase):
  STATES = 3
  ACTIONS = 3
  GAMMA = 0.9
  LEARNING_RATE = 0.1
  EPSILON = 0.1
  TEMPERATURE = 10.0
  THRESHOLD = 0.001

  ACTION_1 = 0
  ACTION_2 = 1
  ACTION_3 = 2

  STATE_1 = 0
  STATE_2 = 1
  STATE_3 = 2
  STATE_VALUES_1 = [0, 3, 5]
  STATE_VALUES_2 = [10, 11, 8]
  STATE_VALUES_3 = [30, 21, 18]

  SOFTMAX_DENOMINATOR_SUM = np.sum(np.exp([-5/TEMPERATURE, -2/TEMPERATURE, 0/TEMPERATURE]))
  SOFTMAX_STATE_VALUES_1 = [
    np.exp(-5/TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
    np.exp(-2/TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
    np.exp(0/TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
  ]


  @parameterized.expand([
    (STATE_1, ACTION_3),
    (STATE_2, ACTION_2),
    (STATE_3, ACTION_1),
  ])
  @mock.patch('Q_learning.np')
  def should_select_the_highest_valued_action_for_egreedy_policy_if_the_sampled_selection_probability_is_greater_than_epsilon(
    self,
    state,
    action,
    mocked_numpy
  ):

    # given
    agent = QLearningAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1,
      self.STATE_VALUES_2,
      self.STATE_VALUES_3,
    ]

    mocked_numpy.random.uniform.return_value = self.EPSILON + 0.05
    # when
    selected_action = agent.select_action(state, ActionSelection.E_GREEDY, self.EPSILON)

    # then
    self.assertEqual(selected_action, action)

  @mock.patch('Q_learning.np')
  def should_select_a_random_action_for_egreedy_policy_if_the_sampled_selection_probability_is_smaller_than_epsilon(
    self,
    mocked_numpy
  ):

    # given
    agent = QLearningAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1,
      self.STATE_VALUES_2,
      self.STATE_VALUES_3,
    ]

    mocked_numpy.random.uniform.return_value = self.EPSILON - 0.05
    mocked_numpy.random.randint.return_value = self.ACTION_2

    # when
    selected_action = agent.select_action(self.STATE_1, ActionSelection.E_GREEDY, self.EPSILON)

    # then
    self.assertEqual(selected_action, self.ACTION_2)

  @mock.patch('Q_learning.np')
  def should_select_an_action_for_boltzmann_policy_based_on_softmax_of_actions(self, mocked_numpy):

    # given
    agent = QLearningAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1,
      self.STATE_VALUES_2,
      self.STATE_VALUES_3,
    ]

    def chosen_action(actions, size, p):
      self.assertEqual(actions, range(self.ACTIONS))
      self.assertEqual(size, 1)
      self.assertEqual(list(p), list(self.SOFTMAX_STATE_VALUES_1))
      return [self.ACTION_2]

    mocked_numpy.random.choice.side_effect = chosen_action
    # when
    selected_action = agent.select_action(self.STATE_1, ActionSelection.BOLTZMANN.value, temp=self.TEMPERATURE)

    # then
    self.assertEqual(selected_action, self.ACTION_2)
