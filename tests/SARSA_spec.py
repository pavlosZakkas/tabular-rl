import unittest
from parameterized import parameterized
import numpy as np
import mock
from mock import Mock

from ActionSelectionPolicy import EGreedyPolicy, SoftMaxPolicy
from SARSA import SarsaAgent, sarsa

class SARSASpec(unittest.TestCase):
  STATES = 3
  ACTIONS = 3
  GAMMA = 0.9
  LEARNING_RATE = 0.1
  EPSILON = 0.1
  TEMPERATURE = 10.0
  THRESHOLD = 0.001
  TIMESTEPS = 1000

  ACTION_1 = 0
  ACTION_2 = 1
  ACTION_3 = 2

  STATE_1 = 0
  STATE_2 = 1
  STATE_3 = 2
  STATE_VALUES_1 = [0, 3, 5]
  STATE_VALUES_2 = [10, 11, 8]
  STATE_VALUES_3 = [30, 21, 18]

  SOFTMAX_DENOMINATOR_SUM = np.sum(np.exp([-5 / TEMPERATURE, -2 / TEMPERATURE, 0 / TEMPERATURE]))
  SOFTMAX_STATE_VALUES_1 = [
    np.exp(-5 / TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
    np.exp(-2 / TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
    np.exp(0 / TEMPERATURE) / SOFTMAX_DENOMINATOR_SUM,
  ]

  MOCKED_RANDOM_STATE = STATE_2
  MOCKED_NEXT_STATE = STATE_1
  MOCKED_REWARD = -1.0
  MOCKED_UPDATE = 1.0
  DONE = True
  MOCKED_SELECTED_ACTION = ACTION_3
  MOCKED_NEXT_ACTION = ACTION_3
  STEPS_TO_REACH_GOAL = 10

  @parameterized.expand([
    (STATE_1, ACTION_3),
    (STATE_2, ACTION_2),
    (STATE_3, ACTION_1),
  ])
  @mock.patch('ActionSelectionPolicy.np')
  def should_select_the_highest_valued_action_for_egreedy_policy_if_the_sampled_selection_probability_is_greater_than_epsilon(
    self,
    state,
    action,
    mocked_numpy
  ):
    # given
    agent = SarsaAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1,
      self.STATE_VALUES_2,
      self.STATE_VALUES_3,
    ]

    mocked_numpy.random.uniform.return_value = self.EPSILON + 0.05
    # when
    selected_action = agent.select_action(state, EGreedyPolicy(self.EPSILON))

    # then
    self.assertEqual(selected_action, action)

  @mock.patch('ActionSelectionPolicy.np')
  def should_select_a_random_action_for_egreedy_policy_if_the_sampled_selection_probability_is_smaller_than_epsilon(
    self,
    mocked_numpy
  ):
    # given
    agent = SarsaAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1,
      self.STATE_VALUES_2,
      self.STATE_VALUES_3,
    ]

    mocked_numpy.random.uniform.return_value = self.EPSILON - 0.05
    mocked_numpy.random.randint.return_value = self.ACTION_2

    # when
    selected_action = agent.select_action(self.STATE_1, EGreedyPolicy(self.EPSILON))

    # then
    self.assertEqual(selected_action, self.ACTION_2)

  @mock.patch('ActionSelectionPolicy.np')
  def should_select_an_action_for_boltzmann_policy_based_on_softmax_of_actions(self, mocked_numpy):
    # given
    agent = SarsaAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
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
    selected_action = agent.select_action(self.STATE_1, SoftMaxPolicy(self.TEMPERATURE))

    # then
    self.assertEqual(selected_action, self.ACTION_2)

  def should_update_Q_table_based_on_one_step_sarsa_update(self):
    # given
    agent = SarsaAgent(self.STATES, self.ACTIONS, self.LEARNING_RATE, self.GAMMA)
    agent.Q_sa = [
      self.STATE_VALUES_1.copy(),
      self.STATE_VALUES_2.copy(),
      self.STATE_VALUES_3.copy(),
    ]
    reward = 10.0

    # when
    agent.update(self.STATE_1, self.ACTION_2, reward, self.STATE_2, self.ACTION_3, None)

    # then
    updated_Q_value = 3 + self.LEARNING_RATE * ((reward + self.GAMMA * self.STATE_VALUES_2[self.ACTION_3]) - 3)
    assert agent.Q_sa == [
      [0, updated_Q_value, 5],
      self.STATE_VALUES_2,
      self.STATE_VALUES_3
    ]

  @mock.patch('Environment.StochasticWindyGridworld')
  @mock.patch('SARSA.SarsaAgent')
  def should_update_Q_table_and_return_rewards_per_timestep_of_sarsa_algorithm(
    self,
    mocked_SarsaAgent,
    mocked_windy_grid
  ):
    # given
    mocked_env = self.a_mocked_env()
    mocked_agent = self.a_mocked_agent()
    mocked_SarsaAgent.return_value = mocked_agent

    # when
    rewards = sarsa(
      self.TIMESTEPS,
      self.LEARNING_RATE,
      self.GAMMA,
      policy=EGreedyPolicy(epsilon=self.EPSILON),
      plot=False,
      env=mocked_env
    )

    # then
    expected_update_calls = [
                              mock.call(
                                self.STATE_1,
                                self.MOCKED_SELECTED_ACTION,
                                self.MOCKED_REWARD,
                                self.MOCKED_NEXT_STATE,
                                self.MOCKED_SELECTED_ACTION,
                                not self.DONE
                              )
                            ] + (self.TIMESTEPS - 1) * [
                              mock.call(
                                self.MOCKED_NEXT_STATE,
                                self.MOCKED_SELECTED_ACTION,
                                self.MOCKED_REWARD,
                                self.MOCKED_NEXT_STATE,
                                self.MOCKED_NEXT_ACTION,
                                not self.DONE
                              )
                            ]

    self.assertEqual(mocked_agent.update.call_count, len(expected_update_calls))
    mocked_agent.update.assert_has_calls(expected_update_calls)
    np.testing.assert_array_equal(
      mocked_agent.Q_sa,
      [[0., 0., self.TIMESTEPS], [0., 0., 0.], [0., 0., 0.]]
    )

    self.assertEqual(rewards, [self.MOCKED_REWARD] * self.TIMESTEPS)

  def a_mocked_env(self, steps_to_reach_goal=None):
    def mocked_env_steps(action):
      return self.MOCKED_NEXT_STATE, self.MOCKED_REWARD, not self.DONE

    def mocked_env_reset(**kwargs):
      return self.STATE_1

    env = Mock()
    env.n_states = self.STATES
    env.n_actions = self.ACTIONS

    env.step = Mock(side_effect=mocked_env_steps)
    env.reset = Mock(side_effect=mocked_env_reset)
    return env

  def a_mocked_agent(self):
    agent = Mock()

    agent.n_states = self.STATES
    agent.n_actions = self.ACTIONS
    agent.Q_sa = np.zeros((self.STATES, self.ACTIONS))

    def mocked_update_agent_Q_sa(state, action, reward, next_state, next_action, done, **kwargs):
      assert action == self.MOCKED_SELECTED_ACTION
      agent.Q_sa[state][action] += self.MOCKED_UPDATE

    agent.update.side_effect = mocked_update_agent_Q_sa
    agent.select_action = Mock(return_value=self.MOCKED_SELECTED_ACTION)
    return agent
