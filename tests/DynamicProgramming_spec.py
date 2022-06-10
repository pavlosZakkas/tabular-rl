import unittest
from parameterized import parameterized
from mock import Mock
import mock
import numpy as np

from DynamicProgramming import QValueIterationAgent, Q_value_iteration

class DynamincProgrammingSpec(unittest.TestCase):

    STATES = 3
    ACTIONS = 3
    GAMMA = 0.9
    THRESHOLD = 0.001

    STATE_1 = 0
    STATE_2 = 1
    STATE_3 = 2
    STATE_VALUES_1 = [0, 3, 5]
    STATE_VALUES_2 = [10, 11, 8]
    STATE_VALUES_3 = [30, 21, 18]

    ACTION_1 = 0
    ACTION_2 = 1
    ACTION_3 = 2

    PROBS_SAS = [
        [
            [0.2, 0.7, 0.1], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]
        ],
        [
            [0.1, 0.8, 0.1], [0.5, 0.2, 0.3], [0.3, 0.3, 0.4]
        ],
        [
            [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.9, 0.1, 0.0]
        ],
    ]

    REWARDS_SAS = [
        [
            [4, 2, 1], [2, 12, 6], [-3, 3, 4]
        ],
        [
            [1, 18, 11], [5, -2, 7], [8, 10, 0]
        ],
        [
            [3, 5, 10], [21, 23, 26], [9, 10, 20]
        ],
    ]

    @parameterized.expand([
        (STATE_1, ACTION_3),
        (STATE_2, ACTION_2),
        (STATE_3, ACTION_1),
    ])
    def should_return_the_action_that_maximizes_value_from_current_state(self, state, expected_action):

        # given
        agent = QValueIterationAgent(self.STATES, self.ACTIONS, self.GAMMA)
        agent.Q_sa = [
            self.STATE_VALUES_1,
            self.STATE_VALUES_2,
            self.STATE_VALUES_3,
        ]

        # when
        selected_action = agent.select_action(state)

        # then
        self.assertEqual(selected_action, expected_action) 
 
    def should_perform_a_Q_iteration_update_based_on_probabilities_and_rewards(self):

        # given
        probs_sas = self.PROBS_SAS.copy()
        rewards_sas = self.REWARDS_SAS.copy()
        agent = QValueIterationAgent(self.STATES, self.ACTIONS, self.GAMMA)
        agent.Q_sa = [
            self.STATE_VALUES_1.copy(),
            self.STATE_VALUES_2.copy(),
            self.STATE_VALUES_3.copy(),
        ]
        
        # when
        agent.update(self.STATE_1, self.ACTION_1, probs_sas, rewards_sas)

        # then
        expected_updated_Q_sa = [
            [
                0.2 * (4 + self.GAMMA * 5) + 
                0.7 * (2 + self.GAMMA * 11) + 
                0.1 * (1 + self.GAMMA * 30), 
                3, 
                5
            ],
            self.STATE_VALUES_2,
            self.STATE_VALUES_3,
        ]

        np.testing.assert_array_equal(agent.Q_sa, expected_updated_Q_sa)

    @mock.patch('DynamicProgramming.QValueIterationAgent')
    def should_execute_Q_value_iteration_until_convergence(self, mocked_QValueIterationAgent):

        # given
        env = self.a_mocked_env()
        mocked_agent = self.a_mocked_agent(env)
        mocked_QValueIterationAgent.return_value = mocked_agent

        # when
        agent = Q_value_iteration(env, gamma=self.GAMMA, threshold=self.THRESHOLD)

        # then
        expected_update_calls = 2 * [
            mock.call(state, action, env.p_sas, env.r_sas)
            for state in range(env.n_states)
            for action in range(env.n_actions)
        ]

        self.assertEqual(mocked_agent.update.call_count, len(expected_update_calls))
        mocked_agent.update.assert_has_calls(expected_update_calls, any_order=False)
        self.assertEqual(agent, mocked_agent)
        assert np.all(agent.Q_sa > 0)

    def a_mocked_env(self):
        env = Mock()
        env.n_states = self.STATES
        env.n_actions = self.ACTIONS
        env.p_sas = self.PROBS_SAS.copy()
        env.r_sas = self.REWARDS_SAS.copy()
        return env

    def a_mocked_agent(self, env):
        agent = Mock()

        agent.n_states = env.n_states
        agent.n_actions = env.n_actions
        agent.Q_sa = np.zeros((env.n_states, env.n_actions))

        def mocked_updates(states, actions):
            for item in range(states * actions):
                yield np.random.uniform(2 * self.THRESHOLD, 10)
            for item in range(states * actions):
                yield np.random.uniform(0, self.THRESHOLD)

        updates_generator = mocked_updates(env.n_states, env.n_actions)
        def mocked_update_agent_Q_sa(state, action, p_sas, r_sas, **kwargs):
            agent.Q_sa[state][action] = np.add(agent.Q_sa[state][action], updates_generator.__next__())

        agent.update.side_effect = mocked_update_agent_Q_sa

        return agent

    # def should_test(self):
    #     env = self.a_mocked_env()
    #     agent = Q_value_iteration(env, gamma=1.0, threshold=self.THRESHOLD)
    #
    #     assert agent.Q_sa == []