import unittest
from parameterized import parameterized, parameterized_class

import DynamicProgramming
from DynamicProgramming import QValueIterationAgent

class DynamincProgrammingSpec(unittest.TestCase):

    STATES = 3
    ACTIONS = 3
    GAMMA = 0.9

    STATE_1 = 0
    STATE_2 = 1
    STATE_3 = 2
    STATE_VALUES_1 = [0, 3, 5]
    STATE_VALUES_2 = [10, 11, 8]
    STATE_VALUES_3 = [30, 21, 18]

    ACTION_1 = 0
    ACTION_2 = 1
    ACTION_3 = 2

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
        probs_sas = [
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
        rewards_sas = [
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
        # 0.2 * 4 + gamma*5 + 0.7 * 2 * gamma*11 + 0.1 *1 * gamma * 30
        # self.assertEqual(agent.Q_sa, expected_updated_Q_sa)
        import numpy as np
        np.testing.assert_array_equal(agent.Q_sa, expected_updated_Q_sa)