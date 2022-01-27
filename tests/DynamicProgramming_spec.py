import unittest
from parameterized import parameterized, parameterized_class

import DynamicProgramming
from DynamicProgramming import QValueIterationAgent

class DynamincProgrammingSpec(unittest.TestCase):

    STATES = 3
    ACTIONS = 5
    GAMMA = 0.9

    STATE_1 = 0
    STATE_2 = 1
    STATE_3 = 2
    STATE_VALUES_1 = [1, 3, 5, 6, 7]
    STATE_VALUES_2 = [10, 11, 8, 3, 2]
    STATE_VALUES_3 = [30, 21, 18, 24, 28]

    @parameterized.expand([
        (STATE_1, 4),
        (STATE_2, 1),
        (STATE_3, 0),
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
 