#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
from Environment import StochasticWindyGridworld
from TaxiEnvironment import TaxiEnvironment
from Helper import argmax
import gym

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, state):
        ''' Returns the greedy best action in state s ''' 
        return argmax(self.Q_sa[state])
        
    def update(self, state, action, prob_sas, reward_sas):
        states_probs = prob_sas[state][action]
        states_rewards = reward_sas[state][action]
        
        updated_Q_sa = np.sum(np.multiply(
            states_probs, 
            np.add(
                states_rewards,
                [self.gamma * self.Q_sa[s][self.select_action(s)] for s in range(self.n_states)]
            )
        ))
        self.Q_sa[state][action] = updated_Q_sa
        pass

    def get_updated_value(self, state, action, prob_sas, reward_sas):
        states_probs = prob_sas[state][action]
        states_rewards = reward_sas[state][action]

        updated_Q_sa = np.sum(np.multiply(
            states_probs,
            np.add(
                states_rewards,
                [self.gamma * self.Q_sa[s][self.select_action(s)] for s in range(self.n_states)]
            )
        ))
        return updated_Q_sa

    def update_whole_table(self, Q_sa):
        self.Q_sa = Q_sa

    def get_value_of(self, state):
        best_action = self.select_action(state)
        return self.Q_sa[state][best_action]

def Q_value_iteration(env, gamma=1.0, threshold=0.001, in_place_update = True):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
    env.save_as_image(0)

    initial_state_Vs, below_state_Vs, left_state_Vs, right_state_Vs = [], [], [], []
    initial_state = env.reset()
    below_state = env.state_2_below_from_target
    right_state = env.state_2_right_from_target
    left_state = env.state_2_left_from_target
    initial_state_Vs.append(QIagent.get_value_of(initial_state))
    below_state_Vs.append(QIagent.get_value_of(below_state))
    left_state_Vs.append(QIagent.get_value_of(left_state))
    right_state_Vs.append(QIagent.get_value_of(right_state))

    iteration = 1
    delta = threshold
    while delta >= threshold:
        delta = 0
        Q_sa = QIagent.Q_sa.copy()
        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                initial_value = QIagent.Q_sa[state][action]
                if in_place_update:
                    QIagent.update(state, action, env.p_sas, env.r_sas)
                    delta = max(delta, np.abs(initial_value - QIagent.Q_sa[state][action]))
                else:
                    Q_sa[state][action] = QIagent.get_updated_value(state, action, env.p_sas, env.r_sas)
                    delta = max(delta, np.abs(initial_value - Q_sa[state][action]))

        if not in_place_update:
            QIagent.update_whole_table(Q_sa)

        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        env.save_as_image(iteration)
        initial_state_Vs.append(QIagent.get_value_of(initial_state))
        below_state_Vs.append(QIagent.get_value_of(below_state))
        left_state_Vs.append(QIagent.get_value_of(left_state))
        right_state_Vs.append(QIagent.get_value_of(right_state))

        print(f"Q-value iteration, iteration {iteration}, max error {delta}")
        iteration += 1

    return QIagent, initial_state_Vs, below_state_Vs, left_state_Vs, right_state_Vs

def experiment(in_place_updates=True):
    name = 'DynamicProgramming_inPlaceUpdates' if in_place_updates else 'DynamicProgramming_batchUpdate'
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True, name=name)
    env.render()
    QIagent, initial_state_Vs, below_state_Vs, left_state_Vs, right_state_Vs = \
        Q_value_iteration(env, gamma, threshold, in_place_updates)

    fig, ax = plt.subplots(1,1)
    ax.plot(range(len(initial_state_Vs)), initial_state_Vs, label='initial state')
    ax.plot(range(len(below_state_Vs)), below_state_Vs, label='2 states below state')
    ax.plot(range(len(left_state_Vs)), left_state_Vs, label='2 states left from target')
    ax.plot(range(len(right_state_Vs)), right_state_Vs, label='2 states right from target')
    ax.set_title('Value of states per iteration')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Value of state')
    ax.legend()
    fig.savefig(f'figures/{name}/initial_state_Vs.png')

    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    initial_state = env.reset()
    best_action = QIagent.select_action(initial_state)
    initial_state_value = QIagent.Q_sa[initial_state][best_action]
    mean_reward_per_timestep =  initial_state_value / (env.goal_reward - initial_state_value)
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    print('Executing Q value iteration with in place updates of Q table')
    experiment(True)
    print('\nExecuting Q value iteration with one-off updates of the whole Q table')
    experiment(False)
