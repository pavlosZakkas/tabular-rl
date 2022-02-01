#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, state):
        ''' Returns the greedy best action in state s ''' 
        return np.argmax(self.Q_sa[state])
        
    def update(self, state, action, prob_sas, reward_sas):
        # p_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        # r_sas = np.zeros((self.n_states, self.n_actions, self.n_states)) + self.reward_per_step 
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
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
     
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
