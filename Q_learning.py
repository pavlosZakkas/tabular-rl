#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

from enum import Enum

class ActionSelection(Enum):
    E_GREEDY = 'egreedy'
    BOLTZMANN = 'softmax'

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def random_action_to_be_selected(self, epsilon):
        return False if np.random.uniform(0, 1) > epsilon else True

    def random_action(self):
        return np.random.randint(0, self.n_actions)

    def highest_valued_action_from(self, state):
        return argmax(self.Q_sa[state])

    def select_action(self, state, policy=ActionSelection.E_GREEDY.value, epsilon=None, temp=None):
        
        if policy == ActionSelection.E_GREEDY:
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            action = self.random_action() \
                if self.random_action_to_be_selected(epsilon) \
                else self.highest_valued_action_from(state)
                
        elif policy == ActionSelection.BOLTZMANN.value:
            if temp is None:
                raise KeyError("Provide a temperature")
                
            action_probs = softmax(self.Q_sa[state], temp)
            action = np.random.choice(range(self.n_actions), 1, p=action_probs)[0]
            
        return action
        
    def update(self, state, action, reward, next_state, done):
        back_up_estimate = reward + self.gamma * max(self.Q_sa[next_state])
        self.Q_sa[state][action] += self.learning_rate * (back_up_estimate - self.Q_sa[state][action])
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Q-learning algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    return rewards 

def test():
    
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
