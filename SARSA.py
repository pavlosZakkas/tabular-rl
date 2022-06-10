#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
np.random.seed(42)

from ActionSelectionPolicy import ActionSelectionPolicy, AnnealingEGreedyPolicy
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        
    def select_action(self, state, policy: ActionSelectionPolicy):
        return policy.select_action_from(state, self.Q_sa, self.n_actions)
        
    def update(self, state, action, reward, next_state, next_action, done):
        back_up_estimate = reward + self.gamma * self.Q_sa[next_state][next_action]
        self.Q_sa[state][action] += self.learning_rate * (back_up_estimate - self.Q_sa[state][action])

def sarsa(
    n_timesteps,
    learning_rate,
    gamma,
    policy: ActionSelectionPolicy,
    plot=True,
    env=StochasticWindyGridworld(initialize_model=False)
):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    agent = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    times_reached_goal = 0

    state = env.reset()
    action = agent.select_action(state, policy)

    for timestep in range(n_timesteps):
        next_state, reward, done = env.step(action)
        rewards.append(reward)
        next_action = agent.select_action(next_state, policy)
        agent.update(state, action, reward, next_state, next_action, done)

        if done:
            state = env.reset()
            action = agent.select_action(state, policy)
            times_reached_goal += 1
        else:
            state = next_state
            action = next_action

        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)

    print(f'Found goal state {times_reached_goal} times')
    return rewards


def test():
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    # policy = EGreedyPolicy(epsilon=0.1)
    policy = AnnealingEGreedyPolicy(
        timesteps=n_timesteps,
        initial_epsilon=0.9,
        final_epsilon=0.001,
        steps_percentage=0.9
    )
    # policy = SoftMaxPolicy(temperature=1.0)
    # policy = AnnealingSoftMaxPolicy(
    #     timesteps=n_timesteps,
    #     initial_temperature=10,
    #     final_temperature=0.001,
    #     steps_percentage=0.9
    # )
    
    # Plotting parameters
    plot = False

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, plot)
    print("Obtained rewards: {}".format(rewards))
    
if __name__ == '__main__':
    test()
