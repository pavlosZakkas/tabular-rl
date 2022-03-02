#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import time

import numpy as np
np.random.seed(42)
from ActionSelectionPolicy import ActionSelectionPolicy, AnnealingEGreedyPolicy, EGreedyPolicy
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def select_action(self, state, policy: ActionSelectionPolicy):
        return policy.select_action_from(state, self.Q_sa, self.n_actions)

    def update(self, states, actions, rewards):
        '''
        states is a list of states observed in the episode, of length T_ep
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        '''

        rewards_sum = 0
        for index, (state, action, reward) in enumerate(zip(states[::-1], actions[::-1], rewards[::-1])):
            rewards_sum = reward + self.gamma*rewards_sum
            self.Q_sa[state][action] += self.learning_rate * (rewards_sum - self.Q_sa[state][action])

def monte_carlo(
  n_timesteps,
  max_episode_length,
  learning_rate,
  gamma,
  policy: ActionSelectionPolicy,
  plot=True
):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    agent = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    times_reached_goal = 0

    timestep = 0
    while timestep < n_timesteps:
        state = env.reset()
        env.set_location_from(state)

        # play episode
        states, actions, timestep_rewards = [], [], []
        done = False

        for episode_step in range(max_episode_length):
            selected_action = agent.select_action(state, policy)
            states.append(state)
            actions.append(selected_action)

            next_state, reward, done = env.step(selected_action)
            rewards.append(reward)
            timestep_rewards.append(reward)

            state = next_state

            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)

            timestep += 1
            if done or timestep == n_timesteps:
                if done:
                    times_reached_goal += 1
                break

        # rewards.append(timestep_rewards)

        agent.update(states, actions, timestep_rewards)
        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=10.0)

        # time.sleep(100)
    print(f'Found goal state {times_reached_goal} times')
    return rewards 
    
def test():
    n_timesteps = 2
    max_episode_length = 10
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = EGreedyPolicy(epsilon=0.1)
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
    plot = True

    rewards = monte_carlo(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        plot
    )

    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
