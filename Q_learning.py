#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(42)

from ActionSelectionPolicy import ActionSelectionPolicy, EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy, \
    AnnealingSoftMaxPolicy
from Environment import StochasticWindyGridworld

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, state, policy: ActionSelectionPolicy):
        return policy.select_action_from(state, self.Q_sa, self.n_actions)

    def update(self, state, action, reward, next_state, done):
        back_up_estimate = reward + self.gamma * max(self.Q_sa[next_state])
        self.Q_sa[state][action] += self.learning_rate * (back_up_estimate - self.Q_sa[state][action])


def q_learning(
  n_timesteps,
  learning_rate,
  gamma,
  policy: ActionSelectionPolicy,
  plot=True
):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, learning_rate, gamma)
    rewards = []
    # first_timestep = 0

    state = env.reset()
    for timestep in range(n_timesteps):
        selected_action = agent.select_action(state, policy)
        next_state, reward, done, _ = env.step(selected_action)
        rewards.append(reward)
        agent.update(state, selected_action, reward, next_state, done)
        state = next_state

        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.2)

        if done:
            # print(f'Found goal in {timestep - first_timestep} steps')
            # first_timestep = timestep
            state = env.reset()

    return rewards

def test():

    n_timesteps = 1000
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

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
