#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
np.random.seed(42)
from ActionSelectionPolicy import ActionSelectionPolicy, EGreedyPolicy, AnnealingEGreedyPolicy
from Environment import StochasticWindyGridworld

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, state, policy: ActionSelectionPolicy):
        return policy.select_action_from(state, self.Q_sa, self.n_actions)


    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        current_state = states[0]
        current_action = actions[0]
        last_state = states[-1]

        back_up_estimate = self.target_without_bootstrap(rewards) if done \
            else self.target_with_bootstrap(rewards, last_state)

        self.Q_sa[current_state][current_action] += self.learning_rate * \
                                                    (back_up_estimate - self.Q_sa[current_state][current_action])

    def target_without_bootstrap(self, rewards):
        return self.discounted_sum_of(rewards)

    def target_with_bootstrap(self, rewards, last_state):
        return self.discounted_sum_of(rewards) + \
            self.gamma**len(rewards) * max(self.Q_sa[last_state])

    def discounted_sum_of(self, rewards):
        return np.sum([self.gamma**i * reward for i, reward in enumerate(rewards)])

def n_step_Q(
  n_timesteps,
  max_episode_length,
  learning_rate,
  gamma,
  policy: ActionSelectionPolicy,
  plot=True,
  depth_n=5
):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    agent = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, depth_n)
    rewards = []

    times_reached_goal = 0
    timestep = 0
    while timestep < n_timesteps:
        state = env.reset()

        # play episode
        states, actions, timestep_rewards = [state], [], []
        done = False

        for episode_step in range(max_episode_length):

            selected_action = agent.select_action(state, policy)
            actions.append(selected_action)

            next_state, reward, done = env.step(selected_action)
            states.append(next_state)
            timestep_rewards.append(reward)
            rewards.append(reward)

            state = next_state

            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)

            timestep += 1
            if done or timestep == n_timesteps:
                if done:
                    times_reached_goal += 1
                break

        episode_length = episode_step + 1
        # rewards.append(timestep_rewards)

        # update Q table
        for episode_step in range(episode_length):
            n_states_plus_last_state = states[episode_step:episode_step + depth_n + 1]
            n_actions = actions[episode_step:episode_step + depth_n]
            n_timestep_rewards = timestep_rewards[episode_step:episode_step + depth_n]

            agent.update(n_states_plus_last_state, n_actions, n_timestep_rewards, done)
            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)

    print(f'Found goal state {times_reached_goal} times')
    return rewards

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

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

    rewards = n_step_Q(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        plot,
        depth_n=n
    )

    # print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
