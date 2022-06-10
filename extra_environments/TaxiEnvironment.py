#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
np.random.seed(42)
import gym

class TaxiEnvironment:

    def __init__(self, name='taxi'):
        self.env = gym.make('Taxi-v3')
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.p_sas, self.r_sas = self.get_transitions_and_rewards()
        self.reset()
        self.name = name
        self.type = 'gym'

    def get_transitions_and_rewards(self):
        p_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        r_sas = np.zeros((self.n_states, self.n_actions, self.n_states))

        for state, info_per_action in self.env.P.items():
            for action, info in info_per_action.items():
                for prob, new_state, reward, _ in info:
                    p_sas[state][action][new_state] = prob
                    r_sas[state][action][new_state] = reward

        return p_sas, r_sas

    def reset(self):
        return self.env.reset()

    def step(self, a):
        next_state, reward, done, info = self.env.step(a)
        return next_state, reward, done

    def model(self, s, a):
        return self.p_sas[s, a], self.r_sas[s, a]

    def render(self, Q_sa=None, plot_optimal_policy=False, step_pause=0.001):
        self.env.render()

    def save_as_image(self, iteration=None):
        pass

