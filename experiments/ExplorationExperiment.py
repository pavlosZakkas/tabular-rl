#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ActionSelectionPolicy import EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy, AnnealingSoftMaxPolicy
from ExperimentHelper import average_over_repetitions
from Helper import LearningCurvePlot


GAMMA = 1.0
# GAMMAS = [0.9, 0.99, 1.0]
LEARNING_RATE = 0.25
# LEARNING_RATES = [0.01, 0.1, 0.25, 0.7]
BACKUP = 'q'
TIMESTEPS = 50000
REPETITIONS = 50
SMOOTHING_WINDOW = 2001
PLOT = False
OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP = 1.0958

# Exploration:
# action selection policies to be tested
EGREEDY_POLICIES = [EGreedyPolicy(0.01), EGreedyPolicy(0.05), EGreedyPolicy(0.2)]
SOFTMAX_POLICIES = [SoftMaxPolicy(0.01), SoftMaxPolicy(0.1), SoftMaxPolicy(1.0)]

# after first experiment, we keep the best one out of egreedy and softmax policies
BEST_EGREEDY = EGreedyPolicy(0.01)
BEST_SOFTMAX = SoftMaxPolicy(0.1)

# annealing egreedy policies to compare with best egreedy policy
LOW_ANNEALING_EGREEDY_POLICIES = [
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.0, 0.3),
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.05, 0.2),
    AnnealingEGreedyPolicy(TIMESTEPS, 0.8, 0.01, 0.2),
]
MID_ANNEALING_EGREEDY_POLICIES = [
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.0, 0.6),
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.05, 0.5),
    AnnealingEGreedyPolicy(TIMESTEPS, 0.8, 0.01, 0.5),
]
HIGH_ANNEALING_EGREEDY_POLICIES = [
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.05, 0.3),
    AnnealingEGreedyPolicy(TIMESTEPS, 1.0, 0.0, 0.2),
    AnnealingEGreedyPolicy(TIMESTEPS, 0.8, 0.01, 0.2),
]
# annealing softmax policies to compare with best softmax policy
LOW_ANNEALING_SOFTMAX_POLICIES = [
    AnnealingSoftMaxPolicy(TIMESTEPS, 10.0, 0.01, 0.3),
    AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.2),
    AnnealingSoftMaxPolicy(TIMESTEPS, 1.0, 0.05, 0.2),
]
MID_ANNEALING_SOFTMAX_POLICIES = [
    AnnealingSoftMaxPolicy(TIMESTEPS, 10.0, 0.01, 0.6),
    AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.5),
    AnnealingSoftMaxPolicy(TIMESTEPS, 1.0, 0.05, 0.5),
]
HIGH_ANNEALING_SOFTMAX_POLICIES = [
    AnnealingSoftMaxPolicy(TIMESTEPS, 10.0, 0.01, 0.8),
    AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.7),
    AnnealingSoftMaxPolicy(TIMESTEPS, 1.0, 0.01, 0.7),
]


FIGURES_DIR = '../figures'
EXPLORATION_DIR = f'{FIGURES_DIR}/TaxiExploration'

if not os.path.isdir(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
if not os.path.isdir(EXPLORATION_DIR):
    os.makedirs(EXPLORATION_DIR)

def run_experiment_and_save_figure(policies, fig_title, file_name):
    Plot = LearningCurvePlot(title=fig_title)
    learning_curve_per_policy = {}
    for policy in policies:
        learning_curve = average_over_repetitions(
            BACKUP,
            REPETITIONS,
            TIMESTEPS,
            None,
            LEARNING_RATE,
            GAMMA,
            policy,
            SMOOTHING_WINDOW,
            PLOT,
            None
        )
        learning_curve_per_policy[policy] = learning_curve

    for policy, learning_curve in learning_curve_per_policy.items():
        Plot.add_curve(
            learning_curve,
            label=policy.label(TIMESTEPS)
        )

    Plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    Plot.save(os.path.join(EXPLORATION_DIR, file_name))

def experiment():

    # run_experiment_and_save_figure(
    #     policies=EGREEDY_POLICIES + SOFTMAX_POLICIES,
    #     fig_title='Q-learning: effect of $\epsilon$-greedy versus softmax exploration',
    #     file_name=f'exploration_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    # )

    run_experiment_and_save_figure(
        policies=[BEST_EGREEDY] + LOW_ANNEALING_EGREEDY_POLICIES,
        fig_title='Q-learning: effect of $\epsilon$-greedy versus annealing $\epsilon$-greedy',
        file_name=f'exploration_low_annealing_egreedy_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )

    run_experiment_and_save_figure(
        policies=[BEST_SOFTMAX] + LOW_ANNEALING_SOFTMAX_POLICIES,
        fig_title='Q-learning: effect of softmax versus annealing softmax',
        file_name=f'exploration_low_annealing_softmax_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )
    run_experiment_and_save_figure(
        policies=[BEST_EGREEDY] + MID_ANNEALING_EGREEDY_POLICIES,
        fig_title='Q-learning: effect of $\epsilon$-greedy versus annealing $\epsilon$-greedy',
        file_name=f'exploration_mid_annealing_egreedy_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )

    run_experiment_and_save_figure(
        policies=[BEST_SOFTMAX] + MID_ANNEALING_SOFTMAX_POLICIES,
        fig_title='Q-learning: effect of softmax versus annealing softmax',
        file_name=f'exploration_mid_annealing_softmax_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )
    run_experiment_and_save_figure(
        policies=[BEST_EGREEDY] + HIGH_ANNEALING_EGREEDY_POLICIES,
        fig_title='Q-learning: effect of $\epsilon$-greedy versus annealing $\epsilon$-greedy',
        file_name=f'exploration_high_annealing_egreedy_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )

    run_experiment_and_save_figure(
        policies=[BEST_SOFTMAX] + HIGH_ANNEALING_SOFTMAX_POLICIES,
        fig_title='Q-learning: effect of softmax versus annealing softmax',
        file_name=f'exploration_high_annealing_softmax_g-{GAMMA}_lr-{LEARNING_RATE}.png'
    )


if __name__ == '__main__':
    experiment()