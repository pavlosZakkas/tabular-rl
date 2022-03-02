#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ActionSelectionPolicy import ActionSelectionPolicy, EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy, \
    AnnealingSoftMaxPolicy
from ExperimentHelper import average_over_repetitions
from Helper import LearningCurvePlot, smooth
import os

GAMMA = 1.0
GAMMAS = [1.0]
LEARNING_RATE = 0.25
LEARNING_RATES = [0.05, 0.2, 0.4]
# LEARNING_RATES = [0.4, 0.8]
NSTEPS = [1, 3, 5, 10, 20, 100]

NSTEP_BACKUP = 'nstep'
MONTE_CARLO_BACKUP = 'mc'

TIMESTEPS = 50000
MAX_EPISODE_LENGTH = 100
MAX_EPISODE_LENGTHS = [100, 50, 500]

REPETITIONS = 5
SMOOTHING_WINDOW = 2001
PLOT = False
OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP = 1.0958
ALPHA = r'$\alpha$'

# Exploration:
# BEST_EGREEDY = EGreedyPolicy(0.01)
BEST_POLICY = AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.7)
# BEST_SOFTMAX = SoftMaxPolicy(0.01)

FIGURES_DIR = '../figures'
NSTEP_EXPERIMENTS_DIR = f'{FIGURES_DIR}/NStep'

if not os.path.isdir(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
if not os.path.isdir(NSTEP_EXPERIMENTS_DIR):
    os.makedirs(NSTEP_EXPERIMENTS_DIR)

def run_experiment_and_save_figure(gamma, nsteps, learning_rate, fig_title, file_name):

    Plot = LearningCurvePlot(title=fig_title)
    learning_curve_per_backup_and_n = {}

    for n in nsteps:
        learning_curve = average_over_repetitions(
            NSTEP_BACKUP,
            REPETITIONS,
            TIMESTEPS,
            MAX_EPISODE_LENGTH,
            learning_rate,
            gamma,
            BEST_POLICY,
            SMOOTHING_WINDOW,
            PLOT,
            n
        )
        learning_curve_per_backup_and_n[(NSTEP_BACKUP, n)] = learning_curve

    learning_curve = average_over_repetitions(
        MONTE_CARLO_BACKUP,
        REPETITIONS,
        TIMESTEPS,
        MAX_EPISODE_LENGTH,
        learning_rate,
        gamma,
        BEST_POLICY,
        SMOOTHING_WINDOW,
        PLOT,
        None
    )
    learning_curve_per_backup_and_n[(MONTE_CARLO_BACKUP, None)] = learning_curve

    for (backup, n), learning_curve in learning_curve_per_backup_and_n.items():
        Plot.add_curve(
            learning_curve,
            label=f'{n}-step Q-learning' if backup==NSTEP_BACKUP else 'Monte Carlo'
        )

    Plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    Plot.save(os.path.join(NSTEP_EXPERIMENTS_DIR, file_name), 'upper left')

def experiment():

    run_experiment_and_save_figure(
        gamma=GAMMA,
        nsteps=NSTEPS,
        learning_rate=LEARNING_RATE,
        fig_title=r'Effect of target depth ($\gamma$=' + str(GAMMA) + f', {ALPHA}={str(LEARNING_RATE)})',
        file_name=f'nstep_gamma-g_{GAMMA}-lr_{LEARNING_RATE}.png'
    )
    #
    # for gamma in GAMMAS:
    #     run_experiment_and_save_figure(
    #         gamma=gamma,
    #         nsteps=NSTEPS,
    #         learning_rate=LEARNING_RATE,
    #         fig_title=r'Effect of target depth ($\gamma$=' + str(gamma) + f', {ALPHA}={str(LEARNING_RATE)})',
    #         file_name=f'nstep_gamma-g_{gamma}-lr_{LEARNING_RATE}.png'
    #     )
    #
    # for learning_rate in LEARNING_RATES:
    #     run_experiment_and_save_figure(
    #         gamma=GAMMA,
    #         nsteps=NSTEPS,
    #         learning_rate=learning_rate,
    #         fig_title=r'Effect of target depth ($\gamma$=' + str(GAMMA) + f', {ALPHA}={str(learning_rate)})',
    #         file_name=f'nstep_gamma-g_{GAMMA}-lr_{learning_rate}.png'
    #     )

if __name__ == '__main__':
    experiment()