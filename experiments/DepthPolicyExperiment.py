#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os

from Environment import StochasticWindyGridworld

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ActionSelectionPolicy import AnnealingSoftMaxPolicy, EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy
from experiments.ExperimentHelper import average_over_repetitions, OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP
from Helper import LearningCurvePlot
import os

GAMMA = 1.0
GAMMAS = [1.0, 0.9]
LEARNING_RATE = 0.25
LEARNING_RATES = [0.05, 0.2, 0.4]
NSTEPS = [1, 3, 5, 10, 20, 100]

NSTEP_BACKUP = 'nstep'
MONTE_CARLO_BACKUP = 'mc'

TIMESTEPS = 50000
MAX_EPISODE_LENGTH = 100
MAX_EPISODE_LENGTHS = [100, 50, 500]

REPETITIONS = 50
SMOOTHING_WINDOW = 1201
PLOT = False
ALPHA = r'$\alpha$'

# Exploration:
DEAFULT_EGREEDY = EGreedyPolicy(0.05)
# BEST_EGREEDY = EGreedyPolicy(0.01)
# BEST_SOFTMAX = SoftMaxPolicy(0.1)
# BEST_ANNEALING_EGREEDY = AnnealingEGreedyPolicy(TIMESTEPS, 0.8, 0.01, 0.2)
# BEST_ANNEALING_SOFTMAX = AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.7)
SELECTED_POLICY = DEAFULT_EGREEDY

FIGURES_DIR = 'figures'
NSTEP_EXPERIMENTS_DIR = f'{FIGURES_DIR}/NStep'

def run_experiment_and_save_figure(gamma, nsteps, learning_rate, fig_title, file_name, env):

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
            SELECTED_POLICY,
            SMOOTHING_WINDOW,
            PLOT,
            n,
            env
        )
        learning_curve_per_backup_and_n[(NSTEP_BACKUP, n)] = learning_curve

    learning_curve = average_over_repetitions(
        MONTE_CARLO_BACKUP,
        REPETITIONS,
        TIMESTEPS,
        MAX_EPISODE_LENGTH,
        learning_rate,
        gamma,
        SELECTED_POLICY,
        SMOOTHING_WINDOW,
        PLOT,
        None,
        env
    )
    learning_curve_per_backup_and_n[(MONTE_CARLO_BACKUP, None)] = learning_curve

    for (backup, n), learning_curve in learning_curve_per_backup_and_n.items():
        Plot.add_curve(
            learning_curve,
            label=f'{n}-step Q-learning' if backup==NSTEP_BACKUP else 'Monte Carlo'
        )

    if env.name == 'windy_world':
        Plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")

    Plot.save(os.path.join(NSTEP_EXPERIMENTS_DIR, env.name, file_name), 'upper left')

def experiment(env=StochasticWindyGridworld(initialize_model=False)):
    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    if not os.path.isdir(NSTEP_EXPERIMENTS_DIR):
        os.makedirs(NSTEP_EXPERIMENTS_DIR)
    if not os.path.isdir(os.path.join(NSTEP_EXPERIMENTS_DIR, env.name)):
        os.makedirs(os.path.join(NSTEP_EXPERIMENTS_DIR, env.name))

    run_experiment_and_save_figure(
        gamma=GAMMA,
        nsteps=NSTEPS,
        learning_rate=LEARNING_RATE,
        fig_title=r'Effect of target depth ($\gamma$=' + str(GAMMA) + f', {ALPHA}={str(LEARNING_RATE)})',
        file_name=f'nstep_gamma-g_{GAMMA}-lr_{LEARNING_RATE}.png',
        env=env
    )
    #
    for learning_rate in LEARNING_RATES:
        run_experiment_and_save_figure(
            gamma=GAMMA,
            nsteps=NSTEPS,
            learning_rate=learning_rate,
            fig_title=r'Effect of target depth ($\gamma$=' + str(GAMMA) + f', {ALPHA}={str(learning_rate)})',
            file_name=f'nstep_gamma-g_{GAMMA}-lr_{learning_rate}.png',
            env=env
        )

    for gamma in GAMMAS:
        run_experiment_and_save_figure(
            gamma=gamma,
            nsteps=NSTEPS,
            learning_rate=LEARNING_RATE,
            fig_title=r'Effect of target depth ($\gamma$=' + str(gamma) + f', {ALPHA}={str(LEARNING_RATE)})',
            file_name=f'nstep_gamma-g_{gamma}-lr_{LEARNING_RATE}.png',
            env=env
        )


if __name__ == '__main__':
    experiment()