#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os

from Environment import StochasticWindyGridworld

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ActionSelectionPolicy import EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy, \
    AnnealingSoftMaxPolicy
from experiments.ExperimentHelper import average_over_repetitions, OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP
from Helper import LearningCurvePlot

GAMMA = 1.0
GAMMAS = [1.0, 0.99, 0.9]
LEARNING_RATES = [0.05, 0.2, 0.4, 0.8, 1.5]

BACKUPS = ['q', 'sarsa']
NAME_PER_BACKUP = {
    'q': 'Q-learning',
    'sarsa': 'SARSA'
}

TIMESTEPS = 50000
REPETITIONS = 50
SMOOTHING_WINDOW = 1001
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
BACKUP_POLICY_DIR = f'{FIGURES_DIR}/BackUpPolicy'

def run_experiment_and_save_figure(gamma, backups, learning_rates, fig_title, file_name, env):
    Plot = LearningCurvePlot(title=fig_title)
    learning_curve_per_backup_and_lr = {}
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(
                backup,
                REPETITIONS,
                TIMESTEPS,
                None,
                learning_rate,
                gamma,
                SELECTED_POLICY,
                SMOOTHING_WINDOW,
                PLOT,
                None,
                env
            )
            learning_curve_per_backup_and_lr[(backup, learning_rate)] = learning_curve

    for (backup, learning_rate), learning_curve in learning_curve_per_backup_and_lr.items():
        Plot.add_curve(
            learning_curve,
            label=f'{NAME_PER_BACKUP[backup]} {ALPHA}={learning_rate}'
        )

    if env.name == 'windy_world':
        Plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")

    Plot.save(os.path.join(BACKUP_POLICY_DIR, env.name, file_name))

def experiment(env=StochasticWindyGridworld(initialize_model=False)):
    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    if not os.path.isdir(BACKUP_POLICY_DIR):
        os.makedirs(BACKUP_POLICY_DIR)
    if not os.path.isdir(os.path.join(BACKUP_POLICY_DIR, env.name)):
        os.makedirs(os.path.join(BACKUP_POLICY_DIR, env.name))

    # run_experiment_and_save_figure(
    #     gamma=GAMMA,
    #     backups=BACKUPS,
    #     learning_rates=LEARNING_RATES,
    #     fig_title=r'Q-learning versus SARSA for $\gamma$=' + str(GAMMA),
    #     file_name=f'on_off_policy_gamma-{GAMMA}.png',
    #     env=env
    # )

    for gamma in GAMMAS:
        run_experiment_and_save_figure(
            gamma=gamma,
            backups=BACKUPS,
            learning_rates=LEARNING_RATES,
            fig_title=r'Q-learning versus SARSA for $\gamma$=' + str(gamma),
            file_name=f'on_off_policy_gamma-{gamma}.png',
            env=env
        )

if __name__ == '__main__':
    print('BackUp')
    experiment()