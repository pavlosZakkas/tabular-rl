#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ActionSelectionPolicy import ActionSelectionPolicy, EGreedyPolicy, SoftMaxPolicy, AnnealingEGreedyPolicy, \
    AnnealingSoftMaxPolicy
from ExperimentHelper import average_over_repetitions
from Helper import LearningCurvePlot, smooth
import os

GAMMA = 0.9
GAMMAS = [1.0, 0.99, 0.9]
LEARNING_RATES = [0.05, 0.2, 0.4, 0.8, 1.5]

BACKUPS = ['q', 'sarsa']

TIMESTEPS = 50000
REPETITIONS = 50
SMOOTHING_WINDOW = 1001
PLOT = False
OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP = 1.0958
ALPHA = r'$\alpha$'

# Exploration:
BEST_EGREEDY = EGreedyPolicy(0.01)
BEST_SOFTMAX = SoftMaxPolicy(0.1)
BEST_ANNEALING_EGREEDY = AnnealingEGreedyPolicy(TIMESTEPS, 0.8, 0.01, 0.2)
BEST_ANNEALING_SOFTMAX = AnnealingSoftMaxPolicy(TIMESTEPS, 5.0, 0.01, 0.7)
SELECTED_POLICY = BEST_ANNEALING_SOFTMAX

FIGURES_DIR = '../figures'
BACKUP_POLICY_DIR = f'{FIGURES_DIR}/BackUpPolicy'

if not os.path.isdir(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
if not os.path.isdir(BACKUP_POLICY_DIR):
    os.makedirs(BACKUP_POLICY_DIR)

def run_experiment_and_save_figure(gamma, backups, learning_rates, fig_title, file_name):

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
                None
            )
            learning_curve_per_backup_and_lr[(backup, learning_rate)] = learning_curve

    for (backup, learning_rate), learning_curve in learning_curve_per_backup_and_lr.items():
        Plot.add_curve(
            learning_curve,
            label=f'{backup} {ALPHA}={learning_rate}'
        )

    Plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    Plot.save(os.path.join(BACKUP_POLICY_DIR, file_name))

def experiment():
    run_experiment_and_save_figure(
        gamma=GAMMA,
        backups=BACKUPS,
        learning_rates=LEARNING_RATES,
        fig_title=r'Q-learning versus SARSA for $\gamma$=' + str(GAMMA),
        file_name=f'on_off_policy_gamma-{GAMMA}.png'
    )

    for gamma in GAMMAS:
        run_experiment_and_save_figure(
            gamma=gamma,
            backups=BACKUPS,
            learning_rates=LEARNING_RATES,
            fig_title=r'Q-learning versus SARSA for $\gamma$=' + str(gamma),
            file_name=f'on_off_policy_gamma-{gamma}.png'
        )

if __name__ == '__main__':
    experiment()