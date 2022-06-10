#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from extra_environments.CliffWalkingEnvironment import CliffWalkingEnvironment
from ActionSelectionPolicy import EGreedyPolicy
from experiments.ExperimentHelper import average_over_repetitions
from Helper import LearningCurvePlot

GAMMA = 1.0
LEARNING_RATES = [0.2, 0.4]

BACKUPS = ['q', 'sarsa']
NAME_PER_BACKUP = {
    'q': 'Q-learning',
    'sarsa': 'SARSA'
}

TIMESTEPS = 50000
REPETITIONS = 50
SMOOTHING_WINDOW = 1501
PLOT = False
ALPHA = r'$\alpha$'

# Exploration:
DEAFULT_EGREEDY = EGreedyPolicy(0.05)
SELECTED_POLICY = DEAFULT_EGREEDY

FIGURES_DIR = 'Figures'
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

    Plot.save(os.path.join(BACKUP_POLICY_DIR, env.name, file_name))

def experiment(env=CliffWalkingEnvironment(name='frozen_lake')):
    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    if not os.path.isdir(BACKUP_POLICY_DIR):
        os.makedirs(BACKUP_POLICY_DIR)
    if not os.path.isdir(os.path.join(BACKUP_POLICY_DIR, env.name)):
        os.makedirs(os.path.join(BACKUP_POLICY_DIR, env.name))

    run_experiment_and_save_figure(
        gamma=GAMMA,
        backups=BACKUPS,
        learning_rates=LEARNING_RATES,
        fig_title=r'Q-learning versus SARSA for $\gamma$=' + str(GAMMA),
        file_name=f'on_off_policy_gamma-{GAMMA}.png',
        env=env
    )

if __name__ == '__main__':
    print('Additional BackUp experiment')
    experiment()