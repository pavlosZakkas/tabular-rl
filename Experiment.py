#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

from experiments import ExplorationExperiment, BackUpPolicyExperiment, DepthPolicyExperiment

def experiment():
   # ExplorationExperiment.experiment()
   BackUpPolicyExperiment.experiment()
   # DepthPolicyExperiment.experiment()

if __name__ == '__main__':
    experiment()