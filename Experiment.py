#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""
from extra_environments.TaxiEnvironment import TaxiEnvironment
from extra_environments.WalkEnvironment import RandomWalk
from experiments import BackUpPolicyExperiment, DepthPolicyExperiment, ExplorationExperiment, AdditionalBackUpPolicyExperiment
from Environment import StochasticWindyGridworld
from extra_environments.CliffWalkingEnvironment import CliffWalkingEnvironment
from extra_environments.FrozenLakeEnvironment import FrozenLakeEnvironment

WINDY_WORLD = StochasticWindyGridworld(initialize_model=False, name='windy_world')
CLIFF_WALKING_WORLD = CliffWalkingEnvironment(name='cliff_walking')
FROZEN_LAKE_WORLD = FrozenLakeEnvironment(name='frozen_lake')
# RANDOM_WALK_WORLD = RandomWalk(name='random_walk')
# TAXI_WORLD = TaxiEnvironment(name='taxi')

def experiment():

   ExplorationExperiment.experiment(WINDY_WORLD)
   BackUpPolicyExperiment.experiment(WINDY_WORLD)
   AdditionalBackUpPolicyExperiment.experiment(CLIFF_WALKING_WORLD)
   AdditionalBackUpPolicyExperiment.experiment(FROZEN_LAKE_WORLD)
   DepthPolicyExperiment.experiment(WINDY_WORLD)

   # ExplorationExperiment.experiment(CLIFF_WALKING_WORLD)
   # BackUpPolicyExperiment.experiment(CLIFF_WALKING_WORLD)
   # DepthPolicyExperiment.experiment(CLIFF_WALKING_WORLD)

   # DepthPolicyExperiment.experiment(TAXI_WORLD)


if __name__ == '__main__':
    experiment()