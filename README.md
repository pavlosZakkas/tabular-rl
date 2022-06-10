# Tabular Reinforcement Learning
Experimentation with tabular RL methods was made by using several grid worlds from gym as environments.

## Policies
Specifically, the following methods were implemented:
- Value iteration
- Model-free tabular methods:
    - Q-learning
    - SARSA
    - n-step Q-learning and SARSA
    - Monte Carlo

## Exploration 
Below you may find the exploration methods that were used during action selection:
- e-greedy
- softmax
- annealing e-greedy
- annealing softmax

## Environments
The following stochastic environments are currently supported:
- Windy World
- Cliff walking
- Frozen lake
- Walk environment 

## Experiments
Several experiments were performed including:
- Comparison of the aforementioned policies
- Comparison of different n-step depths during bootstrapping
- Comparison of the aforementioned exploration methods

## Results
The results of our experimentation can be found in `figures` directory. To make our experiments statistically significant, 50 repetitions were made per experiment and the average results were presented.