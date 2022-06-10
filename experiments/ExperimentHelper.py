
import numpy as np

from Environment import StochasticWindyGridworld

np.random.seed(42)
import time

from ActionSelectionPolicy import ActionSelectionPolicy
from Q_learning import q_learning
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import smooth

OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP = 1.0352600662204607

def average_over_repetitions(
  backup,
  n_repetitions,
  n_timesteps,
  max_episode_length,
  learning_rate,
  gamma,
  policy: ActionSelectionPolicy,
  smoothing_window=51,
  plot=False,
  n=5,
  env=StochasticWindyGridworld(initialize_model=False)
):
  reward_results = np.empty([n_repetitions, n_timesteps])  # Result array
  now = time.time()

  for rep in range(n_repetitions):  # Loop over repetitions
    if backup == 'q':
      rewards = q_learning(n_timesteps, learning_rate, gamma, policy, plot, env)
    elif backup == 'sarsa':
      rewards = sarsa(n_timesteps, learning_rate, gamma, policy, plot, env)
    elif backup == 'mc':
      rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                            policy, plot, env)
    elif backup == 'nstep':
      rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                         policy, plot, depth_n=n, env=env)

    reward_results[rep] = rewards

  print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
  learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
  learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
  return learning_curve

