import numpy as np
import random
from scipy.stats import beta

post_a = None
post_b = None
likelihood_mine = None
likelihood_opponent = None
avg_likelihood = None
bandit = None
total_reward = 0
c = 3
debug_print = False
decay = 0.99


def agent(observation, configuration):
    global reward_sums, total_reward, bandit, post_a, post_b, likelihood_mine, likelihood_opponent, c
    n_bandits = configuration.banditCount

    if observation.step == 0:
        post_a = np.ones(n_bandits)
        post_b = np.ones(n_bandits)
        likelihood_mine = np.ones(n_bandits)
        likelihood_opponent = np.ones(n_bandits)
        avg_likelihood = np.ones(n_bandits)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward
       
        ## Update Likehood of Arms
        if debug_print:
            print(observation.lastActions)
            print(configuration.decayRate)

        # likelihood_mine[observation.lastActions[0]] *= configuration.decayRate
        # likelihood_opponent[observation.lastActions[1]] *= configuration.decayRate
        likelihood_mine[observation.lastActions[0]] *= decay
        likelihood_opponent[observation.lastActions[1]] *= decay
        avg_likelihood = (likelihood_mine+likelihood_opponent)/2
        
        # Update Gaussian posterior
        post_a[bandit] += r
        post_b[bandit] += (1 - r)

    samples = np.random.beta(post_a, post_b)
    mean_beta, var_bar, _ , _ = beta.stats(post_a, post_b, moments='mvsk')
    final_pred = mean_beta
    bandit = int(np.argmax(samples))
    return bandit