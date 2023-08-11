import sys, os
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

from min_reward_oracle import min_reward

###################################################
# baselines
###################################################

def use_middle(param_int, agent_oracle, reward_mode):
    """ solve optimal reward relative to midpoint of uncertainty interval
    sequential policy, but based on the center of the uncertainty set """
    attractiveness = param_int.mean(axis=1)
    attractiveness = np.zeros(param_int.shape[0])  # TODO try this; raw attractiveness
    agent_br = agent_oracle.best_response([attractiveness], [1.], reward_mode, display=True)
    return agent_br

def maximin(park_params, agent_oracle, reward_mode):
    """ maximize min reward, analogous to robust adversarial RL (RARL) """
    # n_iters = 10
    n_iters = 5
    min_reward_iters = 300
    # pick initial attractiveness at random
    attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
    attractiveness = attractiveness.astype(float)
    # batch_size = 64
    batch_size = 8

    for iter in range(n_iters):
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], reward_mode, display=False)
        if iter == n_iters - 1: break
        attractiveness = min_reward(park_params,
                                    agent_strategy,
                                    reward_mode,
                                    attractiveness_learning_rate=5e-2,
                                    n_iter=min_reward_iters,
                                    batch_size=batch_size,
                                    visualize=False,
                                    init_attractiveness=attractiveness)

    return agent_strategy

def RARL_regret(park_params, agent_oracle, nature_oracle, reward_mode):
    """ use a weakened form of MIRROR that is equivalent to RARL with regret,
    using the nature oracle to compute regret instead of maximin reward """
    # n_iters = 10
    n_iters = 5
    print("NOTE! DOING RARL REGRET WITH 5 ITER FOR NOW")
    # n_iters = 1
    # pick initial attractiveness at random
    attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
    attractiveness = attractiveness.astype(float)

    for iter in range(n_iters):
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], reward_mode, display=False)
        if iter == n_iters - 1: break
        attractiveness = nature_oracle.best_response([agent_strategy], [1.], [reward_mode], reward_mode, display=False)

    return agent_strategy

# def myopic(param_int, agent_oracle):
#     """ regular myopic - can use whatever method to come up with policies. will need to evaluate based on minimax regret

#     myopic minimax? look at only our reward in the next timestep - would need
#     to use bender's decomposition to compute. and we also have continuous policies
#     """
#     pass

class RandomPolicy:
    def __init__(self, park_params):
        self.n_targets = park_params['n_targets']
        self.budget    = park_params['budget']

    def select_action(self, state):
        max_effort = 1 # max effort at any target

        action = np.random.rand(self.n_targets)
        action /= action.sum()
        action *= self.budget

        # ensure we never exceed effort = 1 on any target
        while len(np.where(action > max_effort)[0]) > 0:
            excess_idx = np.where(action > 1)[0][0]
            excess = action[excess_idx] - max_effort

            action[excess_idx] = max_effort

            # add "excess" amount of effort randomly on other targets
            add = np.random.uniform(size=self.n_targets - 1)
            add = (add / np.sum(add)) * excess

            action[:excess_idx] += add[:excess_idx]
            action[excess_idx+1:] += add[excess_idx:]

        return action