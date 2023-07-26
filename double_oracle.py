"""
double oracle implementation - putting it all together

Lily Xu, 2021
"""

import sys, os
import pickle
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from scipy import signal # for 2D gaussian kernel
import torch

import matplotlib.pyplot as plt

from park import convert_to_a
from agent_oracle import AgentOracle
from nature_oracle import NatureOracle
from nfg_solver import solve_game, solve_minimax_regret, get_payoff, get_nature_best_strategy

from min_reward_oracle import min_reward

if not os.path.exists('plots'):
    os.makedirs('plots')

def breakpoint():
    import pdb; pdb.set_trace()


class DoubleOracle:
    def __init__(self,
                 max_epochs,
                 height,
                 width,
                 budget,
                 horizon,
                 n_perturb,
                 n_eval,
                 agent_n_train,
                 nature_n_train,
                 hunting_attract_vals,
                 logging_attract_vals,
                 psi,
                 alpha,
                 beta,
                 eta,
                 max_interval,
                 wildlife_setting,
                 use_wake,
                 checkpoints,
                 freeze_policy_step,
                 freeze_a_step,
                 exp_path,
                 write_initialization,
                 read_initialization,
                 objective
                 ):
        self.max_epochs  = max_epochs

        self.park_height = height
        self.park_width  = width
        self.budget = budget
        self.horizon = horizon

        self.n_targets = self.park_height * self.park_width
        assert self.budget < self.n_targets, 'Budget must be strictly less than the number of cells in park'

        self.n_perturb = n_perturb
        self.objective = objective
        self.secondary = 'logging' if objective == 'poaching' else 'poaching'

        # attractiveness parameter interval
        hunting_int = np.random.uniform(0, max_interval, size=self.n_targets)
        # self.hunting_param_int = [(hunting_attract_vals[i]-hunting_int[i], hunting_attract_vals[i]+hunting_int[i]) for i in range(self.n_targets)]
        self.hunting_param_int = [(hunting_attract_vals[i], hunting_attract_vals[i]+hunting_int[i]) for i in range(self.n_targets)]
        # print('hunting_param_int', [tuple(np.round(hunting_int, 2)) for hunting_int in self.hunting_param_int])

        self.hunting_param_int = np.array(self.hunting_param_int)
        assert np.all(self.hunting_param_int[:, 1] >= self.hunting_param_int[:, 0])

        # TODO P3: add some way to specify how correlated the hunting and logging are
        logging_int = np.random.uniform(0, max_interval, size=self.n_targets)
        # self.logging_param_int = [(logging_attract_vals[i]-logging_int[i], logging_attract_vals[i]+logging_int[i]) for i in range(self.n_targets)]
        self.logging_param_int = [(logging_attract_vals[i], logging_attract_vals[i]+logging_int[i]) for i in range(self.n_targets)]
        # print('logging_param_int', [tuple(np.round(logging_int, 2)) for logging_int in self.logging_param_int])

        self.logging_param_int = np.array(self.logging_param_int)
        assert np.all(self.logging_param_int[:, 1] >= self.logging_param_int[:, 0])

        self.hunting_param_int = read_write_initialization_vals(self.hunting_param_int, '_hunting_param_int.txt', exp_path, write_initialization, read_initialization)
        self.logging_param_int = read_write_initialization_vals(self.logging_param_int, '_logging_param_int.txt', exp_path, write_initialization, read_initialization)
        print('hunting_param_int', np.round(self.hunting_param_int, 2))
        print('logging_param_int', np.round(self.logging_param_int, 2))


        def gkern(kernlen=21, std=3):
            """ returns a 2D Gaussian kernel array """
            gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
            gkern2d = np.outer(gkern1d, gkern1d)
            return gkern2d.flatten()

        # TODO: disaggregate between wildlife and trees!
        if wildlife_setting == 1: # random
            initial_wildlife = np.random.rand(self.n_targets) * 3
            initial_trees = np.random.rand(self.n_targets) * 3
            # NOTE: might want to equalize/shuffle these :o
        elif wildlife_setting == 2: # gaussian kernel - peaked
            assert height == width
            initial_wildlife = gkern(height, 1.5) * 3
            initial_trees = gkern(height, 1.5) * 3
        elif wildlife_setting == 3: # gaussian kernel - not very peaked
            assert height == width
            initial_wildlife = gkern(height, 5) * 3
            initial_trees = gkern(height, 5) * 3
        else:
            raise Exception('wildlife setting {} not implemented'.format(wildlife_setting))
        
        assert wildlife_setting == 1, "If changing wildlife setting, need to change read/write initialization exp name!"
        initial_wildlife = read_write_initialization_vals(initial_wildlife, '_initial_wildlife.txt', exp_path, write_initialization, read_initialization)
        initial_trees = read_write_initialization_vals(initial_trees, '_initial_trees.txt', exp_path, write_initialization, read_initialization)

        # setup park parameters dict
        self.park_params = {
          'height': self.park_height,
          'width': self.park_width,
          'budget': self.budget,
          'horizon': self.horizon,
          'n_targets': self.n_targets,
          'initial_effort': np.zeros(self.n_targets),
          'initial_wildlife': initial_wildlife,
          'initial_trees': initial_trees,
          'initial_attack': np.zeros(self.n_targets),
          'param_int': self.hunting_param_int,
          'param_int_logging': self.logging_param_int,
          'psi': psi,
          'alpha': alpha,
          'beta': beta,
          'eta': eta,
          'objective': self.objective,
          'secondary': self.secondary
        }

        print('initial wildlife {:.2f} {}'.format(np.sum(initial_wildlife), np.round(initial_wildlife, 2)))
        print('initial trees {:.2f} {}'.format(np.sum(initial_trees), np.round(initial_trees, 2)))

        self.agent_oracle = AgentOracle(self.park_params, checkpoints, agent_n_train, n_eval)
        self.nature_oracle = NatureOracle(self.park_params,
                                          checkpoints,
                                          nature_n_train,
                                          use_wake,
                                          freeze_policy_step,
                                          freeze_a_step,
                                          threat_mode=self.objective)
        # TODO: do we made a second NatureOracle but just for the trees?
        self.nature_oracle_secondary = NatureOracle(self.park_params,  # TODO would need to specify to use logging params
                                                  checkpoints,
                                                  nature_n_train,
                                                  use_wake,
                                                  freeze_policy_step,
                                                  freeze_a_step,
                                                  threat_mode=self.secondary)


        # initialize attractiveness
        init_attractiveness_poaching = (np.random.rand(self.n_targets) - .5) * 2
        init_attractiveness_logging = (np.random.rand(self.n_targets) - .5) * 2

        # initialize strategy sets
        self.agent_strategies  = []  # agent policy
        self.nature_strategies_poaching = [init_attractiveness_poaching]  # attractiveness
        self.nature_strategies_logging = [init_attractiveness_logging]  # attractiveness
        self.payoffs_poaching = [] # agent regret for each (agent strategy, attractiveness) combo
        self.payoffs_logging = [] # agent regret for each (agent strategy, attractiveness) combo


    def run(self):
        agent_eq  = np.array([1.]) # account for baselines
        nature_eq = np.array([1.])  # initialize nature distribution

        # repeat until convergence
        converged = False
        n_epochs = 1
        while not converged:
            print('-----------------------------------')
            print('epoch {}'.format(n_epochs))
            print('-----------------------------------')

            # if first epoch, agent response is ideal agent for initial attractiveness
            nature_strategies = self.nature_strategies_poaching if self.objective == 'poaching' else self.nature_strategies_logging

            agent_br = self.agent_oracle.best_response(nature_strategies, nature_eq, threat_mode=self.objective, display=False)
            nature_br = self.nature_oracle.best_response(self.agent_strategies, agent_eq, display=False)
            nature_br_secondary = self.nature_oracle_secondary.best_response(self.agent_strategies, agent_eq, display=False)
            # REWARD RANDOMIZATION
            # repeat with more perturbations of nature strategies
            print(f'  nature BR for primary ({self.objective})', np.round(nature_br, 3))
            print(f'  nature BR for secondary ({self.secondary})', np.round(nature_br_secondary, 3))

            self.update_payoffs(nature_br, agent_br, payoff_mode=self.objective)
            self.update_payoffs_nature(nature_br_secondary, payoff_mode=self.secondary)
            
            # perturb
            print('Skipping perturbations for simplicity!')
            # for _ in range(self.n_perturb):
            #     perturb = np.random.normal(scale=0.5, size=nature_br.shape)
            #     perturbed_br = nature_br + perturb
            #     print('  perturbed', np.round(perturbed_br, 3))

            #     agent_perturbed_br = self.agent_oracle.best_response([perturbed_br], [1], display=False)

            #     self.update_payoffs(perturbed_br, agent_perturbed_br)

            # find equilibrium of subgame
            agent_eq, nature_eq = self.find_equilibrium(payoff_mode=self.objective)

            print('agent equilibrium  ', np.round(agent_eq, 3))
            print('nature equilibrium ', np.round(nature_eq, 3))

            # max_regret_game = np.array(self.payoffs_poaching) - np.array(self.payoffs_poaching).max(axis=0)

            if n_epochs >= self.max_epochs: # terminate after a max number of epochs
                converged = True
                break

            n_epochs += 1

            assert len(self.payoffs_poaching) == len(self.agent_strategies), '{} payoffs, {} agent strategies'.format(len(self.payoffs_poaching), len(self.agent_strategies))
            assert len(self.payoffs_poaching[0]) == len(self.nature_strategies_poaching), '{} payoffs[0], {} nature strategies'.format(len(self.payoffs_poaching[0]), len(self.nature_strategies_poaching))

            assert len(self.payoffs_logging) == len(self.agent_strategies), '{} payoffs, {} agent strategies'.format(len(self.payoffs_logging), len(self.agent_strategies))
            assert len(self.payoffs_logging[0]) == len(self.nature_strategies_logging), '{} payoffs[0], {} nature strategies'.format(len(self.payoffs_logging[0]), len(self.nature_strategies_logging))

        return agent_eq, nature_eq


    def compute_regret(self, agent_s, nature_s, max_reward, threat_mode):
        reward = self.agent_oracle.simulate_reward([agent_s], [nature_s], threat_mode, display=False)
        regret = max_reward - reward
        if regret < 0:
            print('  uh oh! regret is negative. max reward {:.3f}, reward {:.3f}'.format(max_reward, reward))
        return regret

    def compute_payoff_regret(self, agent_eq, payoff_mode='poaching'):
        """ given a agent mixed strategy, compute the expected regret in the payoff matrix """
        assert abs(sum(agent_eq) - 1) <= 1e-3
        
        if payoff_mode == 'poaching':
            payoffs = do.payoffs_poaching
        else:
            payoffs = do.payoffs_logging

        regret = np.array(payoffs) - np.array(payoffs).max(axis=0)
        # if agent playing a pure strategy
        if len(np.where(agent_eq > 0)[0]) == 1:
            agent_strategy_i = np.where(agent_eq > 0)[0].item()
            strategy_regrets = regret[agent_strategy_i]
            return -np.min(strategy_regrets) # return max regret (min reward)
        else:
            raise Exception('not implemented')

    def find_equilibrium(self, payoff_mode='poaching'):
        """ solve for minimax regret-optimal mixed strategy """
        payoffs = self.payoffs_poaching if payoff_mode == 'poaching' else self.payoffs_logging
        agent_eq, nature_eq = solve_minimax_regret(payoffs)
        return agent_eq, nature_eq

    def update_payoffs(self, nature_br, agent_br, payoff_mode='poaching'):
        """ update payoff matrix (in place) """
        self.update_payoffs_agent(agent_br)
        self.update_payoffs_nature(nature_br, payoff_mode=payoff_mode)

    def update_payoffs_agent(self, agent_br):
        """ update payoff matrix (only adding agent strategy)

        returns index of new strategy """
        self.agent_strategies.append(agent_br)

        # for new agent strategy: compute regret w.r.t. all nature strategies
        new_payoffs_poaching = []
        new_payoffs_logging = []
        for nature_s_poaching in self.nature_strategies_poaching:
            reward = self.agent_oracle.simulate_reward([agent_br], [nature_s_poaching], threat_mode='poaching', display=False)
            new_payoffs_poaching.append(reward)
        for nature_s_logging in self.nature_strategies_logging:  # FIXME figure out how we update nature_s_logging
            reward = self.agent_oracle.simulate_reward([agent_br], [nature_s_logging], threat_mode='logging', display=False)
            new_payoffs_logging.append(reward)
        self.payoffs_poaching.append(new_payoffs_poaching)
        self.payoffs_logging.append(new_payoffs_logging)
        return len(self.agent_strategies) - 1

    def update_payoffs_nature(self, nature_br, payoff_mode='poaching'):
        """ update payoff matrix (only adding nature strategy)

        returns index of new strategy """
        if payoff_mode == 'poaching':
            self.nature_strategies_poaching.append(nature_br)
        else:
            self.nature_strategies_logging.append(nature_br)

        # update payoffs
        # for new nature strategy: compute regret w.r.t. all agent strategies
        for i, agent_s in enumerate(self.agent_strategies):
            reward = self.agent_oracle.simulate_reward([agent_s], [nature_br], payoff_mode, display=False)  # what about the distribution??
            if payoff_mode == 'poaching':
                self.payoffs_poaching[i].append(reward)
            else:
                self.payoffs_logging[i].append(reward)

        return len(self.nature_strategies_poaching) - 1
    
    def print_agent_strategy(self, agent_eq, nature_eq, threat_mode):
        # breakpoint()
        print()
        print(f'Printing policies for threat mode {threat_mode}')
        nonzero_agent_strategies = [self.agent_strategies[i] for i in range(len(agent_eq)) if agent_eq[i] != 0]
        nature_strategies = self.nature_strategies_poaching if threat_mode == 'poaching' else self.nature_strategies_logging
        nonzero_nature_strategies = [nature_strategies[i] for i in range(len(nature_eq)) if nature_eq[i] != 0]
        for i, agent_strategy in enumerate(nonzero_agent_strategies):
            for j, nature_strategy in enumerate(nonzero_nature_strategies):
                policies, rewards, states = self.agent_oracle.simulate_policy(agent_strategy, nature_strategy, threat_mode, display=True)
                print(f'Agent strategy {i}, Nature strategy {j}')
                print('policies')
                print(policies)
                print('rewards')
                print(rewards)
                print('states')
                print(states)
                print()


###################################################
# baselines
###################################################

def use_middle(param_int, agent_oracle, threat_mode):
    """ solve optimal reward relative to midpoint of uncertainty interval
    sequential policy, but based on the center of the uncertainty set """

    attractiveness = param_int.mean(axis=1)
    agent_br = agent_oracle.best_response([attractiveness], [1.], threat_mode, display=True)
    return agent_br

def maximin(park_params, agent_oracle, threat_mode):
    """ maximize min reward, analogous to robust adversarial RL (RARL) """
    # n_iters = 10
    n_iters = 5
    min_reward_iters = 300
    # pick initial attractiveness at random
    attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
    attractiveness = attractiveness.astype(float)
    # batch_size = 64
    batch_size = 4

    for iter in range(n_iters):
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], threat_mode, display=False)
        if iter == n_iters - 1: break
        attractiveness = min_reward(park_params,
                                    agent_strategy,
                                    threat_mode,
                                    attractiveness_learning_rate=5e-2,
                                    n_iter=min_reward_iters,
                                    batch_size=batch_size,
                                    visualize=False,
                                    init_attractiveness=attractiveness)

    return agent_strategy

def RARL_regret(park_params, agent_oracle, nature_oracle, threat_mode):
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
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], threat_mode, display=False)
        if iter == n_iters - 1: break
        attractiveness = nature_oracle.best_response([agent_strategy], [1.], display=False)

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

def read_write_initialization_vals(arr, name, exp_path, write_initialization, read_initialization):
    full_path = exp_path + name
    if write_initialization:
        np.savetxt(full_path, arr)
    if read_initialization:
        arr = np.loadtxt(full_path)
    return arr

def read_write_initialization_pickle(obj, name, exp_path, write_initialization, read_initialization):
    import pickle
    full_path = exp_path + name
    if write_initialization:
        with open(full_path, 'wb') as f:
            pickle.dump(obj, f)

    if read_initialization:
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIRROR robust reinforcement learning under minimax regret')

    parser.add_argument('--seed',         type=int, default=0, help='random seed')
    parser.add_argument('--n_eval',       type=int, default=100, help='number of points to evaluate agent reward')
    parser.add_argument('--agent_train',  type=int, default=100, help='number of training iterations for agent')
    parser.add_argument('--nature_train', type=int, default=100, help='number of training iterations for nature')
    parser.add_argument('--max_epochs',   type=int, default=5, help='max num epochs to run double oracle')
    parser.add_argument('--n_perturb',    type=int, default=3, help='number of perturbations to add in each epoch')
    parser.add_argument('--wake',         type=int, default=1, help='whether to use wake/sleep (binary option)')

    parser.add_argument('--freeze_policy_step', type=int, default=5, help='how often to freeze policy (nature wake/sleep)')
    parser.add_argument('--freeze_a_step',      type=int, default=5, help='how often to unfreeze attractiveness (nature wake/sleep)')

    # set park parameters
    parser.add_argument('--height',  type=int, default=5, help='park height')
    parser.add_argument('--width',   type=int, default=5, help='park width')
    parser.add_argument('--budget',  type=int, default=5, help='agent budget')
    parser.add_argument('--horizon', type=int, default=5, help='agent planning horizon')

    parser.add_argument('--interval',   type=float, default=3, help='uncertainty interval max size')
    parser.add_argument('--wildlife',   type=int,   default=1, help='wildlife option')
    parser.add_argument('--deterrence', type=int,   default=1, help='deterrence option')

    parser.add_argument('--prefix', type=str, default='', help='filename prefix')
    parser.add_argument('--objective', type=str, default='poaching', help='whether to train on poaching or logging')
    parser.add_argument('--balance_attract', type=int, default=1, help='whether to use shuffled hunting attract vals for logging')
    parser.add_argument('--toy', type=int, default=0, help='whether to use toy data')

    args = parser.parse_args()

    seed = args.seed
    n_eval = args.n_eval
    agent_n_train = args.agent_train
    nature_n_train = args.nature_train
    max_epochs = args.max_epochs
    n_perturb = args.n_perturb
    use_wake = args.wake == 1
    is_toy = args.toy == 1

    # parameters for nature oracle wake/sleep
    freeze_policy_step = args.freeze_policy_step
    freeze_a_step = args.freeze_a_step

    height = args.height
    width = args.width
    budget = args.budget
    horizon = args.horizon

    max_interval = args.interval
    wildlife_setting = args.wildlife
    deterrence_setting = args.deterrence

    prefix  = args.prefix
    objective = args.objective
    secondary = 'logging' if objective == 'poaching' else 'poaching'

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_filename = './data/sample.p'
    # TODO: where did this data come from?
    data = pickle.load(open(data_filename, 'rb'))
    hunting_start_idx = np.random.randint(len(data['attract_vals'][0]) - height*width)
    logging_start_idx = np.random.randint(len(data['attract_vals'][0]) - height*width)
    hunting_attract_vals = data['attract_vals'][0][hunting_start_idx:hunting_start_idx + height*width] # pick random series of attractiveness values
    logging_attract_vals = data['attract_vals'][0][logging_start_idx:logging_start_idx + height*width] # pick random series of attractiveness values
    print('hunting_attract_vals', np.round(hunting_attract_vals, 2))
    print('logging_attract_vals', np.round(logging_attract_vals, 2))
    hunting_attract_vals = np.array(hunting_attract_vals) + 13
    logging_attract_vals = np.array(logging_attract_vals) + 13
    print('hunting_attract_vals', np.round(hunting_attract_vals, 2))
    print('logging_attract_vals', np.round(logging_attract_vals, 2))

    write_initialization = False
    read_initialization = True
    assert write_initialization != read_initialization, "Think read/write through carefully"
    if write_initialization:
        print("WRITE MODE")
    else:
        print("READ-ONLY MODE")
    balance_attract = True if args.balance_attract == 1 else 0
    balance_attract_str = '_balance_attract' if balance_attract else ''
    if balance_attract:
        logging_attract_vals = hunting_attract_vals.copy()
        np.random.shuffle(logging_attract_vals)

    exp_path = f'initialization_vals/seed_{seed}_height_{height}_width_{width}_max_interval_{max_interval}{balance_attract_str}'

    if is_toy:
        print('NOTE: OVERRIDING EXP PATH FOR TOY DATA!')
        exp_path = f'initialization_vals/toy'

    hunting_attract_vals = read_write_initialization_vals(hunting_attract_vals, '_hunting_attract_vals.txt', exp_path, write_initialization, read_initialization)
    logging_attract_vals = read_write_initialization_vals(logging_attract_vals, '_logging_attract_vals.txt', exp_path, write_initialization, read_initialization)
    
    psi = 1.1 # wildlife growth ratio
    alpha = .5  # strength that poachers eliminate wildlife
    eta = .3  # effect of neighbors
    if deterrence_setting == 1:
        beta = -5
    elif deterrence_setting == 2:
        beta = -3
    elif deterrence_setting == 3:
        beta = -8
    print('beta is', beta)
    print('eta is', eta)
    print('all beta', data['past_effort_vals'])

    print('beta {:.3f}, eta {:.3f}'.format(beta, eta))

    checkpoints = [1, 50, 100, 500, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 80000, 100000, 120000, 150000, 170000]#, N_TRAIN-1]
    # breakpoint()
    do = DoubleOracle(max_epochs,
                      height,
                      width,
                      budget,
                      horizon,
                      n_perturb,
                      n_eval,
                      agent_n_train,
                      nature_n_train,
                      hunting_attract_vals,
                      logging_attract_vals,
                      psi,
                      alpha,
                      beta,
                      eta,
                      max_interval,
                      wildlife_setting,
                      use_wake,
                      checkpoints,
                      freeze_policy_step,
                      freeze_a_step,
                      exp_path,
                      write_initialization,
                      read_initialization,
                      objective
                      )

    print('max_epochs {}, n_train agent {}, nature {}'.format(max_epochs, agent_n_train, nature_n_train))
    print('n_targets {}, horizon {}, budget {}'.format(do.n_targets, horizon, budget))

    # # baseline: middle of uncertainty interval
    print('########## BASELINE MIDDLE ##########')
    baseline_middle_i = len(do.agent_strategies)
    start_time = time.time()
    assert n_perturb == 0, "Need to modify if scaling up the perturbations"
    for i in range(n_perturb+1):
        # if read_initialization:
        #     baseline_middle = read_write_initialization_pickle(None, f'_baseline_middle_{i}.pickle', exp_path, write_initialization, read_initialization)
        # else:
        baseline_middle = use_middle(do.hunting_param_int, do.agent_oracle, objective)
        # if write_initialization:
        #     read_write_initialization_pickle(baseline_middle, f'_baseline_middle_{i}.pickle', exp_path, write_initialization, read_initialization)
        do.update_payoffs_agent(baseline_middle)
    middle_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline middle runtime {:.1f} seconds'.format(middle_time))

    # baseline: random
    print('########## BASELINE RANDOM ##########')
    baseline_random_i = len(do.agent_strategies)
    start_time = time.time()

    for i in range(n_perturb+1):
        random_policy = RandomPolicy(do.park_params)
        do.update_payoffs_agent(random_policy)
    random_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline random runtime {:.1f} seconds'.format(random_time))
    
    # baseline: maximin robust (robust adversarial RL - RARL)
    # print('Skipping maximin baseline since it take 20 minutes')
    print('########## BASELINE MAXIMIN ##########')
    baseline_maximin_i = len(do.agent_strategies)
    start_time = time.time()
    for i in range(n_perturb+1):
        maximin_policy = maximin(do.park_params, do.agent_oracle, objective)
        do.update_payoffs_agent(maximin_policy)
    maximin_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline maximin runtime {:.1f} seconds'.format(maximin_time))

    # baseline: RARL with regret
    # print('Skipping RARL baseline since it take 77 minutes')
    print('########## BASELINE RARL WITH REGRET ##########')
    baseline_RARL_regret_i = len(do.agent_strategies)
    start_time = time.time()
    for i in range(n_perturb+1):
        RARL_regret_policy = RARL_regret(do.park_params, do.agent_oracle, do.nature_oracle, objective)
        do.update_payoffs_agent(RARL_regret_policy)
    RARL_regret_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline RARL_regret runtime {:.1f} seconds'.format(RARL_regret_time))
    print()
    print('strategies', do.agent_strategies)

    print('########## RUNNING DOUBLE ORACLE ##########')
    start_time = time.time()
    agent_eq, nature_eq = do.run()
    do_time = time.time() - start_time
    print('DO runtime {:.1f} seconds'.format(do_time))

    print('\n\n\n\n\n-----------------------')
    # print('equilibrium value is ', val_upper, val_lower)
    print('agent BR mixed strategy           ', np.round(agent_eq, 4))
    print('Nature attractiveness mixed strategy ', np.round(nature_eq, 4))
    print('Nature attractiveness are')
    for nature_strategy in do.nature_strategies_poaching:
        a = convert_to_a(nature_strategy, do.hunting_param_int)
        print('   ', np.round(a, 3))

    print()
    print('payoffs poaching (regret)', np.array(do.payoffs_poaching).shape)
    regret_poaching = np.array(do.payoffs_poaching) - np.array(do.payoffs_poaching).max(axis=0)
    for p in regret_poaching:
        print('   ', np.round(p, 2))
        
    print()
    print('payoffs logging (regret)', np.array(do.payoffs_logging).shape)
    regret_logging = np.array(do.payoffs_logging) - np.array(do.payoffs_logging).max(axis=0)
    for p in regret_logging:  
        print('   ', np.round(p, 2))


    ##########################################
    # compare and visualize
    ##########################################
    print('----------- BASELINE MIDDLE -----------')
    baseline_middle_regrets_poaching = np.empty(n_perturb+1)
    baseline_middle_regrets_poaching[:] = np.nan
    baseline_middle_regrets_logging = np.empty(n_perturb+1)
    baseline_middle_regrets_logging[:] = np.nan
    for i in range(n_perturb+1):
        baseline_middle_distrib = np.zeros(len(do.agent_strategies))
        baseline_middle_distrib[baseline_middle_i+i] = 1
        baseline_middle_regrets_poaching[i] = do.compute_payoff_regret(baseline_middle_distrib)
        baseline_middle_regrets_logging[i] = do.compute_payoff_regret(baseline_middle_distrib, payoff_mode='logging')
    baseline_middle_regret_poaching = np.min(baseline_middle_regrets_poaching)
    baseline_middle_regret_logging = np.min(baseline_middle_regrets_logging)
    print('avg regret of baseline middle poaching {:.3f}'.format(baseline_middle_regret_poaching))
    print('avg regret of baseline middle logging {:.3f}'.format(baseline_middle_regret_logging))

    print('----------- BASELINE RANDOM REGRETS -----------')
    baseline_random_regrets_poaching = np.empty(n_perturb+1)
    baseline_random_regrets_poaching[:] = np.nan
    baseline_random_regrets_logging = np.empty(n_perturb+1)
    baseline_random_regrets_logging[:] = np.nan

    for i in range(n_perturb+1):
        baseline_random_distrib = np.zeros(len(do.agent_strategies))
        baseline_random_distrib[baseline_random_i+i] = 1
        baseline_random_regrets_poaching[i] = do.compute_payoff_regret(baseline_random_distrib)
        baseline_random_regrets_logging[i] = do.compute_payoff_regret(baseline_random_distrib, payoff_mode='logging')
    baseline_random_regret_poaching = np.min(baseline_random_regrets_poaching)
    baseline_random_regret_logging = np.min(baseline_random_regrets_logging)
    print('avg regret of baseline random poaching {:.3f}'.format(baseline_random_regret_poaching))
    print('avg regret of baseline random logging {:.3f}'.format(baseline_random_regret_logging))

    print('----------- BASELINE MAXIMIN REGRETS -----------')
    baseline_maximin_regrets_poaching = np.empty(n_perturb+1)
    baseline_maximin_regrets_poaching[:] = np.nan
    baseline_maximin_regrets_logging = np.empty(n_perturb+1)
    baseline_maximin_regrets_logging[:] = np.nan
    for i in range(n_perturb+1):
        baseline_maximin_distrib = np.zeros(len(do.agent_strategies))
        baseline_maximin_distrib[baseline_maximin_i+i] = 1
        baseline_maximin_regrets_poaching[i] = do.compute_payoff_regret(baseline_maximin_distrib)
        baseline_maximin_regrets_logging[i] = do.compute_payoff_regret(baseline_maximin_distrib, payoff_mode='logging')
    baseline_maximin_regret_poaching = np.min(baseline_maximin_regrets_poaching)
    baseline_maximin_regret_logging = np.min(baseline_maximin_regrets_logging)
    print('avg regret of baseline maximin {:.3f}'.format(baseline_maximin_regret_poaching))
    print('avg regret of baseline maximin {:.3f}'.format(baseline_maximin_regret_logging))

    print('----------- BASELINE RARL REGRETS -----------')
    baseline_RARL_regret_regrets_poaching = np.empty(n_perturb+1)
    baseline_RARL_regret_regrets_poaching[:] = np.nan
    baseline_RARL_regret_regrets_logging = np.empty(n_perturb+1)
    baseline_RARL_regret_regrets_logging[:] = np.nan
    for i in range(n_perturb+1):
        baseline_RARL_regret_distrib = np.zeros(len(do.agent_strategies))
        baseline_RARL_regret_distrib[baseline_RARL_regret_i+i] = 1
        baseline_RARL_regret_regrets_poaching[i] = do.compute_payoff_regret(baseline_RARL_regret_distrib)
        baseline_RARL_regret_regrets_logging[i] = do.compute_payoff_regret(baseline_RARL_regret_distrib, payoff_mode='logging')
    baseline_RARL_regret_regret_poaching = np.min(baseline_RARL_regret_regrets_poaching)
    baseline_RARL_regret_regret_logging = np.min(baseline_RARL_regret_regrets_logging)
    print('avg regret of baseline RARL_regret {:.3f}'.format(baseline_RARL_regret_regret_poaching))
    print('avg regret of baseline RARL_regret {:.3f}'.format(baseline_RARL_regret_regret_logging))
 
    # # My impression is that this is what we want to do for the primary objective
    # # But this might be wrong...
    # if do.objective == 'poaching':
    #     do_regret_poaching = -get_payoff(regret_poaching, agent_eq, nature_eq)
    #     print('avg regret of DO poaching {:.3f}'.format(do_regret_poaching))

    #     nature_eq_logging = get_nature_best_strategy(regret_logging, agent_eq)
    #     do_regret_logging = -get_payoff(regret_logging, agent_eq, nature_eq_logging)
    #     print('avg regret of DO logging {:.3f}'.format(do_regret_logging))
    # else:
    #     do_regret_logging = -get_payoff(regret_logging, agent_eq, nature_eq)
    #     print('avg regret of DO logging {:.3f}'.format(do_regret_logging))

    #     nature_eq_poaching = get_nature_best_strategy(regret_poaching, agent_eq)
    #     do_regret_poaching = -get_payoff(regret_poaching, agent_eq, nature_eq_poaching)
    #     print('avg regret of DO poaching {:.3f}'.format(do_regret_poaching))

    #########
    if do.objective == 'poaching':
        do_regret_poaching = -get_payoff(regret_poaching, agent_eq, nature_eq)
        print('avg regret of DO poaching {:.3f}'.format(do_regret_poaching))
    else:
        do_regret_logging = -get_payoff(regret_logging, agent_eq, nature_eq)
        print('avg regret of DO logging {:.3f}'.format(do_regret_logging))

    nature_br_secondary = do.nature_oracle_secondary.best_response(do.agent_strategies, agent_eq, display=False)
    do.update_payoffs_nature(nature_br_secondary, payoff_mode=do.secondary)
    # TODO: do I know for sure that this is working?

    if do.objective == 'poaching':  # calculate for secondary
        regret_logging = np.array(do.payoffs_logging) - np.array(do.payoffs_logging).max(axis=0)  # recalc because 
        secondary_eq = np.zeros(len(regret_logging[0]))
        secondary_eq[-1] = 1
        do_regret_logging = -get_payoff(regret_logging, agent_eq, secondary_eq)
    else:
        regret_poaching = np.array(do.payoffs_poaching) - np.array(do.payoffs_poaching).max(axis=0)
        secondary_eq = np.zeros(len(regret_poaching[0]))
        secondary_eq[-1] = 1
        do_regret_poaching = -get_payoff(regret_poaching, agent_eq, secondary_eq)


    # maybe do this
    # mixed_strategy = get_nature_best_strategy(payoffs, agent_eq)
    # -get_payoff(regret_logging, agent_eq, nature_eq_logging)

    # milind seems to want me to do this:
    # nature_br = self.nature_oracle.best_response(self.agent_strategies, agent_eq, display=False)
    # calculate agent payoffs
    # then calculate the expected regret I think

    # FIXME this is not quite the right nature_eq because it's assuming a different agent_eq
    # _, nature_eq_logging = do.find_equilibrium(payoff_mode='logging')
    # do_regret_logging = -get_payoff(regret_logging, agent_eq, nature_eq_logging)  # FIXME need diff nature_eq
    # print('avg regret of DO logging {:.3f}'.format(do_regret_logging))


    print('max_epochs {}, n_train agent {}, nature {}'.format(max_epochs, agent_n_train, nature_n_train))
    print('n_targets {}, horizon {}, budget {}'.format(do.n_targets, horizon, budget))

    bar_vals_poaching = [do_regret_poaching, baseline_middle_regret_poaching, baseline_random_regret_poaching, baseline_maximin_regret_poaching, baseline_RARL_regret_regret_poaching]
    bar_vals_logging = [do_regret_logging, baseline_middle_regret_logging, baseline_random_regret_logging, baseline_maximin_regret_logging, baseline_RARL_regret_regret_logging]
    # bar_vals_poaching = [do_regret_poaching, baseline_middle_regret_poaching, baseline_random_regret_poaching]#, baseline_maximin_regret_poaching, baseline_RARL_regret_regret]
    tick_names = ('double oracle', 'baseline middle', 'baseline random', 'baseline maximin', 'baseline RARL regret')

    print('regrets', tick_names)
    print(np.round(bar_vals_poaching, 3))

    now = datetime.now()
    str_time = now.strftime('%d-%m-%Y_%H:%M:%S')

    filename = '{}double_oracle.csv'.format(prefix)
    with open(filename, 'a') as f:
        if f.tell() == 0:
            print('creating file {}'.format(filename))
            f.write(("seed, n_targets, budget, horizon, do_regret_poaching,"
            "baseline_middle_regret_poaching, baseline_random_regret_poaching, baseline_maximin_regret_poaching, baseline_RARL_regret_regret_poaching,"
            "n_eval, agent_n_train, nature_n_train, max_epochs, n_perturb,"
            "max_interval, wildlife, deterrence, use_wake,"
            "freeze_policy_step, freeze_a_step, middle_time, maximin_time, do_time,"
            "time\n"))
        f.write((f"{seed}, {do.n_targets}, {budget}, {horizon}, {do_regret_poaching:.5f},"
        f"{baseline_middle_regret_poaching:.5f}, {baseline_random_regret_poaching:.5f}, {baseline_maximin_regret_poaching:.5f}, {baseline_RARL_regret_regret_poaching:.5f},"
        f"{n_eval}, {agent_n_train}, {nature_n_train}, {max_epochs}, {n_perturb},"
        f"{max_interval}, {wildlife_setting}, {deterrence_setting}, {use_wake},"
        f"{freeze_policy_step}, {freeze_a_step}, {middle_time}, {maximin_time}, {do_time},"
        f"{str_time}") +
        '\n')
    
    full_filename = '{}double_oracle_both.csv'.format(prefix)
    with open(full_filename, 'a') as f:
        if f.tell() == 0:
            print('creating file {}'.format(full_filename))
            f.write(("seed, n_targets, budget, horizon, do_regret_poaching, do_regret_logging,"
            "baseline_middle_regret_poaching, baseline_middle_regret_logging, baseline_random_regret_poaching, baseline_random_regret_logging, baseline_maximin_regret_poaching, baseline_maximin_regret_logging, baseline_RARL_regret_regret_poaching, baseline_RARL_regret_regret_logging,"
            "n_eval, agent_n_train, nature_n_train, max_epochs, n_perturb,"
            "max_interval, wildlife, deterrence, use_wake,"
            "freeze_policy_step, freeze_a_step, middle_time, maximin_time, do_time, objective, balance_attract"
            "time, is_toy\n"))
        f.write((f"{seed}, {do.n_targets}, {budget}, {horizon}, {do_regret_poaching:.5f}, {do_regret_logging:.5f},"
        f"{baseline_middle_regret_poaching:.5f}, {baseline_middle_regret_logging:.5f}, {baseline_random_regret_poaching:.5f}, {baseline_random_regret_logging:.5f}, {baseline_maximin_regret_poaching:.5f}, {baseline_maximin_regret_logging:.5f}, {baseline_RARL_regret_regret_poaching:.5f}, {baseline_RARL_regret_regret_logging:.5f},"
        f"{n_eval}, {agent_n_train}, {nature_n_train}, {max_epochs}, {n_perturb},"
        f"{max_interval}, {wildlife_setting}, {deterrence_setting}, {use_wake},"
        f"{freeze_policy_step}, {freeze_a_step}, {middle_time}, {maximin_time}, {do_time}, {do.objective}, {balance_attract},"
        f"{str_time}, {is_toy}") +
        '\n')


    do.print_agent_strategy(agent_eq, nature_eq, objective)
    do.print_agent_strategy(agent_eq, secondary_eq, secondary)


    x = np.arange(len(bar_vals_poaching))
    # plt.figure()
    # plt.bar(x, bar_vals_poaching)
    # plt.xticks(x, tick_names, rotation='vertical')
    # plt.xlabel('method')
    # plt.ylabel('avg regret')
    # plt.title('n_targets {}, budget {}, horizon {}, max_epochs {}'.format(do.n_targets, budget, horizon, max_epochs))
    # plt.savefig('plots/regret_n{}_b{}_h{}_epoch{}_{}.png'.format(do.n_targets, budget, horizon, max_epochs, str_time))
    plt.figure()
    for i in range(len(bar_vals_poaching)):
        plt.scatter(bar_vals_poaching[i], bar_vals_logging[i], label=tick_names[i])
    # plt.xticks(x, tick_names, rotation='vertical')
    # plt.yticks(y, tick_names)
    plt.legend()
    plt.xlabel('avg poaching regret')
    plt.ylabel('avg logging regret')
    plt.title('n_targets {}, budget {}, horizon {}, max_epochs {}'.format(do.n_targets, budget, horizon, max_epochs))
    plt.savefig('plots/regret_{}_n{}_b{}_h{}_epoch{}_{}.png'.format(do.objective, do.n_targets, budget, horizon, max_epochs, str_time))
    # plt.show()

    with open(exp_path + f'_strategies_{str_time}', 'wb') as f:
        pickle.dump([do.agent_strategies[:1] + do.agent_strategies[2:], do.nature_strategies_poaching, do.nature_strategies_logging], f)

    with open(exp_path + f'_payoffs_{str_time}', 'wb') as f:
        pickle.dump([regret_poaching, regret_logging], f)

    with open(exp_path + f'_eqs_{str_time}', 'wb') as f:
        pickle.dump([agent_eq, nature_eq], f)

