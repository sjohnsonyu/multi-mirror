import pickle
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from double_oracle import DoubleOracle, get_payoff
from graphing_utils import get_rows, validate_comparable, graph_pareto
from tqdm import tqdm

CHECKPOINTS = [1, 50, 100, 500, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 80000, 100000, 120000, 150000, 170000]#, N_TRAIN-1]
PSI = 1.1 # wildlife growth ratio
ALPHA = .5  # strength that poachers eliminate wildlife
ETA = .3
BETA = -5

    
def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_relevant_strategies(poaching_timestr, logging_timestr, size):
    strategies_P_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_strategies_{poaching_timestr}'
    strategies_L_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_strategies_{logging_timestr}'    
    eqs_P_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_eqs_{poaching_timestr}'
    eqs_L_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_eqs_{logging_timestr}'
    payoffs_P_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_payoffs_{poaching_timestr}'
    payoffs_L_path = f'results/seed_0_height_{size}_width_{size}_max_interval_2.0_balance_attract_payoffs_{logging_timestr}'
    
    all_strategies_P = read_pickle(strategies_P_path)
    all_strategies_L = read_pickle(strategies_L_path)
    eqs_P = read_pickle(eqs_P_path)
    eqs_L = read_pickle(eqs_L_path)
    all_payoffs_P = read_pickle(payoffs_P_path)
    all_payoffs_L = read_pickle(payoffs_L_path)

    agent_eq_P = eqs_P[0]
    nature_eq_P = eqs_P[1]
    assert agent_eq_P[1] == 0, "Random policy was part of equilibrium in poaching-objective game"
    payoff_agent_inds_P = np.flatnonzero(agent_eq_P)
    payoff_nature_inds_P = np.flatnonzero(nature_eq_P)
    strategies_inds_P = np.where(payoff_agent_inds_P > 1, payoff_agent_inds_P - 1, payoff_agent_inds_P)
    strategies_P = [all_strategies_P[0][i] for i in range(len(all_strategies_P[0])) if i in strategies_inds_P]
    # do we use payoffs here or recalculate them? Maybe just use these ones for now?

    agent_eq_L = eqs_L[0]
    nature_eq_L = eqs_L[1]
    assert agent_eq_L[1] == 0, "Random policy was part of equilibrium in logging-objective game"
    payoff_agent_inds_L = np.flatnonzero(agent_eq_L)
    payoff_nature_inds_L = np.flatnonzero(nature_eq_L)
    strategies_inds_L = np.where(payoff_agent_inds_L > 1, payoff_agent_inds_L - 1, payoff_agent_inds_L)
    strategies_L = [all_strategies_L[0][i] for i in range(len(all_strategies_L[0])) if i in strategies_inds_L]

    all_agent_strategies = strategies_P + strategies_L
    eq_P = np.zeros(len(all_agent_strategies))
    eq_P[:len(strategies_P)] = agent_eq_P[payoff_agent_inds_P]

    eq_L = np.zeros(len(all_agent_strategies))
    eq_L[len(strategies_P):] = agent_eq_L[payoff_agent_inds_L]
    
    return all_agent_strategies, eq_P, eq_L

    # we will have the set of strategies
    # then we need to play around and see what happens when we insert different equilibria
    # I think maybe we then instantiate the DoubleOracle module?

def get_poaching_logging_regrets(all_agent_strategies,
                                 agent_eq,
                                 size,
                                 budget,
                                 horizon,
                                 nature_n_train,
                                 max_interval,
                                 is_toy,
                                 is_paws,
                                 exp_name,
                                 seed=0
                                ):
    # exp_name = f'seed_0_height_{size}_width_{size}_max_interval_{max_interval}_balance_attract'
    initialization_path = f'initialization_vals/' + exp_name
    if is_toy:
        initialization_path = 'initialization_vals/toy'
    if is_paws:
        initialization_path = 'initialization_vals/paws_mini'
    
    torch.manual_seed(seed)
    np.random.seed(seed)
                      
    do = DoubleOracle(max_epochs=0,
                      height=size,
                      width=size,
                      budget=budget,
                      horizon=horizon,
                      n_perturb=0,
                      n_eval=50,
                      agent_n_train=0,
                      nature_n_train=nature_n_train,
                      psi=PSI,
                      alpha=ALPHA,
                      beta=BETA,
                      eta=ETA,
                      max_interval=max_interval,
                      wildlife_setting=1,
                      use_wake=True,
                      checkpoints=CHECKPOINTS,
                      freeze_policy_step=5,
                      freeze_a_step=5,
                      initialization_path=initialization_path,
                      write_initialization=False,
                      read_initialization=True,
                      objective="poaching",
                      verbose=False
                      )
    # TODO: what to use for agent_eq?
    do.agent_strategies = all_agent_strategies.copy()

    nature_br_poaching = do.nature_oracle.best_response(all_agent_strategies, agent_eq, display=False)
    nature_br_logging = do.nature_oracle_secondary.best_response(do.agent_strategies, agent_eq, display=False)
    
    agent_opt_strategy_poaching = do.agent_oracle.best_response([nature_br_poaching], [1], "poaching", display=False)
    agent_opt_strategy_logging = do.agent_oracle_secondary.best_response([nature_br_logging], [1], "logging", display=False)

    do.nature_strategies_poaching = []
    do.nature_strategies_logging = []
    do.payoffs_poaching = [[] for _ in range(len(all_agent_strategies))]
    do.payoffs_logging = [[] for _ in range(len(all_agent_strategies))]

    do.update_payoffs(nature_br_poaching, agent_opt_strategy_poaching, payoff_mode='poaching')
    do.update_payoffs(nature_br_logging, agent_opt_strategy_logging, payoff_mode='logging')

    # print('agent_eq', agent_eq)
    # do.print_agent_strategy(agent_eq, [1, 0], "poaching")
    # return
    # do.print_agent_strategy(agent_eq, [0, 1], "logging")
    
    regret_poaching = np.array(do.payoffs_poaching) - np.array(do.payoffs_poaching).max(axis=0)
    regret_logging = np.array(do.payoffs_logging) - np.array(do.payoffs_logging).max(axis=0)

    
    do_regret_poaching = -get_payoff(regret_poaching, np.append(agent_eq, [0, 0]), [1.])
    do_regret_logging = -get_payoff(regret_logging, np.append(agent_eq, [0, 0]), [1.])

    # now log these puppies!
    print('agent_eq', agent_eq)
    print('do_regret_poaching', do_regret_poaching)
    print('do_regret_logging', do_regret_logging)

    return do_regret_poaching, do_regret_logging


def get_agent_eqs(agent_strategies):
    """[1, 0, 0, 0]
    [0.75, 0.25, 0, 0] x 3 x 4
    [0.5, 0.25, 0.25, 0] x 3 x 4
    [0.25, 0.25, 0.25, 0.25]
    I should implement a function that takes in policies, original mixture, and then generates this csv!
    [1, 0, 0] x 3
    [0.75, 0.25, 0] x 6, [0.75, 0.125, 0.125] x 3
    [0.5, 0.25, 0.25]
    """
    agent_eqs = []
    num_strategies = len(agent_strategies)
    if num_strategies == 3:
        for i in range(3):
            onehot_eq = np.zeros(num_strategies)
            onehot_eq[i] = 1
            agent_eqs.append(onehot_eq)

            quarters_split_eq = np.array([0.125, 0.125, 0.125])
            quarters_split_eq[i] = 0.75
            agent_eqs.append(quarters_split_eq)
            half_split_eq = np.array([0.25, 0.25, 0.25])
            half_split_eq[i] = 0.75
            agent_eqs.append(half_split_eq)
    
            for j in range(num_strategies):
                if i == j: continue
                quarters_two_eq = np.zeros(num_strategies)
                quarters_two_eq[i] = 0.75
                quarters_two_eq[j] = 0.25
                agent_eqs.append(quarters_two_eq)
    elif num_strategies == 4:
        agent_eqs.append(np.array([0.25, 0.25, 0.25, 0.25]))
        for i in range(num_strategies):
            onehot_eq = np.zeros(num_strategies)
            onehot_eq[i] = 1
            agent_eqs.append(onehot_eq)

            for j in range(num_strategies):
                if i == j: continue
                quarters_split_eq = np.zeros(num_strategies)
                quarters_split_eq[i] = 0.75
                quarters_split_eq[j] = 0.25
                agent_eqs.append(quarters_split_eq)

                print("skipping for speed!")
                # quarters_two_eq = np.array([0.25, 0.25, 0.25, 0.25])
                # quarters_two_eq[i] = 0.75
                # quarters_two_eq[j] = 0
                # agent_eqs.append(quarters_two_eq)
    return agent_eqs


def get_poaching_logging_regrets_variance(all_agent_strategies,
                                 agent_eq,
                                 size,
                                 budget,
                                 horizon,
                                 nature_n_train,
                                 max_interval,
                                 is_toy,
                                 is_paws,
                                 exp_name,
                                 seed=0
                                ):
    # exp_name = f'seed_0_height_{size}_width_{size}_max_interval_{max_interval}_balance_attract'
    initialization_path = f'initialization_vals/' + exp_name
    if is_toy:
        initialization_path = 'initialization_vals/toy'
    if is_paws:
        initialization_path = 'initialization_vals/paws_mini'

    do = DoubleOracle(max_epochs=0,
                      height=size,
                      width=size,
                      budget=budget,
                      horizon=horizon,
                      n_perturb=0,
                      n_eval=100,
                      agent_n_train=0,
                      nature_n_train=nature_n_train,
                      psi=PSI,
                      alpha=ALPHA,
                      beta=BETA,
                      eta=ETA,
                      max_interval=max_interval,
                      wildlife_setting=1,
                      use_wake=True,
                      checkpoints=CHECKPOINTS,
                      freeze_policy_step=5,
                      freeze_a_step=5,
                      initialization_path=initialization_path,
                      write_initialization=False,
                      read_initialization=True,
                      objective="poaching",
                      verbose=False
                      )
    # TODO: what to use for agent_eq?
    do.agent_strategies = all_agent_strategies.copy()
    poaching_regrets_no_max = []
    logging_regrets_no_max = []
    poaching_regrets = []
    logging_regrets = []
    for seed in range(20):
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # this is the first variable
        nature_br_poaching = do.nature_oracle.best_response(all_agent_strategies, agent_eq, display=False)
        nature_br_logging = do.nature_oracle_secondary.best_response(do.agent_strategies, agent_eq, display=False)

        agent_opt_strategy_poaching = do.agent_oracle.best_response([nature_br_poaching], [1], "poaching", display=False)
        agent_opt_strategy_logging = do.agent_oracle_secondary.best_response([nature_br_logging], [1], "logging", display=False)

        do.nature_strategies_poaching = []
        do.nature_strategies_logging = []
        do.payoffs_poaching = [[] for _ in range(len(all_agent_strategies))]
        do.payoffs_logging = [[] for _ in range(len(all_agent_strategies))]
        do.agent_strategies = do.agent_strategies[:4]
        
        do.update_payoffs(nature_br_poaching, agent_opt_strategy_poaching, payoff_mode='poaching')
        do.update_payoffs(nature_br_logging, agent_opt_strategy_logging, payoff_mode='logging')

        # this is the second variable
        # do.nature_strategies_poaching = []
        # do.nature_strategies_logging = []
        # do.payoffs_poaching = [[] for _ in range(len(all_agent_strategies))]
        # do.payoffs_logging = [[] for _ in range(len(all_agent_strategies))]
    
        # do.update_payoffs_nature(nature_br_logging, payoff_mode='logging')
        # do.update_payoffs_nature(nature_br_poaching, payoff_mode='poaching')

        regret_poaching = np.array(do.payoffs_poaching) - np.array(do.payoffs_poaching).max(axis=0)
        regret_logging = np.array(do.payoffs_logging) - np.array(do.payoffs_logging).max(axis=0)

        do_regret_poaching = -get_payoff(regret_poaching, np.append(agent_eq, [0, 0]), [1.])
        do_regret_logging = -get_payoff(regret_logging, np.append(agent_eq, [0, 0]), [1.])

        poaching_regrets.append(do_regret_poaching)
        logging_regrets.append(do_regret_logging)

        regret_poaching_no_max = np.array(do.payoffs_poaching)
        regret_logging_no_max = np.array(do.payoffs_logging)

        # NOTE: should I just be caching the do.payoffs_poaching without doing the regret step?
        do_regret_poaching_no_max = -get_payoff(regret_poaching_no_max, np.append(agent_eq, [0, 0]), [1.])
        do_regret_logging_no_max = -get_payoff(regret_logging_no_max, np.append(agent_eq, [0, 0]), [1.])

        poaching_regrets_no_max.append(do_regret_poaching_no_max)
        logging_regrets_no_max.append(do_regret_logging_no_max)

    print(poaching_regrets)
    print(logging_regrets)
    data = np.array([poaching_regrets, logging_regrets]).T
    np.savetxt(f'regret_variance_analysis_simulate_reward_agent_br.txt', data)
    data = np.array([poaching_regrets_no_max, logging_regrets_no_max]).T
    np.savetxt(f'regret_variance_analysis_simulate_reward_agent_br_no_max.txt', data)
        # import matplotlib.pyplot as plt
        # plt.boxplot(data)
        # plt.savefig(f'regret_variance_analysis_simulate_reward_{n_eval}_no_max.png')
    

if __name__ == '__main__':
    parser = ArgumentParser(description='Pareto curve evaluation for multi-MIRROR')
    parser.add_argument('--exp_timestrs', required=True, nargs='+')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    exp_timestrs = args.exp_timestrs
    seed = args.seed
    poaching_row, logging_row = get_rows(exp_timestrs)
    validate_comparable(poaching_row, logging_row)
    
    size = int(np.sqrt(poaching_row['n_targets'].values[0]))
    budget = poaching_row['budget'].values[0]
    horizon = poaching_row['horizon'].values[0]
    nature_n_train = poaching_row['nature_n_train'].values[0]
    max_interval = poaching_row['max_interval'].values[0]
    poaching_timestr = poaching_row['time'].values[0]
    logging_timestr = logging_row['time'].values[0]
    is_toy = poaching_row['is_toy'].values[0]
    assert is_toy == logging_row['is_toy'].values[0]
    is_paws = poaching_row['is_paws'].values[0]
    assert is_paws == logging_row['is_paws'].values[0]
    exp_seed = poaching_row['seed'].values[0]
    assert exp_seed == logging_row['seed'].values[0]
    exp_name = f'seed_{exp_seed}_height_{size}_width_{size}_max_interval_{max_interval}_balance_attract'

    agent_strategies, eq_P, eq_L = get_relevant_strategies(poaching_timestr, logging_timestr, size)
    # agent_eqs = get_agent_eqs(agent_strategies)

    data = defaultdict(list)
    
    agent_eq = [.75, 0, 0, .25]
    poaching_regret, logging_regret = get_poaching_logging_regrets_variance(agent_strategies,
                                                                agent_eq,
                                                                size,
                                                                budget,
                                                                horizon,
                                                                nature_n_train,
                                                                max_interval,
                                                                is_toy,
                                                                is_paws,
                                                                exp_name,
                                                                seed)
    print('done')

    # for agent_eq in tqdm(agent_eqs + [eq_P, eq_L]):
    # for agent_eq in [[1, 0, 0, 0], [.75, 0, 0, .25], [.25, 0 ,0, .75], [0, 0, 0, 1]]:
    # for agent_eq in [[.75, 0, 0, .25], [.25, 0 ,0, .75], [0, 0, 0, 1]]:
    # poaching_regrets = []
    # logging_regrets = []
    # for seed in range(20):
        # poaching_regret, logging_regret = get_poaching_logging_regrets(agent_strategies,
        #                                                                 agent_eq,
        #                                                                 size,
        #                                                                 budget,
        #                                                                 horizon,
        #                                                                 nature_n_train,
        #                                                                 max_interval,
        #                                                                 is_toy,
        #                                                                 is_paws,
        #                                                                 exp_name,
        #                                                                 seed
        #                                                                 )

        # for i, eq_val in enumerate(agent_eq):
        #     data[f'strategy_{i}'].append(eq_val)
        # data['poaching_regret'].append(poaching_regret)
        # data['logging_regret'].append(logging_regret)
        # data['eq_str'].append(str(list(np.round(agent_eq, 2))))

    
    # df = pd.DataFrame(data)
    # df.to_csv(f'results/pareto_seed_{seed}_{poaching_timestr}_{logging_timestr}.csv', index=False)
    
    # graph_pareto(poaching_timestr, logging_timestr, seed)


