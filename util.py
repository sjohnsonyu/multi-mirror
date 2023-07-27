import numpy as np


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


def create_pareto_curve(agent_strategies, agent_eq, nature_eq, regrets):
    """[1, 0, 0, 0]
    [0.75, 0.25, 0, 0] x 3 x 4
    [0.5, 0.25, 0.25, 0] x 3 x 4
    [0.25, 0.25, 0.25, 0.25]
    I should implement a function that takes in policies, original mixture, and then generates this csv!
    """
    

    
    if do.objective == 'poaching':
        do_regret_poaching = -get_payoff(regrets, agent_eq, nature_eq)
        print('avg regret of DO poaching {:.3f}'.format(do_regret_poaching))
    else:
        do_regret_logging = -get_payoff(regrets, agent_eq, nature_eq)
        print('avg regret of DO logging {:.3f}'.format(do_regret_logging))

    nature_br_secondary = do.nature_oracle_secondary.best_response(do.agent_strategies, agent_eq, display=False)
    do.update_payoffs_nature(nature_br_secondary, payoff_mode=do.secondary)

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



