import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def plot_scatter():
    bar_vals_poaching = [do_regret_poaching, baseline_middle_regret_poaching, baseline_random_regret_poaching, baseline_maximin_regret_poaching, baseline_RARL_regret_regret_poaching]
    bar_vals_logging = [do_regret_logging, baseline_middle_regret_logging, baseline_random_regret_logging, baseline_maximin_regret_logging, baseline_RARL_regret_regret_logging]
    tick_names = ('double oracle', 'baseline middle', 'baseline random', 'baseline maximin', 'baseline RARL regret')
    x = np.arange(len(bar_vals_poaching))
    plt.figure()
    for i in range(len(bar_vals_poaching)):
        plt.scatter(bar_vals_poaching[i], bar_vals_logging[i], label=tick_names[i])
    plt.legend()
    plt.xlabel('avg poaching regret')
    plt.ylabel('avg logging regret')
    plt.title('n_targets {}, budget {}, horizon {}, max_epochs {}'.format(do.n_targets, budget, horizon, max_epochs))
    plt.savefig('plots/regret_{}_n{}_b{}_h{}_epoch{}_{}.png'.format(do.objective, do.n_targets, budget, horizon, max_epochs, str_time))


    pass


def get_rows(exp_timestrs):
    df = pd.read_csv('double_oracle_both.csv', sep=', ', engine='python')
    import pdb; pdb.set_trace()
    pass


if __name__ == '__main__':
    parser = ArgumentParser(description='Graphing for multi-MIRROR')
    parser.add_argument('--exp_timestrs', required=True, nargs='+')
    # parser.add_argument('--name', required=True)
    args = parser.parse_args()
    get_rows(args.exp_timestrs)


