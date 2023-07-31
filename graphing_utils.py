import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser

REGRET_COLS = ['do_regret_poaching', 'baseline_middle_regret_poaching', 'baseline_random_regret_poaching', 'baseline_maximin_regret_poaching', 'baseline_RARL_regret_regret_poaching', 'do_regret_logging', 'baseline_middle_regret_logging', 'baseline_random_regret_logging', 'baseline_maximin_regret_logging', 'baseline_RARL_regret_regret_logging']


def plot_scatter(poaching_row, logging_row, exp_timestrs):
    bar_vals_poaching = [poaching_row['do_regret_poaching'].values[0],
                         logging_row['do_regret_poaching'].values[0],
                         poaching_row['baseline_middle_regret_poaching'].values[0],
                         logging_row['baseline_middle_regret_poaching'].values[0],
                         poaching_row['baseline_random_regret_poaching'].values[0],
                        #  logging_row['baseline_random_regret_poaching'].values[0],
                         poaching_row['baseline_maximin_regret_poaching'].values[0],
                         logging_row['baseline_maximin_regret_poaching'].values[0],
                         poaching_row['baseline_RARL_regret_regret_poaching'].values[0],
                         logging_row['baseline_RARL_regret_regret_poaching'].values[0]
    ]

    bar_vals_logging = [poaching_row['do_regret_logging'].values[0],
                        logging_row['do_regret_logging'].values[0],
                        poaching_row['baseline_middle_regret_logging'].values[0],
                        logging_row['baseline_middle_regret_logging'].values[0],
                        poaching_row['baseline_random_regret_logging'].values[0],
                       #  logging_row['baseline_random_regret_logging'].values[0],
                        poaching_row['baseline_maximin_regret_logging'].values[0],
                        logging_row['baseline_maximin_regret_logging'].values[0],
                        poaching_row['baseline_RARL_regret_regret_logging'].values[0],
                        logging_row['baseline_RARL_regret_regret_logging'].values[0]
    ]
    # tick_names = ('double oracle (P)',
    #               'double oracle (L)',
    #               'baseline middle (P)',
    #               'baseline middle (L)',
    #               'baseline random',
    #               'baseline maximin (P)',
    #               'baseline maximin (L)',
    #               'baseline RARL regret (P)',
    #               'baseline RARL regret (L)')
    tick_names = ('DO_P',
                  'DO_L',
                  'mid_P',
                  'mid_L',
                  'random',
                  'maximin_P',
                  'maximin_L',
                  'RARL_P',
                  'RARL_L')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(bar_vals_poaching)):
        # plt.scatter(bar_vals_poaching[i], bar_vals_logging[i], label=tick_names[i])
        plt.scatter(bar_vals_poaching[i], bar_vals_logging[i])
    # plt.scatter(bar_vals_poaching, bar_vals_logging)
    
    for i, label in enumerate(tick_names):
        ax.annotate(label, (bar_vals_poaching[i] + 0.01, bar_vals_logging[i] + 0.01))

    plt.xlabel('avg poaching regret')
    plt.ylabel('avg logging regret')
    title, n_targets, budget, horizon, max_epochs = get_title(poaching_row)
    plt.title(title)

    plt.savefig(f'plots/scatter_n_targets_{n_targets}_budget_{budget}_horizon_{horizon}_max_epochs_{max_epochs}_{exp_timestrs[0]}_{exp_timestrs[1]}.png')
    plt.xlim(-0.01, 1)
    plt.ylim(-0.01, 1)
    plt.savefig(f'plots/scatter_crop_n_targets_{n_targets}_budget_{budget}_horizon_{horizon}_max_epochs_{max_epochs}_{exp_timestrs[0]}_{exp_timestrs[1]}.png')


def get_title(poaching_row):
    n_targets = poaching_row['n_targets'].values[0]
    budget = poaching_row['budget'].values[0]
    horizon = poaching_row['horizon'].values[0]
    max_epochs = poaching_row['max_epochs'].values[0]
    return f'n_targets {n_targets}, budget {budget}, horizon {horizon}, max_epochs {max_epochs}', n_targets, budget, horizon, max_epochs


def get_rows(exp_timestrs):
    assert len(exp_timestrs) == 2
    df = pd.read_csv('double_oracle_both.csv', sep=', ', engine='python')
    for col in REGRET_COLS:
        df[col].astype(float)

    row0 = df[df['time'] == exp_timestrs[0]]
    row1 = df[df['time'] == exp_timestrs[1]]
    row0_objective = row0['objective'].values[0]
    row1_objective = row1['objective'].values[0]
    assert row0_objective != row1_objective
    poaching_row = row0 if row0_objective == 'poaching' else row1
    logging_row = row1 if row1_objective == 'logging' else row0
    return poaching_row, logging_row


def validate_comparable(poaching_row, logging_row):
    assert poaching_row['n_targets'].values[0] == logging_row['n_targets'].values[0]
    assert poaching_row['budget'].values[0] == logging_row['budget'].values[0]
    assert poaching_row['horizon'].values[0] == logging_row['horizon'].values[0]
    assert poaching_row['max_epochs'].values[0] == logging_row['max_epochs'].values[0]


def graph_pareto(poaching_timestr, logging_timestr, seed):
    filename = f'results/pareto_seed_{seed}_{poaching_timestr}_{logging_timestr}.csv'
    df = pd.read_csv(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(df['poaching_regret'], df['logging_regret'])
    
    for i, label in enumerate(df['eq_str']):
        ax.annotate(label, (df['poaching_regret'][i] + 0.01, df['logging_regret'][i] + 0.01))

    plt.xlabel('avg poaching regret')
    plt.ylabel('avg logging regret')
    plt.title('regret per equilibrium')

    out_filename = f'plots/pareto_seed_{seed}_{poaching_timestr}_{logging_timestr}.png'
    plt.savefig(out_filename)


if __name__ == '__main__':
    parser = ArgumentParser(description='Graphing for multi-MIRROR')
    parser.add_argument('--exp_timestrs', required=True, nargs='+')
    args = parser.parse_args()
    exp_timestrs = args.exp_timestrs
    poaching_row, logging_row = get_rows(exp_timestrs)
    validate_comparable(poaching_row, logging_row)
    plot_scatter(poaching_row, logging_row, exp_timestrs)

