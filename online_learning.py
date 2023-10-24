import numpy as np
import torch 
import random
import pickle
import argparse
from learning_epochs import *
from plotters import plot_regret_online_learning
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5,
                        help='numnber of tasks')
    parser.add_argument('--d', type=int, default=4,
                        help='dimension')
    parser.add_argument('--tasks_dev', type=float, default=.5,
                        help='tasks deviation from the mean (\epsilon)')
    parser.add_argument('--R', type=float, default=1.0,
                        help='noise std.')
    parser.add_argument('--repeats', type=int, default=1,
                        help='number of repetitions for each problem')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of runs, each with a different random problem')
    parser.add_argument('--use_b', type=float, default=None,
                        help='set b hyperparameter')
    parser.add_argument('--find_best_b', action='store_true', default=False,
                        help='find best b hyperparameter')
    parser.add_argument('--add_adamtucb', action='store_true', default=False,
                        help='run also AdaMT-UCB algorithm (takes longer)')
    parser.add_argument('--dont_recompute', action='store_true', default=False,
                        help='recompute data or just do plotting')
    parser.add_argument('--datapath', type=str, default='',
                        help='where to load saved data')
    parser.add_argument('--real_data', action='store_true', default=False,
                        help='whether to run on real iedb dataset')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tasks_data = None
    if args.real_data:
        tasks_data, N, d = load_iedb_tasks()
    else:
        N = args.N
        d = args.d
    tasks_dev = args.tasks_dev
    R = args.R 
    N_runs = args.runs
    N_repeats = args.repeats

    T = min(N*d*5, 600)
    algo_list = ['indep', 'single', 'naive', 'improved']
    algo_legend = ['Independent', 'Single', r'MT-UCB, naive (GoB.Lin)', 
                   r'MT-UCB, improved confidence']
   
    if args.add_adamtucb:
        algo_list.append('AdaMT-UCB')
        algo_legend.append('AdaMT-UCB')

    use_b = None
    if args.use_b is not None:
        use_b = args.use_b
    if args.find_best_b and args.use_b is None:
        algo = 'improved'
        print('---------------------------------------------')
        print('Finding best b hyperparameter...')
        print('---------------------------------------------')
        b_list = [0, 0.005, 0.005, 0.1, 0.2] #TODO: specify desired hyperparam list
        scores = []
        for b in b_list:
            data_temp = run_experiment(algo, N, d, R, tasks_dev, T,  N_runs=3, N_repeats=1, 
                                       use_b=b,
                                       T_init=1, 
                                       savedatafile=None, 
                                       learning_epoch=online_learning_epoch, 
                                       tasks_data=tasks_data)
            simple_regrets = data_temp['regrets']
            cum_regrets = np.cumsum(data_temp['regrets'], axis=1)
            mean_simple = np.mean(simple_regrets, axis = 0)
            mean_cum = np.mean(cum_regrets, axis = 0)
            scores.append(mean_cum[-1] + 1e4*np.mean(mean_simple[-N:]))
        use_b = b_list[np.argmin(scores)]
        print(f'best b found: {use_b} with regret {scores[np.argmin(scores)]}')

    if not args.dont_recompute:
        for algo in algo_list:
            data = run_experiment(algo, N, d, R, tasks_dev, T, N_runs, N_repeats,
                                  use_b=use_b,
                                  T_init=1,
                                  savedatafile=f'data_online/data_{algo}_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pkl',
                                  learning_epoch=online_learning_epoch,
                                  tasks_data=tasks_data)

    # Plot regrets
    fig, ax = plt.subplots(1,1,figsize=(3.3,3), dpi=100)
    ymax = None
    for algo in algo_list:
        with open(f"{args.datapath}data_online/data_{algo}_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pkl", 'rb') as fil:
            data = pickle.load(fil)
            plot_regret_online_learning(ax, data, algo)
            #ax.legend(algo_legend)
            #if 'used_b' in data:
            #    ax.set_title(f'N={N}, d={d}, dev={tasks_dev}, R={R}, b={data["used_b"]}')
            #else:
            #    ax.set_title(f'N={N}, d={d}, dev={tasks_dev}, R={R}')
            if algo=='indep':
                cum_regrets = np.cumsum(data['regrets'], axis=1)
                mean = np.mean(cum_regrets, axis = 0)
                stds =  np.std(cum_regrets, axis = 0)
                ymax =np.max(mean) + 0.5*stds[-1]
    if ymax is not None:
        ax.set_ylim(top=ymax*1, bottom=0.5)
    ax.set_xlim(left=-1, right=T)
    ax.patch.set_facecolor('gainsboro')
    ax.patch.set_alpha(0.4)
    if args.real_data:
        ax.set_title('Drug discovery')
    else:
        ax.set_title('Synthetic')
    fig.set_tight_layout(True)
    if args.real_data:
        fig.savefig(f'figs_online/fig_real_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'figs_online/fig_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')


    figlegend= plt.figure(figsize=(.5, 5), dpi=100)
    legend = figlegend.legend(ax.get_lines(), algo_legend, loc='center', fontsize=18, columnspacing=0.8, handlelength=1, ncol=3)
    legend.get_frame().set_facecolor('whitesmoke')
    figlegend.set_tight_layout(True)
    figlegend.savefig(f'figs_online/legend.pdf', bbox_inches='tight')