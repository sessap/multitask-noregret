import numpy as np
import torch 
import random
import pickle
import argparse
from learning_epochs import *
from plotters import plot_regret_active_learning, plot_task_activations
from aux import *
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=3,
                        help='numnber of tasks')
    parser.add_argument('--d', type=int, default=4,
                        help='dimension')
    parser.add_argument('--tasks_dev', type=float, default=.2,
                        help='tasks deviation from the mean (\epsilon)')
    parser.add_argument('--R', type=float, default=1.0,
                        help='noise std.')
    parser.add_argument('--repeats', type=int, default=1,
                        help='number of repetitions for each problem')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of runs, each with a different random problem')
    parser.add_argument('--use_b', type=float, default=None,
                        help='set b hyperparameter')
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
    algo_list = ['uniform naive', 'active naive', 'uniform improved',
                 'active improved', 'MTS', 'AE-LSVI']
    algo_legend = ['Unif., naive', r'MT-AL, naive', 'Unif., improved confidence', 
                   r'MT-AL, improved confidence', 'MTS', 'AE-LSVI']

    use_b = None
    if args.use_b is not None:
        use_b = args.use_b
    
    if not args.dont_recompute:
        for algo in algo_list:
            data = run_experiment(algo, N, d, R, tasks_dev, T, N_runs, N_repeats, 
                                  use_b=use_b, 
                                  T_init=1,
                                  savedatafile=f'data_active/data_{algo}_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pkl',
                                  learning_epoch=active_learning_epoch,
                                  tasks_data=tasks_data)
    
    # Plot regrets and task activations
    fig, ax = plt.subplots(1,1,figsize=(3.3,3), dpi=100)
    fig2, ax2 = plt.subplots(1,1,figsize=(6,3), dpi=100)
    ymax_indep = 0
    ymax_naive = 0
    for _, algo in enumerate(algo_list):
        with open(f"{args.datapath}data_active/data_{algo}_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pkl", 'rb') as fil:
            data = pickle.load(fil)

            # Add histogram of number of tasks
            plot_task_activations(ax2, data, algo, idx_select=range(5))
            
            plot_regret_active_learning(ax, data, algo)
            #ax.legend(algo_legend)
            #if 'used_b' in data:
            #    ax.set_title(f'N={N}, d={d}, dev={tasks_dev}, R={R}, b={data["used_b"]}')
            #else:
            #    ax.set_title(f'N={N}, d={d}, dev={tasks_dev}, R={R}')
            if algo == 'uniform indep':
                cum_regrets = np.cumsum(data['regrets'], axis=1)
                mean = np.mean(cum_regrets, axis = 0)
                stds = np.std(cum_regrets, axis=0)
                ymax_indep =np.max(mean) + 0.5*stds[-1]
                ymin = np.min(mean)
            if algo == 'uniform naive':
                cum_regrets = np.cumsum(data['regrets'], axis=1)
                mean = np.mean(cum_regrets, axis = 0)
                ymax_naive =np.max(mean) #+ 0.5*np.std(mean) 
    ymax = max(ymax_indep, ymax_naive)
    ax.set_ylim(top=ymax*1.1, bottom=0.5)
    ax.set_xlim(left=-1, right=T)
    ax.patch.set_facecolor('gainsboro')
    ax.patch.set_alpha(0.4)
    if args.real_data:
        ax.set_title('Drug discovery')
    else:
        ax.set_title('Synthetic')
    fig.set_tight_layout(True)

    # Histograms
    ax2.patch.set_facecolor('gainsboro')
    ax2.patch.set_alpha(0.4) 
    if args.real_data:
        ax2.set_title('Drug discovery')
    else:
        ax2.set_title('Synthetic')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(np.arange(N)+0.2)
    ax2.set_xticklabels([f"task-{i+1}" for i in range(N)])
    ax2.tick_params(axis=u'both', which=u'both',length=0)
    ax2.legend(algo_legend[2:], ncols=5)
    fig2.set_tight_layout(True)

    if args.real_data:
        fig.savefig(f'figs_active/fig_real_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')
        fig2.savefig(f'figs_active/fig_hisrogram_real_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'figs_active/fig_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')
        fig2.savefig(f'figs_active/fig_histogram_N_{N}_d_{d}_dev_{tasks_dev}_R_{R}_b_{use_b}.pdf', bbox_inches='tight')

    figlegend= plt.figure(figsize=(.5, 5), dpi=100)
    legend = figlegend.legend(ax.get_lines(), algo_legend, loc='center', fontsize=18, columnspacing=0.8, handlelength=1, ncol=3)
    legend.get_frame().set_facecolor('whitesmoke')
    figlegend.set_tight_layout(True)
    figlegend.savefig(f'figs_active/legend.pdf', bbox_inches='tight')