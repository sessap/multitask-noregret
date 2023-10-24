import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import shutil

if shutil.which('latex') is not None:
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
plt.style.use('ggplot')


def plot_regret_online_learning(ax, data, algo):
    colors = {'indep': 'black', 'single': 'blue', 'naive': 'gray', 'improved': 'red', 'AdaMT-UCB': 'green'}
    markers = {'indep': 'x', 'single': 'v', 'naive': 's', 'improved': 'o', 'AdaMT-UCB': 'd'}
    cum_regrets = np.cumsum(data['regrets'], axis=1)
    mean = np.mean(cum_regrets, axis = 0)
    std = np.std(cum_regrets, axis = 0)
    ax.plot(np.mean(cum_regrets, axis = 0), linewidth=2, label=algo, color=colors[algo], marker=markers[algo], markevery=int(1.5*len(mean)/10), markersize=5)
    T = np.size(cum_regrets,1)
    ax.fill_between(range(T), mean-0.5*std, mean+0.5*std, 
                     alpha = 0.1, label='_nolegend_', color=colors[algo])
    ax.set_xlabel(r'$T$')
    #ax.set_ylabel(r'$R(T)$')


def plot_regret_active_learning(ax, data, algo):
    linestyle ='-'
    if 'AE-LSVI' in algo:
        color = 'purple'
        marker = 'd'
    elif 'MTS' in algo:
        color = 'orange'
        marker = 'v'
    elif 'naive' in algo:
        color = 'gray'
        marker = 's'
    elif 'improved' in algo:
        color = 'red'
        marker = 'o'
    if 'unif' in algo:
        marker = 'none'
        if 'improved' in algo: 
            linestyle = '--'
        if 'naive' in algo: 
            linestyle = ':'
    cum_regrets = np.cumsum(data['regrets'], axis=1)
    #cum_regrets = data['regrets']
    mean = np.mean(cum_regrets, axis = 0)
    std = np.std(cum_regrets, axis = 0)
    ax.plot(np.mean(cum_regrets, axis = 0), linewidth=2, label=algo, linestyle=linestyle, color=color, marker=marker, markevery=int(1.5*len(mean)/10), markersize=5)
    T = np.size(cum_regrets,1)
    ax.fill_between(range(T), mean-0.5*std, mean+0.5*std, 
                     alpha = 0.1, label='_nolegend_', color=ax.lines[-1].get_color())
    ax.set_xlabel(r'$T$')
    #ax.set_ylabel(r'$R_{AL}(T)$')


def plot_task_activations(ax2, data, algo, idx_select):
    hatch = None
    if 'AE-LSVI' in algo:
        idx = 2
        color = 'purple'
        marker = 'd'
    elif 'MTS' in algo:
        idx = 3
        color = 'orange'
        marker = 'v'
    elif 'naive' in algo:
        return
        color = 'gray'
        marker = 's'
    elif 'improved' in algo:
        idx = 1
        color = 'red'
        marker = 'o'
    if 'unif' in algo:
        color = 'gray'
        idx = 0
        hatch = None
        marker = 'none'
        if 'improved' in algo: 
            linestyle = '--'
        if 'naive' in algo: 
            linestyle = ':'
    N = data['N']
    frequencies = np.zeros(N)
    activations = []
    for _ in idx_select:
        activations.extend(data['tasks_sequence'][_])
    for i in range(N):
        frequencies[i] = activations.count(i)
    frequencies = frequencies / np.sum(frequencies)
    width = 0.15
    ax2.bar(np.arange(N)+ idx*width, frequencies, color=color, width=width, hatch=hatch, label=algo, alpha=0.5)


def plot_confidence_regions(ax, true_y, pred_mean, lcbs_naive, ucbs_naive, lcbs, ucbs, task_id, t):
    idx_sort = np.arange(len(true_y))
    idx_sort = idx_sort[::200]
    ax.plot(true_y[idx_sort], ':', color='black', label='true')
    ax.plot(pred_mean[idx_sort], color='black', label='pred. mean')
    ax.fill_between(np.arange(len(idx_sort)), 
                    lcbs_naive.numpy()[idx_sort], 
                    ucbs_naive.numpy()[idx_sort], 
                    alpha=.25, color='gray', label='naive conf.')
    ax.fill_between(np.arange(len(idx_sort)), 
                    lcbs.numpy()[idx_sort], 
                    ucbs.numpy()[idx_sort], 
                    alpha=.2, color='red', label='improved conf.')
    ax.patch.set_facecolor('gainsboro')
    ax.patch.set_alpha(0.4)
    ax.set_title(f'task {task_id+1}')
    return