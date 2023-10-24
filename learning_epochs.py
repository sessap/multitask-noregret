import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import pickle
from models import *
from aux import *
from plotters import plot_confidence_regions


def run_experiment(algo, N:int, d:int, R:int, tasks_dev:float, T:int, N_runs:int, N_repeats:int,
                    use_b=None, T_init=1, savedatafile=None, learning_epoch=None, tasks_data=None):
    assert learning_epoch is not None, 'a learning_epoch must be specified'
    assert use_b is not None, 'kernel parameter b must be specified'
    seed_all(42)
    if tasks_data is None:
        test_x = domain(d, 10000)
    else:
        test_x = None
    data = {'algo': algo, 'N': N, 'd': d, 'R': R, 'T': T,'tasks_dev': tasks_dev}
    data['tasks_sequence'] = []
    data['rewards'] = []
    data['regrets'] = []
    for _ in range(N_repeats*N_runs):
        if tasks_data is None:
            task_vectors, f, f_avg = generate_tasks(N, d, dev=tasks_dev, seed=_%N_runs)
            B = max([np.linalg.norm(f.numpy(),2) for f in task_vectors])
            eps = np.max([np.linalg.norm(f.numpy( ) - f_avg, 2)/B for f in task_vectors ])
        else:
            task_vectors = None
            B = 5
            eps = tasks_dev
        params = {"N": N, "d": d, "T": T, "R": R, "B": B, "eps": eps}
        # Choose kernel parameter b
        if 'indep' in algo:
            b = 0
            lam = 1
        elif 'single' in algo:
            b = 1e8
            lam = 1/N
        else:
            b = use_b
            A, A_half, A_inv = compute_aux_matrices(N, d, b)
            lam = A_inv[0,0]
        lam = lam*R**2
        print('-----------------------------------------------------')
        print(f'Params,  B: {B}, eps: {eps}, b: {b}, lam: {lam}')
        print('-----------------------------------------------------')
        
        # Initialize datasets
        seed_all(_)
        full_train_x = torch.empty(0)
        full_train_i = torch.empty(0)
        full_train_y = torch.empty(0)
        for i in range(N):
            for j in range(T_init):
                if tasks_data is None:
                    train_xi = domain(d, 1)
                    train_i_taski = torch.full((train_xi.shape[0],1), 
                                            dtype=torch.long, fill_value=i)
                    train_yi = torch.matmul(train_xi, task_vectors[i]).squeeze() 
                    + torch.randn(1) * R
                else:
                    idx_random = np.random.choice(len(tasks_data['domains'][i]))
                    train_xi = tasks_data['domains'][i][idx_random].unsqueeze(dim=0)
                    train_i_taski = torch.full((train_xi.shape[0],1), 
                                            dtype=torch.long, fill_value=i)
                    train_yi = tasks_data['labels'][i][idx_random].unsqueeze(0) 
                    + torch.randn(1) * 0.01
                full_train_x = torch.cat([full_train_x, train_xi.reshape(1,d)], axis=0)
                full_train_i = torch.cat([full_train_i, train_i_taski.reshape(1,1)])
                full_train_y = torch.cat([full_train_y, train_yi])

        rewards, regrets, tasks_sequence = learning_epoch(_, algo, N, d, R, T,
                                                        full_train_x, full_train_i, full_train_y, 
                                                        test_x, task_vectors, b, lam, B, eps,
                                                        tasks_data=tasks_data)
        data['rewards'].append(rewards)
        data['regrets'].append(regrets)
        data['tasks_sequence'].append(tasks_sequence)

    if use_b is not None:
        data['used_b'] = use_b
    if savedatafile:
        with open(savedatafile, 'wb') as fil:
            pickle.dump(data, fil)
    return data


def online_learning_epoch(run_id, algo, N, d, R, T,
                            full_train_x, full_train_i, full_train_y, 
                            test_x, task_vectors, b, lam, B, eps, tasks_data=None):
    if algo == 'AdaMT-UCB':
        eps_list = [.1, .2, .3, .5, .7, .9]
        beta_factor_list = [1.]
        params_list = [[eps, beta_factor] for eps in eps_list 
                        for beta_factor in beta_factor_list]
        eps, beta_factor = params_list[0]
        U_cum = 0
        R_cum = 0
        L_cum = np.zeros(len(params_list))
        print(f'Chosen eps: {eps} (implying b: {b}, lam: {lam}), beta_factor: {beta_factor}')

    random.seed(run_id)
    tasks_sequence = np.random.choice(N,T)
    print(tasks_sequence)
    rewards = []
    regrets = []
    #fig, axs = plt.subplots(1, N, figsize=(N*3,3))
    for t in range(T):
        i_t = tasks_sequence[t]

        # Initialize model and set-it to eval mode
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
        model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, 
                                 likelihood, N, lam, R, B, eps, b)
        likelihood.initialize(noise=lam)
        model.covar_module.initialize(variance=1)
        model.task_covar_module.initialize(
            covar_factor = np.sqrt(b/((1+b)*N))*torch.ones((N,1)), 
            raw_var=np.log(1/(1+b))*torch.ones(N))

        if 0:
            optimize_params(model, likelihood, model.covar_module.parameters(), 
                            full_train_x, full_train_i, full_train_y)
        # Set into eval mode
        model.eval()
        likelihood.eval()
        ##########

        assert beta_t_naive(model=model)[i_t] > -1e-3 + beta_t_improved(model=model)[i_t], f"ERROR,  beta naive:{beta_t_naive(model=model)[i_t]}, improved: {beta_t_improved(model=model)[i_t]}"
        
        if algo == 'naive':
            beta_t = beta_t_naive(model=model)[i_t]
        elif algo == 'improved' or algo == 'indep' or algo == 'single':
            beta_t = beta_t_improved(model=model)[i_t]
        elif algo == 'AdaMT-UCB':
            beta_t = beta_factor*beta_t_improved(model=model)[i_t]
        with torch.no_grad():
            if tasks_data:
                test_x = tasks_data['domains'][i_t]
            test_i_t = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=i_t)

            observed_pred_y = likelihood(model(test_x, test_i_t))
            ucbs = observed_pred_y.mean + beta_t*torch.sqrt(observed_pred_y.variance)
            lcbs = observed_pred_y.mean - beta_t*torch.sqrt(observed_pred_y.variance)
            idx_t = ucbs.argmax().numpy()
            x_t = test_x[idx_t]
            
            if tasks_data is None:
                r_t = torch.matmul(x_t, task_vectors[i_t])
                r_max_t = torch.matmul(test_x, task_vectors[i_t]).max().numpy()
                r_min_t = torch.matmul(test_x, task_vectors[i_t]).min().numpy()
                y_t = r_t + torch.randn(1) * R
            else:
                r_t = tasks_data['labels'][i_t][idx_t]
                r_max_t = tasks_data['labels'][i_t].max().numpy()
                r_min_t = tasks_data['labels'][i_t].min().numpy()
                y_t = r_t + torch.randn(1) * 0.01
            reward = r_t.numpy().squeeze()
            regret = r_max_t - reward
            sigma_t = torch.sqrt(observed_pred_y.variance[idx_t]).squeeze().numpy()
           
        if 0 and t%10 == 0:
            axs[i_t].plot(lcbs.numpy())
            axs[i_t].plot(ucbs.numpy())
        if t%10 == 0:
            print(f"algo: {algo}, run: {run_id}, t: {t}, beta_t: {beta_t}, regret: {regret:.4f}, reward: {reward:.4f}")

        if algo == 'AdaMT-UCB':
            # Perform misspecification test
            U_cum += y_t.squeeze().numpy()
            R_cum += min(2*beta_t*sigma_t, r_max_t-r_min_t)
            eps_prev = None
            for i, (eps_i, beta_factor_i) in enumerate(params_list):
                if eps_i is not eps_prev:
                    b_i = b 
                    A_i, A_half_i, A_inv_i = compute_aux_matrices(N, d, b_i)
                    lam_i = A_inv_i[0,0]
                    # Initialize model and set-it to eval mode
                    likelihood =  gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
                    model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood, N, lam_i, R, B, eps_i, b_i)
                    likelihood.initialize(noise=lam_i)
                    model.covar_module.initialize(variance=1)
                    model.task_covar_module.initialize(covar_factor = np.sqrt(b_i/((1+b_i)*N))*torch.ones((N,1)), raw_var=np.log(1/(1+b_i))*torch.ones(N))
                    if 0:
                        optimize_params(model, likelihood, 
                                        [{"params_mean": model.mean_module.parameters()}, {"params_covar": model.covar_module.parameters()}],
                                        full_train_x, full_train_i, full_train_y)
                    # Set into eval mode
                    model.eval()
                    likelihood.eval()
                    with torch.no_grad():
                        observed_pred_y = likelihood(model(test_x, test_i_t))
                    ##########
                eps_prev = eps_i
                beta_t = beta_factor_i*beta_t_improved(model=model)[i_t]
                with torch.no_grad():
                    lcb_t = observed_pred_y.mean[idx_t] - beta_t*torch.sqrt(observed_pred_y.variance[idx_t])
                    L_cum += max(lcb_t.squeeze().numpy(), r_min_t)
                    break

            # Misspecification test
            if t%10 == 0:
                print(f"{U_cum} + {R_cum} <? {max(L_cum)}")
            if U_cum + R_cum < max(L_cum) :
                print(f'\nMISSPECIFICATION TEST TRIGGERED, removed (eps, beta_factor): {eps, beta_factor}\n')
                params_list = [(e, b) for (e, b) in params_list if e > eps or b > beta_factor]
                U_cum = 0
                R_cum = 0
                L_cum = np.zeros(len(params_list))
                
                idx_next = np.argmin([e**2*b for (e, b) in params_list])
                eps, beta_factor = params_list[idx_next]
                b = b #b_fun(eps, params)
                A, A_half, A_inv = compute_aux_matrices(N, d, b)
                lam = A_inv[0,0]
                print(f'Chosen eps: {eps}, beta_factor: {beta_factor}, implying b: {b}, lam: {lam}')
        
        full_train_x = torch.cat([full_train_x, x_t.reshape(1,d)], axis=0)
        full_train_i = torch.cat([full_train_i, torch.tensor(i_t).reshape(1,1)])
        full_train_y = torch.cat([full_train_y, y_t])
        rewards.append(reward)
        regrets.append(regret)
    return rewards, regrets, tasks_sequence


def active_learning_epoch(run_id, algo, N, d, R, T, 
                            full_train_x, full_train_i, full_train_y, 
                            test_x, task_vectors, b, lam, B, eps, tasks_data=None):
    rewards = []
    regrets = []
    tasks_sequence = []
    x_idx_sequence = []
    for i in range(N):
        x_idx_sequence.append([])

    #fig, axs = plt.subplots(1, N, figsize=(N*3,3))
    for t in range(T):
        # Initialize model and set-it to eval mode
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
        model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, 
                                likelihood, N, lam, R, B, eps, b)
        likelihood.initialize(noise=lam)
        model.covar_module.initialize(variance=1)
        model.task_covar_module.initialize(
            covar_factor = np.sqrt(b/((1+b)*N))*torch.ones((N,1)), 
            raw_var=np.log(1/(1+b))*torch.ones(N))

        if 0:
            optimize_params(model, likelihood, model.likelihood.parameters(), 
                            full_train_x, full_train_i, full_train_y)
        # Set into eval mode
        model.eval()
        likelihood.eval()
        ##########

        if 'naive' in algo:
            beta_t = beta_t_naive(model=model)
        else:
            beta_t = beta_t_improved(model=model)
        with torch.no_grad():
            avg_regret = 0
            task_acquisition = []
            candidate_idxs_t = []
            N_rewards = np.empty(N)
            if t > 0 and t % 10 == 0:
                fig, ax = plt.subplots(1,N,figsize=(3*N,3), dpi=200) 
            for i in range(N):
                if tasks_data:
                    test_x = tasks_data['domains'][i]
                test_i = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=i)
                observed_pred_y = likelihood(model(test_x, test_i))
                if algo == 'MTS':
                    ts = observed_pred_y.sample().numpy()
                    idx_i_t = ts.argmax()
                else:
                    ucbs = observed_pred_y.mean + beta_t[i]*torch.sqrt(observed_pred_y.variance)
                    lcbs = observed_pred_y.mean - beta_t[i]*torch.sqrt(observed_pred_y.variance)
                    idx_i_t = ucbs.argmax().numpy()
                x_i_t = test_x[idx_i_t]

                if t > 0 and t % 10 == 0:
                    true_y = torch.matmul(test_x, task_vectors[i]).numpy().squeeze()
                    pred_mean = observed_pred_y.mean.numpy()
                    beta_t_n = beta_t_naive(model=model)[i]
                    ucbs_naive = observed_pred_y.mean + beta_t_n*torch.sqrt(observed_pred_y.variance)
                    lcbs_naive = observed_pred_y.mean - beta_t_n*torch.sqrt(observed_pred_y.variance)
                    plot_confidence_regions(ax[i], true_y, pred_mean, lcbs_naive, ucbs_naive, lcbs, ucbs, task_id=i, t=t)
                    if i==N-1:
                        fig.text(0.5, 0.005, r'downsampled idxs of domain points $x_i \in \mathbb{R}^4$ ', ha='center', va='center', fontsize=14)
                        fig.text(0.005, 0.5, r'$f(x_i)$', ha='center', va='center', rotation='vertical', fontsize=14)
                        h, l = ax[i].get_legend_handles_labels()
                        fig.legend(h, l , fontsize=14, ncol=6, loc= 'upper center', bbox_to_anchor=(0.5, 1.2), frameon=True)
                        fig.set_tight_layout(True)
                        fig.savefig(f'figs_active/fig_intervals_t_{t}.pdf', bbox_inches='tight')

                if tasks_data is None:
                    r_i_t = torch.matmul(x_i_t, task_vectors[i]).numpy().squeeze()
                    r_max_i_t = torch.matmul(test_x, task_vectors[i]).max().numpy()
                else:
                    r_i_t = tasks_data['labels'][i][idx_i_t].numpy().squeeze()
                    r_max_i_t = tasks_data['labels'][i].max().numpy()
                avg_regret += 1/N*(r_max_i_t - r_i_t)

                # Compute task acquisition function for each algorithm
                if algo == 'MTS':
                    if len(x_idx_sequence[i]) > 0:
                        improvement = ts.max() - np.max([ts[j] for j in x_idx_sequence[i]])
                    else:
                        improvement = ts.max()
                    task_acquisition.append(improvement)
                elif algo == 'AE-LSVI':
                    task_acquisition.append(ucbs.max() - lcbs.max())
                elif 'active' in algo:
                    sigma_i_t = torch.sqrt(observed_pred_y.variance[idx_i_t]).squeeze().numpy()
                    task_acquisition.append(beta_t[i]*sigma_i_t)
                candidate_idxs_t.append(idx_i_t)
                N_rewards[i] = r_i_t

            # Which task to query?
            if 'uniform' in algo:
                i_t = np.random.choice(N)
            else:
                i_t = task_acquisition.index(max(task_acquisition))
            #################################### 
            if tasks_data is None:   
                x_t = test_x[candidate_idxs_t[i_t]]
                r_t = torch.matmul(x_t, task_vectors[i_t])
                y_t = r_t  + torch.randn(1) * R
            else:
                x_t = tasks_data['domains'][i_t][candidate_idxs_t[i_t]]
                r_t = tasks_data['labels'][i_t][candidate_idxs_t[i_t]]
                y_t = r_t + torch.randn(1) * 0.01 
            x_idx_sequence[i_t].append(candidate_idxs_t[i_t])
            
        if 0 and t%10 == 0:
            axs[i_t].plot(lcbs.numpy())
            axs[i_t].plot(ucbs.numpy())
        if t%10 == 0:
            print(f"algo: {algo}, run: {run_id}, beta_t: {beta_t}, t: {t}, avg_regret: {avg_regret:.4f}, rewards: {N_rewards}")

        full_train_x = torch.cat([full_train_x, x_t.reshape(1,d)], axis=0)
        full_train_i = torch.cat([full_train_i, torch.tensor(i_t).reshape(1,1)])
        full_train_y = torch.cat([full_train_y, y_t])
        rewards.append(N_rewards)
        regrets.append(avg_regret)
        tasks_sequence.append(i_t)
    return rewards, regrets, tasks_sequence