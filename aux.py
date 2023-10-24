import numpy as np
import torch
import random
from models import compute_aux_matrices, beta_t_improved
import os 
from scipy.io import loadmat

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_iedb_tasks():
    iedb_path = './iedb_benchmark.mat'
    iedb_data = loadmat(iedb_path)
    x_domains = []
    y_values = []
    for i in range(7):
        x_domains.append(torch.empty(0))
        y_values.append(torch.empty(0))
    idx = 0
    for i in range(len(iedb_data['labels'])):
        task_id = int(iedb_data['contexts'][i])
        x_domains[task_id] = torch.cat([x_domains[task_id], 
                    torch.tensor(iedb_data['examples'][:,i].T).unsqueeze(dim=0).float()], axis=0)
        y_values[task_id] = torch.cat([y_values[task_id],
                        torch.tensor(iedb_data['labels'][i]).float()], axis=0) 
    x_domains = x_domains[1:6]
    y_values = y_values[1:6]
    N = len(y_values) # 5
    d = 45
    return {'domains': x_domains,'labels': y_values}, N, d

def generate_tasks(N, d, dev=0.05, seed=42):
    print(f'generating {N} tasks with deviation {dev} and seed {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    task_vectors = []
    norm = 1
    common = torch.randn(d,1)
    common = norm*common/np.linalg.norm(common, 2)
    random.seed(seed)
    for _ in range(N):
        deviation = torch.randn(d,1)
        deviation = norm*deviation/np.linalg.norm(deviation, 2)
        vector = (1-dev)*common + deviation*dev
        task_vectors.append(vector)
    f = np.vstack([t.numpy() for t in task_vectors])
    f_avg = np.mean([t.numpy() for t in task_vectors], axis=0)
    return task_vectors, f, f_avg

def domain(d, N_points):
    X = -1 + 2*torch.rand(N_points,d) 
    #X = torch.randn(N_points,d) 
    X = 10*torch.divide(X, torch.norm(X, dim=1, keepdim=True)) # normalize to ball of radius 10
    return X

