import numpy as np
import gpytorch
import torch

class MultitaskGPModel(gpytorch.models.ExactGP):
    " Multitask GP model with linear kernel on domain and index kernel on tasks"
    def __init__(self, train_x, train_y, likelihood, N, lam, R, B, eps, b):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=N, rank=1)

        self.N = N
        self.lam = lam
        self.R = R
        self.b = b
        # upper bound on tasks norm and variance
        self.B = B
        self.eps = eps
        
    def forward(self,x,i):
        mean_x = self.mean_module(x)
        
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
def optimize_params(model, likelihood, params, full_train_x, full_train_i, full_train_y):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(params, lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for step in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(full_train_x, full_train_i)
        # Calc loss and backprop gradients
        loss = -mll(output, full_train_y)
        loss.backward()
        if step == 49:
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
            step + 1, 30, loss.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

def compute_aux_matrices(N, d, b):
    "Compute interaction matrix and associated ones, given N, d, b parameters"
    A = (1+b)*np.eye(N) - b/N*np.ones((N,N))
    A_inv = np.linalg.inv(A)
    A_half = np.sqrt(1+b)*np.eye(N) + (1-np.sqrt(1+b))/(N)*np.ones((N,N))
    A = np.kron(A, np.eye(d))
    A_half = np.kron(A_half, np.eye(d))
    return A, A_half, A_inv


def beta_t_naive(params=None, model=None, delta=0.01):
    " Naive confidence width for multi-task regression"
    AssertionError(params is None and model is None, "Need to specify params or a model")
    if model:
        t = len(model.train_targets)
        N = model.N
        b = model.b
        B = model.B
        R = model.R
        eps = model.eps 
        lam = model.lam
    else:
        t, N, lam, R, b, B, eps, gamma_t = params["t"], params["N"], params["lam"], params["R"], params["b"], params["B"], params["eps"], params['gamma_t']
    # Bias term:
    first_term = B*np.sqrt(N*(1 + b*eps**2))
    if model:
        K_x = model.covar_module(model.train_inputs[0]).numpy()
        K_tasks = model.task_covar_module(model.train_inputs[1]).numpy()
        K_t = np.multiply(K_x, K_tasks)
        gamma_t = 0.5*np.log(np.linalg.det(np.eye(t)+ lam**-1*K_t))
    # Variance term:
    second_term = lam**(-0.5)*np.sqrt(2*(gamma_t + np.log(1/delta))) 

    beta_t = first_term + R*second_term
    return beta_t*np.ones(N)


def beta_t_improved(params=None, model=None, delta=0.01):
    " Improved confidence width for multi-task regression"
    AssertionError(params is None and model is None, "Need to specify params or a model")
    if model:
        t = len(model.train_targets)
        N = model.N
        b = model.b
        B = model.B
        R = model.R
        eps = model.eps
        lam = model.lam
    else:
        t, N, lam, R, b, B, eps, gamma_t, gamma_t_X = params["t"], params["N"], params["lam"], params["R"], params["b"], params["B"], params["eps"], params['gamma_t'], params['gamma_t_X']

    beta_t = np.zeros(N)

    # Variance term:
    if model:
        K_x = model.covar_module(model.train_inputs[0]).numpy()
        K_tasks = model.task_covar_module(model.train_inputs[1]).numpy()
        K_t = np.multiply(K_x, K_tasks)
        gamma_t = 0.5*np.log(np.linalg.det(np.eye(t)+ lam**-1*K_t))
        second_term = lam**(-0.5)*np.sqrt( 2*(gamma_t + np.log(1/delta)))*np.ones(N) 
        
        second_term_improved = np.empty(N)
        gamma_t_X = np.empty(N)
        idx_tasks = model.train_inputs[1].numpy().squeeze()
        d = model.train_inputs[0][idx_tasks==0].size(1)
        sum_M_mat = np.zeros((d,d))
        term_D = 0
        for i in range(N):
            K_x_i = model.covar_module(model.train_inputs[0][idx_tasks==i]).numpy()
            gamma_t_X[i] = 0.5*np.log(np.linalg.det(np.eye(len(K_x_i))+ lam**-1*K_x_i))
            inner_prod_Mat = torch.inner(model.train_inputs[0][idx_tasks==i].T, model.train_inputs[0][idx_tasks==i].T)
            sum_M_mat += np.linalg.inv(lam*(1+b)*np.eye(d) + inner_prod_Mat.numpy())
            
            sigmas = list(np.linalg.eigvalsh(inner_prod_Mat))
            term_D += sum([s/(s + lam*(1+b)) for s in sigmas])
        for i in range(N):
            second_term_improved[i] = lam**(-0.5)*np.sqrt(2*(gamma_t_X[i] + np.log(1/delta)) + 2*b*np.sum([gamma_t_X[i] + np.log(1/delta) for i in range(N)]))
    
        second_term = np.minimum(second_term, second_term_improved)
    else:
        second_term = lam**(-0.5)*np.sqrt( 2*(gamma_t + np.log(1/delta))) 
        second_term_improved = lam**(-0.5)*np.sqrt(2*(1+b*N)*(gamma_t_X + np.log(1/delta))) 
        second_term = min(second_term, second_term_improved)*np.ones(N)

    # Bias term 
    first_term = B*np.sqrt(N*(1 + b*eps**2)) # naive bound
    first_term_improved_1= B*np.sqrt((1+b*N)/(1+b))*(1 + b*eps) # small-b range
    first_term_improved_2 =  B*np.sqrt( (1 + b*eps)**2/(1+b) + 2*b*N/(1+b) + 
                2*b*(1+b*eps)**2*(t**2)/(N*(lam**2)*((1+b)**3)) ) # large-b range (data-independent)
    norm_D = B*(1+b*eps)*np.linalg.norm(sum_M_mat)
    norm_D_v2 = N*B/(lam*(1+b))  + 1/(lam*(1+b))*term_D
    norm_D = min(norm_D, norm_D_v2)
    first_term_improved_3 = np.sqrt(B**2*(1 + b*eps)**2/(1+b) + 
                                    lam**2*b*(1+b)/N * norm_D**2) # large-b range (data-dependent)
    first_term = min(first_term, first_term_improved_1,
                     first_term_improved_2, first_term_improved_3)
    beta_t = first_term*np.ones(N) + R*second_term
    return beta_t

