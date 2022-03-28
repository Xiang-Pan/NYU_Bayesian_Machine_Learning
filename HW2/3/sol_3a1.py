'''
Author: Xiang Pan
Date: 2021-10-01 17:39:39
LastEditTime: 2021-10-07 23:24:19
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_Bayesian_Machine_Learning/HW2/sol_3a1.py
xiangpan@nyu.edu
'''
import scipy.io as scio
from scipy.optimize import minimize
import scipy.optimize as opt
import torch
import numpy as np
import math
import torch.optim as optim
from torch.autograd import Variable

data_file = './hw2files/occam1.mat'

data = scio.loadmat(data_file)

def kernel1(x):
    return [1, x, x**2, x**3, x**4, x**5]

# def kernel2(x):
#     return 1

# def kernel3(x):
#     return 1

x = data['x']
x = np.array([float(xi) for xi in data['x']], dtype=np.double)



# cov_kx1 = np.cov(kx1)
# inv_cov_kx1 = np.linalg.inv(cov_kx1)

y = data['y']
y = np.array([float(yi) for yi in y], dtype=np.double)
# y = np.array(y)
train_x = torch.DoubleTensor(x)
train_y = torch.DoubleTensor(y)

# N = len(x)

def SigmaM(sigma, alpha):
    M = alpha * cov_kx1  + sigma * np.identity(N) + 1e-6 * np.identity(N)
    return M

def loss_ori(z):
    sigma = z[0]
    alpha = z[1]
    M = SigmaM(sigma, alpha) 
    inv_M = np.linalg.inv(M)
    loss = 1/2 * y.T @ inv_M @ y + 1/2 * np.log(np.linalg.det(M))
    return float(loss)

def grad_loss_ori(z):
    sigma = z[0]
    alpha = z[1]
    M = SigmaM(sigma, alpha) 
    inv_M = np.linalg.inv(M)
    grad_sigma = -1/2 * y.T @ inv_M @ np.identity(N) @ inv_M @ y + 1/2 * np.trace(inv_M @ np.identity(N))
    grad_alpha = -1/2 * y.T @ inv_M @ cov_kx1 @ inv_M @ y + 1/2 *np.trace(inv_M @ cov_kx1)
    grad = np.asarray((float(grad_sigma), float(grad_alpha)), dtype=np.float)
    return grad

def phi1(x, z=None):
    kx1 = [kernel1(xi) for xi in x] 
    kx1 = np.stack(kx1)
    return kx1

def z_grad_phi1(x, z):
    return None

def phi2(x, z):
    kx2 = np.stack([np.exp(- ((x - 1) ** 2 / z[0] ** 2)), np.exp(- ((x - 5) ** 2 / z[1] ** 2))]).squeeze().T
    return kx2

def z_grad_phi2(x, z):
    return [np.stack([2 * (x - 1) ** 2 * np.exp(-((x - 1) ** 2) / (z[0] ** 2)) / (z[0] ** 3), np.zeros(x.shape)]).T,
            np.stack([np.zeros(x.shape), 2 * (x - 5) ** 2 * np.exp(-((x - 5) ** 2) / (z[1] ** 2)) / (z[1] ** 3),]).T]
    return None

def phi3(x, z=None):
    kx3 = np.stack([x, np.cos(2 * x)]).squeeze().T
    return kx3

def z_grad_phi3(x,z):
    return None

phi = phi2
z_grad_function = z_grad_phi2

# only work in plot
def compute_evidence(loss, alpha_hat, sigma_hat, z_hat, param_number=6, phi=phi1, x=np.array([float(xi) for xi in scio.loadmat('./hw2files/occam1.mat')['x']], dtype=np.double)):
    N = x.shape[0]
    phi_m = phi(x, z_hat).T @ phi(x, z_hat)
    param_shape = param_number
    log_evidence =  - loss                                   \
                    + 0.5 * param_shape * np.log(2 * np.pi) \
                    + N * np.log((2 * np.pi) ** (-param_shape / (2 * N)) * alpha_hat)\
                    - 0.5 * N * np.log(sigma_hat)           \
                    - 0.5 * param_shape * np.log(alpha_hat) \
                    - 0.5 * np.log(np.linalg.det(- (1 / alpha_hat) * np.identity(param_shape)- (1 / sigma_hat) * phi_m))

    evidence = np.exp(log_evidence)
    print(log_evidence, evidence)
    return log_evidence, evidence

def loss(params):
    alpha_2 = params[0]
    sigma_2 = params[1]
    if len(params) == 2:
        z = None
    else:
        z = params[2:]
    
    phi_x = phi(x,z)
    N = x.shape[0]
    param_shape = phi_x.shape[1]
    phi_xt = phi_x.T
    # refer to our definition of 1/S^2, but S here is differnt S in hw
    S = (1 / alpha_2) * np.identity(param_shape) + (1 / sigma_2) * phi_xt @ phi_x
    
    # add jitter, tried different jitter
    S = S + 1e-6 * np.identity(param_shape)

    # LL 
    LL = - N * np.log(2 * np.pi) / 2 \
        - N * math.log(alpha_2 ** (param_shape / N) * sigma_2) / 2 \
        - math.log(np.linalg.det(-S)) / 2 \
        - y.T @ y / (2 * sigma_2) \
        + y.T @ phi_x @ np.linalg.inv(S) @ phi_xt @ y / (2 * (sigma_2 ** 2))
    
    loss = -LL
    
    print(loss, alpha_2, sigma_2, z)
    return loss

def get_grad_z_i(i, z, S, S_inv, phi_x, phi_xt, alpha_2, sigma_2):
    phi_z_grad = z_grad_function(x, z)[i]
    phi_z_grad_t = z_grad_function(x, z)[i].T

    sigma_z_grad = (phi_z_grad_t @ phi_x + phi_xt @ phi_z_grad) / sigma_2
    # calculate each component,  using matlab result
    grad_z_i = 0.5 * (- np.trace(S_inv @ sigma_z_grad) + (y.T @ phi_z_grad @ S_inv @ phi_xt @ y - y.T @ phi_x @ S_inv @ sigma_z_grad @ S_inv @ phi_xt @ y + y.T @ phi_x @ S_inv @ phi_z_grad_t @ y) / (sigma_2 ** 2))
    return grad_z_i

def grad_loss(params):
    alpha_2 = params[0]
    sigma_2 = params[1]
    if len(params) == 2:
        z = None
        with_z = False
    else:
        z = params[2:]
        with_z = True
        
    phi_x = phi(x, z=z)
    phi_xt = phi_x.T
    phi_m = phi_xt @ phi_x
    N = x.shape[0]
    param_shape = phi_x.shape[1]
    
    S = (1 / alpha_2) * np.identity(param_shape) + (1 / sigma_2) * phi_m # similar to above
    S = S + 1e-6 * np.identity(param_shape)                              # sometimes singular
    S_inv = np.linalg.inv(S)

    grad_z = []
    if with_z:
        grad_z = [get_grad_z_i(i, z, S, S_inv, phi_x, phi_xt, alpha_2, sigma_2) for i in range(param_shape)]

    grad_alpha = 1/2 * (- param_shape / alpha_2 + np.trace(S_inv) / (alpha_2 ** 2) + y.T @ phi_x @ S_inv @ S_inv @ phi_xt @ y / (sigma_2 ** 2 * alpha_2 ** 2))

    grad_sigma = -1/2 * (N / sigma_2 - np.trace(S_inv @ phi_xt @ phi_x) / (sigma_2 ** 2) - y.T @ y / (sigma_2 ** 2) - y.T @ phi_x @ (S_inv @ phi_xt @ phi_x @ S_inv / sigma_2 - 2 * S_inv) @ phi_xt @ y / (sigma_2 ** 3))
    return [-float(grad_alpha), -float(grad_sigma)] + grad_z

# change dataset at the top.
if __name__ == '__main__':
    # model1
    # phi = phi1
    # z_grad_function = z_grad_phi1
    # result = opt.minimize(loss, (300,300,0), bounds=((1e-6,100000),(1e-6,100000),(-np.inf,np.inf)), jac=grad_loss, method="L-BFGS-B",options={'disp': True, 'maxiter': 20000, "fatol": 1e-6,"gtol": 1e-6, "adaptive":True})
    # model2
    phi = phi2
    z_grad_function = z_grad_phi2
    result = opt.minimize(loss, (300,300, 300, 300), bounds=((1e-6,100000),(1e-6,100000),(-np.inf,np.inf), (-np.inf,np.inf)), jac=grad_loss, method="L-BFGS-B",options={'disp': True, 'maxiter': 20000, "fatol": 1e-6,"gtol": 1e-6, "adaptive":True})
    # model3
    # phi = phi3
    # z_grad_function = z_grad_phi3
    # result = opt.minimize(loss, (300,300), bounds=((1e-6,100000),(1e-6,100000)), jac=grad_loss, method="L-BFGS-B",options={'disp': True, 'maxiter': 20000, "fatol": 1e-6,"gtol": 1e-6, "adaptive":True})





# --------------------------------------------------------------------------------
# some attemption
# check differnt optimizer
# options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20}
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters


# --------------------------------------------------------------------------------
# using gp package, but fail, I guess the matrix condition is too bad.
# # "Loss" for GPs - the marginal log likelihood
# import gpytorch
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).double()
# gpytorch.settings.cholesky_jitter(1)
# training_iter = 100000
# for i in range(training_iter):
#     # Zero gradients from previous iteration
#     # optimizer.zero_grad()
#     # Output from model
#     # output = model(train_x)
#     # Calc loss and backprop gradients
#     # loss = -mll(output, train_y)
#     def closure():
#     # Zero gradients from previous iteration
#         optimizer.zero_grad()
#     # Output from model
#         output = model(train_x)
#     # Calc loss and backprop gradients
#         loss = -mll(output, train_y)
#         loss.backward()
#         return loss
#     # loss.backward()
#     # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#     #     i + 1, training_iter, loss.item(),
#     #     model.covar_module.base_kernel.lengthscale.item(),
#     #     model.likelihood.noise.item()
#     # ))
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         i + 1, training_iter, 1,
#         model.covar_module.base_kernel.lengthscale.item(),
#         model.likelihood.noise.item()
#     ))
#     optimizer.step(closure)
# # print(model)