'''
Author: Xiang Pan
Date: 2021-10-01 17:11:49
LastEditTime: 2021-10-07 22:40:37
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_Bayesian_Machine_Learning/HW2/mcmc.py
xiangpan@nyu.edu
'''
import numpy as np
from scipy.io import loadmat
data = loadmat('astro_data.mat')
xx, vv = data['xx'], data['vv']

def norm(x, mean, std):
    return np.exp(-0.5 * ((x - mean) ** 2) / (std** 2)) / std

def log_pstar(state):
    log_omega, mm, pie, mu1, mu2, log_sigma1, log_sigma2 = state
    N = xx.shape[0]

    # exp process
    sigma1 = np.exp(log_sigma1)
    sigma2 = np.exp(log_sigma2)
    omega = np.exp(log_omega)
    
    x_mu = xx.mean()
    x_std = xx.std()
    ext = xx.max() - xx.min()
    
    log_ext = np.log(ext)
    
    forbidden_conditions = [pie < 0,
                            pie > 1,
                            np.abs(mm - x_mu) > 10 * x_std,
                            np.abs(mu1 - log_ext) > 20,
                            np.abs(mu2 - log_ext) > 20,
                            np.abs(log_sigma1) > np.log(20),
                            np.abs(log_sigma2) > np.log(20),
                            np.abs(log_omega) > 20,
                            ]
    
    if any(forbidden_conditions):
        return -np.inf
    
    log_A = 0.5 * np.log((xx - mm) ** 2 + (vv / omega) ** 2)
    
    log_prior = np.sum(np.log(pie * norm(log_A, mu1, sigma1)
                              + (1 - pie) * norm(log_A, mu2, sigma2)))
    log_like = -2 * log_A.sum() - N * np.log(omega)
    logp = log_like + log_prior
    
    return logp


def dumb_metropolis(init, log_ptilde, iters, sigma, xx=xx, vv=vv):
    D = init.shape[0]
    samples = np.zeros(shape=(D, iters))
    
    accept = 0
    # init state
    state = init 
    Logp_state = log_ptilde(state)
    
    for iter in range(0, iters):
        propose = state + sigma * np.random.randn(len(state))
        Logp_prop = log_ptilde(propose)
        if (np.log(np.random.rand()) < (Logp_prop - Logp_state)).all():
            accept += 1
            state = propose        # accept propose param
            Logp_state = Logp_prop # update state
        samples[:, iter] = state.squeeze()
    accept_rate = accept / iters
    return (samples, accept_rate)
