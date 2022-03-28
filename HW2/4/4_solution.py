'''
Author: Xiang Pan
Date: 2021-10-07 17:35:53
LastEditTime: 2021-10-07 17:42:43
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_Bayesian_Machine_Learning/HW2/4_solution.py
xiangpan@nyu.edu
'''
from mcmc import *
from scipy.io import loadmat
import numpy as np
# from murray import log_pstar, slice_sample, dumb_metropolis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



log_omega = 1
mm = 1
pie = 1
mu1 = 1
mu2 = 1
log_sigma1 = 1
log_sigma2 = 1
# logp = log_pstar(log_omega, mm, pie, mu1, mu2, log_sigma1, log_sigma2, xx, vv)
params = np.array([log_omega, mm, pie ,mu1, mu2, log_sigma1, log_sigma2,])
samples, acceptance_rate = dumb_metropolis(params, log_pstar, 1000, 1, xx, vv)