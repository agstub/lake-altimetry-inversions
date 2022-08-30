# example script running the inversion on some synthetic data
# and plotting the results (MAP point)

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import x,y,t,Nt,data_dir,H,x0,y0,t0,Nx,Ny,lamda0,beta0
import numpy as np
import os
from error_model import noise_var
import matplotlib.pyplot as plt
from localization import localize
from conj_grad import norm

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h = np.load(data_dir+'/h.npy')
noise_h = np.random.normal(size=(Nt,Nx,Ny),scale=np.sqrt(noise_var))
h_obs = h + noise_h
h_obs = localize(h_obs)

w_true = np.load(data_dir+'/w_true.npy')

kappa = 0.0001
tau = 10
a = 5

num = 500                     # num posterior samples

w_map,sample,h_fwd,mis = invert(h_obs,kappa=kappa,tau=tau,a=a,num=num)    # good for beta = 1e9
print('rel. misfit norm = '+str(mis))
print('rel. noise norm = '+str(norm(noise_h)/norm(h_obs)))
