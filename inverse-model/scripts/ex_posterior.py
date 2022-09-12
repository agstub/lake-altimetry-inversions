# example script running the inversion on some synthetic data
# and plotting the results (MAP point)

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import x,y,t,Nt,data_dir,H,x0,y0,t0,Nx,Ny,lamda0,beta0
import numpy as np
import os
from error_model import noise_var
from localization import localize
from conj_grad import norm

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h_obs = np.load(data_dir+'/h_obs.npy')

kappa = 0.0001
tau = 10
a = 10

num = 2                     # num posterior samples

w_map,sample,h_fwd,mis = invert(h_obs,kappa=kappa,tau=tau,a=a,num=num)    # good for beta = 1e9
print('rel. misfit norm = '+str(mis))
