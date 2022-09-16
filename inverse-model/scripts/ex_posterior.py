# example script running the inversion on some synthetic data
# and plotting the results (MAP point)

import sys
sys.path.insert(0, '../source')

from inversion import invert
import numpy as np
import os
from params import data_dir

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h_obs = np.load(data_dir+'/h_obs.npy')

kappa = 0.001
tau = 0.1
a = 10

num = int(input('input number of posterior samples: '))                                                # num posterior samples

w_map,sample,h_fwd,mis = invert(h_obs,kappa=kappa,tau=tau,a=a,num=num)    # good for beta = 1e9
print('rel. misfit norm = '+str(mis))
