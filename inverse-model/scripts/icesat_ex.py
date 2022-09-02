# example script running the inversion on some ICESat data

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import data_dir
import numpy as np

h_obs = np.load(data_dir+'/h_obs.npy')

kappa = 0.0001
tau = 2e0 ## use for white noise
a = 5

num = 2                     # num posterior samples

w_map,sample,h_fwd,mis = invert(h_obs,kappa=kappa,tau=tau,a=a,num=num)    # good for beta = 1e9
print('rel. misfit norm = '+str(mis))
