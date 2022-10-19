# example script running the inversion

import sys
sys.path.insert(0, '../source')

from inversion import invert
import numpy as np
import os
from params import data_dir

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h_obs = np.load(data_dir+'/h_obs.npy')

eps = 1e-3*50

w_map,h_fwd,mis = invert(h_obs,eps=eps)    # good for beta = 1e9
print('rel. misfit norm = '+str(mis))
