# example script running the inversion

import sys
sys.path.insert(0, '../source')

from inversion import invert
import numpy as np
import os
from params import data_dir
from plot_results import plot_movie, plot_snap

# load elevation data (h_obs)
h_obs = np.load(data_dir+'/h_obs.npy')

eps = 1e0

t_ref = 0*1.25 # t_ref = 1.25 good for SLM?

w_map,h_fwd,mis = invert(h_obs,eps=eps,t_ref=t_ref)

#plot_movie()

plot_snap(74)
