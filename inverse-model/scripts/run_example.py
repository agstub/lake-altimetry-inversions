# example script that runs the inversion

import sys
sys.path.insert(0, '../source')

from inversion import invert
import numpy as np
import os
from params import data_dir
from plot_results import plot_movie, plot_snap
from print_params import print_params

# print the auxiliary model parameters
print_params()

# load elevation data (h_obs)
h_obs = np.load(data_dir+'/h_obs.npy')

# set regularization parameter
eps = 1e0

# set reference time to define elevation anomaly
t_ref = 0 # t_ref = 1.25 good for SLM?

# solve for the basal vertical velocity inversion
w_map,h_fwd,mis = invert(h_obs,eps=eps,t_ref=t_ref)

# plot the results
plot_movie()    # plot a movie

plot_snap(74)   # plot snapshot at specified timestep
