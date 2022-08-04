import sys
sys.path.insert(0, '../source')

from params import data_dir
from inversion import invert
import numpy as np

h_obs = np.load(data_dir+'/h.npy')

w_inv,h_fwd,dV_inv,mis = invert(h_obs,2e-3)
