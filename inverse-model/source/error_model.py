# define noise covariance operator
import os
import numpy as np
from params import data_name

uq_dir = '../UQ'

if os.path.isfile(uq_dir+'/var_red.npy')==True:
    model_var = np.load(uq_dir+'/var_red.npy')
else:
    model_var = 0.1

noise_var = 1e-3

var = model_var + noise_var

def Cerr_inv(f):
    # identity operator divided by variance
    return f/var
