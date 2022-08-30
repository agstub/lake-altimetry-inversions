# define noise covariance operator
import os
import numpy as np

if os.path.isfile('../uncertainty/var_red.npy')==True:
    model_var = np.load('../uncertainty/var_red.npy')
else:
    model_var = 0

noise_var = 1e-4

var = model_var + noise_var

def Cerr_inv(f):
    # identity operator divided by variance
    return f/(noise_var+model_var)
