# define noise covariance operator
import os
import numpy as np
from params import data_dir
from kernel_fcns import conv,xcor,fftd,ifftd
from prior import Cpri_inv
from conj_grad import cg_solve

model_var = 1e-1
noise_var = 1e-3

def Cerr_inv(f):
    # identity operator divided by variance
    X = f/(noise_var + model_var)
    return X
