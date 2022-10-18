# define noise covariance operator
import os
import numpy as np
from params import data_dir
from kernel_fcns import conv,xcor,fftd,ifftd
from prior import Cpri_inv
from conj_grad import cg_solve

noise_var = 1e-3

def Cerr_inv(f):
    # identity operator divided by variance
    return f/noise_var
