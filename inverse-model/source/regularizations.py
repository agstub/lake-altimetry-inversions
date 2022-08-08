# this file contains the regularization options
# * these are just standard Tikhonov regularizations of either
# (1) the function ("L2"), or
# (2) gradients of the functions ("H1")


from params import k,x,y,dt
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # negative Laplacian computed via Fourier transform
    return ifftd((k**2)*fftd(f)).real


def reg(f,eps_1,eps_2):
    # elliptic operator
    R = eps_1*f+eps_2*lap(f)    # note: adding an L2 term here seems to help with convergence
    return R

def Cpri_inv(f,eps_1,eps2):
    # squared elliptic operator (inverse of prior covariance operator)
    return reg(reg(X,eps_1,eps_2),eps_1,eps_2)
