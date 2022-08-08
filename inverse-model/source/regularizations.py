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
    # first variation of regularization functional
    R = eps_1*f+eps_2*lap(f)    # note: adding an L2 term here seems to help with convergence
    return R
