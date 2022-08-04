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


def reg(f,reg_type):
    # first variation of regularization functional
    if reg_type == 'H1':
        R = lap(f)       # note: adding an L2 term here seems to help with convergence
    elif reg_type == 'L2':
        R = f
    return R
