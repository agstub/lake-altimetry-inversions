# this defines the regularization

from params import k,x,y,dt,t
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # Negative Laplacian computed via Fourier transform
    return ifftd((k**2)*fftd(f)).real


def reg(f,eps):
    # regularization
    # return eps*f     # alternative "L2" regularization
    return eps*lap(f) # "H1" gradient/smoothness regularization
