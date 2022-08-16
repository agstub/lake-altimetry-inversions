# this file contains the inverse prior covariance operator

from params import k,x,y,dt
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # negative Laplacian computed via Fourier transform
    return ifftd((k**2)*fftd(f)).real


def A(f,eps_1,eps_2):
    # elliptic operator
    R = eps_1*f+eps_2*lap(f)
    return R

def Cpri_inv(f,eps_1,eps_2):
    # squared elliptic operator (inverse of prior covariance operator)
    return A(A(f,eps_1,eps_2),eps_1,eps_2)
