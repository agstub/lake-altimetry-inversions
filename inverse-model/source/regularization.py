# this defines the regularization

from params import k,kx,ky,x,y,dt,t
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # Negative Laplacian computed via Fourier transform
    return ifftd((k**2)*fftd(f)).real

def grad(f):
    fx = ifftd(1j*kx*fftd(f)).real
    fy = ifftd(1j*ky*fftd(f)).real
    return np.sqrt(fx**2 + fy**2)

def reg(f,eps):
    # # regularization
    #return eps*f      # alternative "L2" regularization
    return eps*lap(f)  # "H1" gradient/smoothness regularization
