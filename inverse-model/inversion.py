# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve,norm
from operators import forward_w,adjoint_w
import numpy as np
from scipy.fft import fft2,ifft2
from params import h0,t,lamda
from kernel_fcns import Rg_

def invert(data,eps_w):
    # invert for w given the observed elevation change h_obs
    # data = h_obs

    print('Solving normal equations with CG....\n')
    b = adjoint_w(fft2(data)-np.exp(-lamda*Rg_*t)*fft2(h0))
    sol = cg_solve(b,eps_w)
    fwd = ifft2(forward_w(sol)).real+ifft2(np.exp(-lamda*Rg_*t)*fft2(h0)).real
    mis = norm(fwd-data)/norm(data)

    return sol,fwd,mis
