# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve,norm
from operators import forward_w,adjoint_w
import numpy as np
from kernel_fcns import fftd,ifftd
from params import t,lamda,x,Nx,h0
from kernel_fcns import Rg_
import os
from post_process import calc_dV_w

def invert(h_obs,eps_w):
    # invert for w given the observed elevation change h_obs
    # data = h_obs

    print('Solving normal equations with CG....\n')
    b = adjoint_w(fftd(h_obs)-np.exp(-lamda*Rg_*t)*fftd(h0))
    w_inv = cg_solve(b,eps_w)

    h_fwd = ifftd(forward_w(w_inv)).real+ifftd(np.exp(-lamda*Rg_*t)*fftd(h0)).real
    mis = norm(h_fwd-h_obs)/norm(h_obs)

    if os.path.isdir('../results')==False:
        os.mkdir('../results')       # make a directory for the results.

    np.save('../results/w_inv.npy',w_inv)
    np.save('../results/h_fwd.npy',h_fwd)



    return w_inv,h_fwd,mis
