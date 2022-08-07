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
from localization import localize

def invert(h_obs,eps_w):
    # invert for basal vertical velocity w given the observed elevation change h_obs

    print('Solving normal equations with CG....\n')
    # extract initial elevation profile (goes on RHS of normal equations "b")
    h0 = h_obs[0,:,:] + np.zeros(np.shape(h_obs))
    b = adjoint_w(fftd(h_obs)-np.exp(-lamda*Rg_*t)*fftd(h0))

    # solve the normal equations with CG for the basal vertical velocity anomaly
    w_inv = cg_solve(b,eps_w)

    # get the forward solution (elevation profile) associated with the inversion
    h_fwd = ifftd(forward_w(w_inv)).real+ifftd(np.exp(-lamda*Rg_*t)*fftd(h0)).real

    # calculate the relative misfit between the observed and modelled elevation
    mis = norm(h_fwd-h_obs)/norm(h_obs)

    # subtract off-lake component in the inversion w_inv
    # (data have off-lake component removed, so you have to remove this from the inversion too)
    w_inv = localize(w_inv)

     # make a directory for the results
    if os.path.isdir('../results')==False:
        os.mkdir('../results')

    # save the results
    np.save('../results/w_inv.npy',w_inv)
    np.save('../results/h_fwd.npy',h_fwd)

    return w_inv,h_fwd,mis
