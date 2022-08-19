# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve,norm
from operators import fwd,adj,Cpost_inv
import numpy as np
from kernel_fcns import fftd,ifftd
from params import t,lamda,x,Nx
from kernel_fcns import Rg_
import os
from post_process import calc_dV_w
from localization import localize
from noise import Cnoise_inv

def invert(h_obs,eps_1,eps_2,num):
    # invert for basal vertical velocity w given the observed elevation change h_obs

    print('Solving normal equations with CG....\n')
    # extract initial elevation profile (goes on RHS of normal equations "b")
    h0 = h_obs[0,:,:] + np.zeros(np.shape(h_obs))
    b = adj(Cnoise_inv(fftd(h_obs)-np.exp(-lamda*Rg_*t)*fftd(h0)))


    # solve the normal equations with CG for the basal vertical velocity anomaly
    w_map,sample = cg_solve(lambda X: Cpost_inv(X,eps_1,eps_2),b,num)

    # get the forward solution (elevation profile) associated with the inversion
    h_fwd = ifftd(fwd(w_map)).real+ifftd(np.exp(-lamda*Rg_*t)*fftd(h0)).real

    # calculate the relative misfit between the observed and modelled elevation
    mis = norm(h_fwd-h_obs)/norm(h_obs)

    # subtract off-lake component in the inversion w_map
    # (data have off-lake component removed, so you have to remove this from the inversion too)
    w_map = localize(w_map)

    for l in range(num):
        sample[:,:,:,l] = localize(sample[:,:,:,l])

     # make a directory for the results
    if os.path.isdir('../results')==False:
        os.mkdir('../results')

    # save the results
    np.save('../results/w_map.npy',w_map)
    np.save('../results/h_fwd.npy',h_fwd)

    return w_map,sample,h_fwd,mis
