# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve,norm
from operators import fwd,adj,A
import numpy as np
from kernel_fcns import fftd,ifftd
from params import t,t0,x,Nx,lamda0,beta0,results_dir
from kernel_fcns import Rg
import os
from localization import localize

def invert(h_obs,eps,t_ref=0):
    # invert for basal vertical velocity w given the observed elevation change h_obs

    # difference elevation data from reference time t_ref
    i0 = np.argmin(np.abs(t0-t_ref))
    h_ref = h_obs[i0,:,:] + 0*h_obs
    h_obs -= h_ref

    print('Solving normal equations with CG....\n')
    # extract initial elevation profile (goes on RHS of normal equations "b")
    h0 = h_obs[0,:,:] + 0*h_obs
    b = adj(fftd(h_obs)-np.exp(-lamda0*Rg()*t)*fftd(h0))

    # solve the normal equations with CG for the basal vertical velocity anomaly
    w_inv = cg_solve(lambda X: A(X,eps),b)

    # get the forward solution (elevation profile) associated with the inversion
    h_fwd = ifftd(fwd(w_inv)).real+ifftd(np.exp(-lamda0*Rg()*t)*fftd(h0)).real

    # calculate the misfit between the observed and modelled elevation
    mis = norm(h_fwd-h_obs)

    # subtract off-lake component in the inversion w_inv
    # (data have off-lake component removed, so you have to remove this from the inversion too)
    w_inv = localize(w_inv)

    # make a directory for the results
    if os.path.isdir(results_dir)==False:
        os.mkdir(results_dir)

    # save the results
    np.save(results_dir+'/w_inv.npy',w_inv)
    np.save(results_dir+'/h_fwd.npy',h_fwd)
    np.save(results_dir+'/t_ref.npy',np.array([t_ref]))


    return w_inv,h_fwd,mis
