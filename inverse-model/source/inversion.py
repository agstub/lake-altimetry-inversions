# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve,norm
from operators import fwd,adj,Cpost_inv,adj_fwd_l,adj_l
import numpy as np
from kernel_fcns import fftd,ifftd
from params import t,x,Nx,lamda0,beta0,results_dir,ind
from kernel_fcns import Rg
import os
from localization import localize
from error_model import Cerr_inv

def invert(h_obs,kappa,tau,a,lamda=lamda0,beta=beta0,num=1):
    # invert for basal vertical velocity w given the observed elevation change h_obs

    print('Solving normal equations with CG....\n')
    # extract initial elevation profile (goes on RHS of normal equations "b")
    h0 = h_obs[0,:,:] + 0*h_obs
    b = adj(Cerr_inv(fftd(h_obs)-np.exp(-lamda*Rg(beta)*t)*fftd(h0)))

    # solve the normal equations with CG for the basal vertical velocity anomaly
    w_map,sample = cg_solve(lambda X: Cpost_inv(X,kappa,tau,a)*ind,b*ind,num)

    # compute sensitivity with respect to lamda parameter (surface decay rate)
    b_l = adj_l(Cerr_inv(fftd(h_obs)-np.exp(-lamda*Rg(beta)*t)*fftd(h0)))
    b_l += adj(Cerr_inv(Rg(beta)*t*np.exp(-lamda*Rg(beta)*t)*fftd(h0)))

    rhs = -(adj_fwd_l(w_map)-b_l)
    w_lamda,sample = cg_solve(lambda X: Cpost_inv(X,kappa,tau,a)*ind,rhs*ind,num)

    # get the forward solution (elevation profile) associated with the inversion
    h_fwd = ifftd(fwd(w_map)).real+ifftd(np.exp(-lamda*Rg(beta)*t)*fftd(h0)).real

    # calculate the relative misfit between the observed and modelled elevation
    mis = norm(h_fwd-h_obs)/norm(h_obs)

    # subtract off-lake component in the inversion w_map
    # (data have off-lake component removed, so you have to remove this from the inversion too)
    w_map = localize(np.nan_to_num(w_map))

    for l in range(num):
        sample[:,:,:,l] = localize(np.nan_to_num(sample[:,:,:,l]))

     # make a directory for the results

    if os.path.isdir(results_dir)==False:
        os.mkdir(results_dir)

    # save the results
    np.save(results_dir+'/w_map.npy',w_map)
    np.save(results_dir+'/post_samples.npy',sample)
    np.save(results_dir+'/h_fwd.npy',h_fwd)
    np.save(results_dir+'/w_lamda.npy',w_lamda)


    return w_map,sample,h_fwd,mis,w_lamda
