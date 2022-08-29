import sys
sys.path.insert(0, '../source')
import os
if os.path.isdir('../error_pngs')==False:
    os.mkdir('../error_pngs')

from conj_grad import cg_solve
from params import x,x0,y0,eta_d,beta_d,t_sc,rho_i,g,H,lamda0,beta0,Nt,Ny,Nx,dt,t
from prior import Cpri_inv,A
import numpy as np
import matplotlib.pyplot as plt
from operators import fwd_uq
from kernel_fcns import conv

num = 500                                                    # number of samples

# prior parameters
kappa = 0.0001
tau = 10
a = 5

w_pri = np.zeros((Nt,Ny,Nx,num))

for i in range(num):
    print('\n prior sample '+str(i+1)+' out of '+str(num))
    f = np.random.default_rng().normal(size=np.shape(x))
    X, sample = cg_solve(lambda X: A(X,kappa=kappa),f,tol=1e-3)
    w_pri[...,i] = conv(np.exp(-a*t),X/tau)

etad_dist = 10**np.random.default_rng().normal(loc=np.log10(eta_d),scale=2.0/3.0,size=num)
betad_dist = 10**np.random.default_rng().normal(loc=np.log10(beta_d),scale=2.0/3.0,size=num)

t_r = 2*etad_dist/(rho_i*g*H)     # viscous relaxation time

# nondimensional parameters
lamda_dist = t_sc/t_r           # process timescale relative to
                                # surface relaxation timescale

beta_dist = betad_dist*H/(2*etad_dist) # drag coefficient relative to ice viscosity and thickness

err_mean = 0*x
err_var = 0*x

err = np.zeros((Nt,Ny,Nx,num))

for i in range(num):
    print('i = '+str(i))
    err[...,i] = fwd_uq(w_pri[...,i],lamda0,beta0)-fwd_uq(w_pri[...,i],lamda_dist[i],beta_dist[i])
    print('max err = '+str(np.max(np.abs(err[...,i]))))
    err_mean += err[...,i]/num

for i in range(num):
    err_var += (1/(num-1))*(err[...,i] - err_mean)**2


## Plot mean error
# for i in range(Nt):
#     plt.figure(figsize=(8,6))
#     plt.title(r'variance error',fontsize=24)
#     plt.contourf(x0,y0,err_mean[i,:,:].T/np.max(np.abs(err_mean)),cmap='coolwarm',extend='both',levels=np.arange(-1,1.1,0.1))
#     plt.colorbar()
#     plt.ylabel(r'$y$ (km)',fontsize=20)
#     plt.xlabel(r'$x$ (km)',fontsize=20)
#     plt.xlim(-40,40)
#     plt.ylim(-40,40)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.savefig('../error_pngs/err_'+str(i))
#     plt.close()


## Plot variance
# for i in range(Nt):
#     plt.figure(figsize=(8,6))
#     plt.title(r'mean error',fontsize=24)
#     plt.contourf(x0,y0,err_var[i,:,:].T/np.max(np.abs(err_var)),cmap='coolwarm',extend='both',levels=np.arange(-1,1.1,0.1))
#     plt.colorbar()
#     plt.ylabel(r'$y$ (km)',fontsize=20)
#     plt.xlabel(r'$x$ (km)',fontsize=20)
#     plt.xlim(-40,40)
#     plt.ylim(-40,40)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.savefig('../error_pngs/err_'+str(i))
#     plt.close()
