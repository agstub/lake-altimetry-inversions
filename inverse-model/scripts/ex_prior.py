# generate sample from prior distribution and plot the result
#

import sys
sys.path.insert(0, '../source')
import os
if os.path.isdir('../prior_pngs')==False:
    os.mkdir('../prior_pngs')    # make a directory for the results.

from conj_grad import cg_solve
from params import x,x0,y0,t0,eta_d,beta_d,t_sc,rho_i,g,H,lamda0,beta0,Nt,Ny,Nx,dt,t
from prior import Cpri_inv,A
import numpy as np
import matplotlib.pyplot as plt
from operators import fwd_uq
from scipy.integrate import cumulative_trapezoid
from kernel_fcns import conv
from scipy.special import gamma

M = 1

kappa = 0.05
tau = 60
a = 5

# var = gamma(1)/ (gamma(2)*4*np.pi*(kappa**(2))*(tau**2))
# print('3 sigma = '+str(np.sqrt(var)))
# print('cor length = '+str(np.sqrt(8/kappa)))

w_pri = np.zeros((Nt,Ny,Nx))

# # for i in range(M):
f = np.random.default_rng().normal(size=np.shape(x))
X, sample = cg_solve(lambda X: A(X,kappa=kappa),f,tol=1e-5)

w_pri = (np.sqrt(2*a)/tau)*conv(np.exp(-a*t),X)


for i in range(Nt):
    plt.figure(figsize=(8,6))
    plt.title(r'prior sample',fontsize=18)
    plt.contourf(x0,y0,w_pri[i,:,:].T/np.max(np.abs(w_pri)),cmap='coolwarm',extend='both',levels=np.arange(-1,1.1,0.1))
    plt.colorbar()
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('../prior_pngs/'+str(i))
    plt.close()


## plot snapshot with time series: 
# plt.figure(figsize=(16,6))
# plt.subplot(121)
# plt.title(r'prior sample (Matérn in space)',fontsize=20)
# plt.contourf(x0,y0,w_pri[50,:,:].T/np.max(np.abs(w_pri)),cmap='coolwarm',extend='both',levels=np.arange(-1,1.1,0.1))
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# plt.ylabel(r'$y$ (km)',fontsize=20)
# plt.xlabel(r'$x$ (km)',fontsize=20)
# plt.xlim(-40,40)
# plt.ylim(-40,40)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.subplot(122)
# plt.title(r'prior sample (OU in time)',fontsize=20)
# plt.plot(t0,w_pri[:,::25,40]/np.max(np.abs(w_pri)),linewidth=3)
# plt.xlabel(r'$t$ (yr)',fontsize=20)
# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.savefig('ex_prior')
# plt.close()
