import sys
sys.path.insert(0, '../source')
import os

from conj_grad import cg_solve
from params import x,x0,y0,eta_d,beta_d,t_sc,rho_i,g,H,lamda0,beta0,Nt,Ny,Nx,dt,t,t0,data_name
from prior import Cpri_inv,A
import numpy as np
import matplotlib.pyplot as plt
from operators import fwd_uq
from kernel_fcns import conv

num = int(input('input number of samples to draw: '))                                                    # number of samples

# prior parameters
kappa = 0.001
tau = 0.1
a = 10

w_pri = np.zeros((Nt,Ny,Nx,num))

for i in range(num):
    print('\n prior sample '+str(i+1)+' out of '+str(num))
    f = np.random.default_rng().normal(size=np.shape(x))
    X, sample = cg_solve(lambda X: A(X,kappa=kappa),f,restart='off')
    w_pri[...,i] = conv(np.exp(-a*t),X/tau)


etad_dist = 10**np.random.default_rng().normal(loc=np.log10(eta_d),scale=1.0/3.0,size=num)
betad_dist = 10**np.random.default_rng().normal(loc=np.log10(beta_d),scale=1.0/3.0,size=num)

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

var_red = np.multiply.outer(np.mean(err_var,axis=(1,2)),np.ones((Ny,Nx)))


uq_dir = '../UQ'
if os.path.isdir(uq_dir)==False:
    os.mkdir(uq_dir)
    np.save(uq_dir+'/var_red.npy',var_red)
    np.save(uq_dir+'/err_mean.npy',err_mean)
    np.save(uq_dir+'/err_var.npy',err_var)
    np.save(uq_dir+'/betad_dist.npy',betad_dist)
    np.save(uq_dir+'/etad_dist.npy',etad_dist)



    ## PLOTTING-----------------------------------------------------------------
    # xy_str = H/1e3
    #
    # xp = xy_str*x0
    # yp = xy_str*y0
    #
    # plt.figure(figsize=(16,10))
    # sc = np.around(np.max(np.abs(w_pri))/5.,decimals=0)*5
    #
    # plt.suptitle(r'Prior model and parameter uncertainties',fontsize=28,y=1.02,bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
    # plt.subplot(221)
    # plt.title(r'prior sample (Matérn in space)',fontsize=20)
    # plt.contourf(xp,yp,w_pri[50,:,:,0].T,cmap='coolwarm',extend='both',levels=sc*np.arange(-1,1.1,0.1))
    # cbar = plt.colorbar()
    # cbar.set_label(r'$w_\mathrm{pri}$ (m/yr)',fontsize=20)
    # cbar.ax.tick_params(labelsize=16)
    # plt.ylabel(r'$y$ (km)',fontsize=20)
    # plt.xlabel(r'$x$ (km)',fontsize=20)
    # plt.xlim(xp.min(),xp.max())
    # plt.ylim(yp.min(),yp.max())
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    # plt.subplot(222)
    # plt.title(r'prior sample (OU in time)',fontsize=20)
    # plt.plot(t0,w_pri[:,60,20,0],linewidth=3)
    # plt.xlabel(r'$t$ (yr)',fontsize=20)
    # plt.ylabel(r'$w_\mathrm{pri}$ (m/yr)',fontsize=20)
    # plt.gca().yaxis.tick_right()
    # plt.gca().yaxis.set_label_position("right")
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    #
    # plt.subplot(223)
    # bins = np.arange(np.log10(beta_d)-2,np.log10(beta_d)+2,0.1)
    # plt.hist(np.log10(betad_dist),bins)
    # plt.yticks([])
    # plt.xticks(fontsize=16)
    # plt.xlabel(r'log$_{10}(\beta)$',fontsize=20)
    # plt.ylabel(r'$\beta$, $\eta$ uncertainties',fontsize=20)
    #
    # plt.subplot(224)
    # bins = np.arange(11,15.1,0.1)
    # plt.hist(np.log10(etad_dist),bins)
    # plt.xlabel(r'log$_{10}(\eta)$',fontsize=20)
    # plt.yticks([])
    # plt.xticks(fontsize=16)
    #
    # plt.tight_layout()
    # plt.savefig('../ex_prior',bbox_inches='tight')
    #
    # plt.close()


#
# etad_dist = 10**np.random.default_rng().normal(loc=np.log10(eta_d),scale=1.0/3.0,size=200)
# betad_dist = 10**np.random.default_rng().normal(loc=np.log10(beta_d),scale=2.0/3.0,size=200)
#
# xy_str = H/1e3
#
# xp = xy_str*x0
# yp = xy_str*y0
#
# plt.figure(figsize=(16,10))
# sc = np.around(np.max(np.abs(w_pri))/5.,decimals=0)*5
#
# plt.suptitle(r'Prior model and parameter uncertainties',fontsize=28,y=1.02,bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
# plt.subplot(221)
# plt.title(r'prior sample (Matérn in space)',fontsize=20)
# plt.contourf(xp,yp,w_pri[50,:,:,0].T,cmap='coolwarm',extend='both',levels=sc*np.arange(-1,1.1,0.1))
# cbar = plt.colorbar()
# cbar.set_label(r'$w_\mathrm{pri}$ (m/yr)',fontsize=20)
# cbar.ax.tick_params(labelsize=16)
# plt.ylabel(r'$y$ (km)',fontsize=20)
# plt.xlabel(r'$x$ (km)',fontsize=20)
# plt.xlim(xp.min(),xp.max())
# plt.ylim(yp.min(),yp.max())
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.subplot(222)
# plt.title(r'prior sample (OU in time)',fontsize=20)
# plt.plot(t0,w_pri[:,60,20,0],linewidth=3)
# plt.xlabel(r'$t$ (yr)',fontsize=20)
# plt.ylabel(r'$w_\mathrm{pri}$ (m/yr)',fontsize=20)
# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
#
# plt.subplot(223)
# bins = np.arange(np.log10(beta_d)-2,np.log10(beta_d)+2,0.1)
# plt.hist(np.log10(betad_dist),bins)
# plt.yticks([])
# plt.xticks(fontsize=16)
# plt.xlabel(r'log$_{10}(\beta)$',fontsize=20)
# plt.ylabel(r'$\beta$, $\eta$ uncertainties',fontsize=20)
#
# plt.subplot(224)
# bins = np.arange(11,15.1,0.1)
# plt.hist(np.log10(etad_dist),bins)
# plt.xlabel(r'log$_{10}(\eta)$',fontsize=20)
# plt.yticks([])
# plt.xticks(fontsize=16)
#
# plt.tight_layout()
# plt.savefig('../ex_prior',bbox_inches='tight')
#
#
# plt.close()
