# example script running the inversion on some synthetic data
# and plotting the results (MAP point)

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import x,y,t,Nt,data_dir,H,x0,y0,t0,Nx,Ny,lamda0,beta0
import numpy as np
from post_process import calc_dV_w,calc_dV_h,calc_bdry_w,calc_bdry_h
import os
if os.path.isdir('../post_pngs')==False:
    os.mkdir('../post_pngs')    # make a directory for the results.
from error_model import noise_var
import matplotlib.pyplot as plt
from localization import localize
from conj_grad import norm

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h = np.load(data_dir+'/h.npy')
noise_h = np.random.normal(size=(Nt,Nx,Ny),scale=np.sqrt(noise_var))
h_obs = h + noise_h
h_obs = localize(h_obs)

w_true = np.load(data_dir+'/w_true.npy')

kappa = 0.0001
tau = 10
a = 5

w_map,sample,h_fwd,mis = invert(h_obs,kappa=kappa,tau=tau,a=a,num=1)    # good for beta = 1e9

sample = sample[:,:,:,0]

print('rel. misfit norm = '+str(mis))
print('rel. noise norm = '+str(norm(noise_h)/norm(h_obs)))

# lake boundary estimates
# h_bdry = calc_bdry_h(h_obs,0.1)
true_bdry = calc_bdry_w(w_true,0.025)
map_bdry = calc_bdry_w(w_map,0.025)
samp_bdry = calc_bdry_w(sample,0.025)

# calculate volume change time series:
#dV_alt = calc_dV_h(h_obs,h_bdry)        # volume change estimate from h_obs alone
dV_map = calc_dV_w(w_map,map_bdry)        # volume change from inversion
dV_samp = calc_dV_w(sample,samp_bdry)        # volume change from inversion
dV_true = calc_dV_w(w_true,true_bdry)        # volume change from inversion


# plot everything
xy_str = H/1e3
for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    plt.figure(figsize=(12,7))

    plt.subplot(121)
    plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linewidth=3,label=r'true sol.')
    plt.plot(t0[0:i],dV_map[0:i]*(H**2)/1e9,color='crimson',linestyle='-',linewidth=3,label=r'MAP point')
    #plt.plot(t0[0:i],dV_alt[0:i]*(H**2)/1e9,color='crimson',linestyle='-.',linewidth=3,label=r'altimetry')
    plt.plot(t0[0:i],dV_samp[0:i]*(H**2)/1e9,color='k',linestyle='-',linewidth=1,label=r'posterior sample')

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(-0.2,0.4)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc='upper left')


    plt.subplot(222)
    plt.annotate(r'$\Delta h^\mathrm{obs}$',fontsize=20,xy=(27,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,h_obs[i,:,:].T,cmap='coolwarm',levels=np.arange(-0.5,0.55,0.1),extend='both')
#    plt.contour(xy_str*x0,xy_str*y0,h_bdry[:,:].T,colors='crimson',linestyles='-.',linewidths=3,levels=[1e-10])
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.tick_right()
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.gca().yaxis.set_label_position("right")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(224)
    plt.annotate(r'$w^\mathrm{MAP}$',fontsize=20,xy=(28,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,w_map[i,:,:].T,cmap='coolwarm',extend='both',levels=np.arange(-5,5.5,0.5))
    plt.contour(xy_str*x0,xy_str*y0,map_bdry[:,:].T,colors='crimson',linestyles='--',linewidths=3,levels=[1e-10])
    plt.contour(xy_str*x0,xy_str*y0,true_bdry[:,:].T,colors='royalblue',linewidths=3,levels=[1e-10])
    plt.contour(xy_str*x0,xy_str*y0,samp_bdry[:,:].T,colors='k',linewidths=2,levels=[1e-10])
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('../post_pngs/'+str(i))
    plt.close()
