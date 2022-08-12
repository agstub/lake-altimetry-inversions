# example script running the inversion on some synthetic data
# and plotting the results

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import x,y,t,Nt,data_dir,H,x0,y0,t0,Nx,Ny
import numpy as np
from post_process import calc_dV_w,calc_dV_h
import os
if os.path.isdir('../pngs')==False:
    os.mkdir('../pngs')    # make a directory for the results.


import matplotlib.pyplot as plt
from localization import localize
from conj_grad import norm

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h = np.load(data_dir+'/h.npy')
noise_h = np.random.normal(size=(Nt,Nx,Ny))
noise_level = 0.25
h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)
h_obs = localize(h_obs)

w_true = np.load(data_dir+'/w_true.npy')

w_inv,h_fwd,mis = invert(h_obs,eps_1=1e-2,eps_2=5e0)

print('misfit norm = '+str(mis))

# lake boundary estimate (h_bdry) = 1 inside lake, 0 outside lake
h_bdry = 1+0*x
h_bdry[np.abs(h_obs)<0.1] = 0
h_bdry = np.mean(h_bdry,axis=0)

# boundary of true basal vertical velocity ("true" lake boundary)
w_bdry = 1+0*x
w_bdry[np.abs(w_true)<0.02*np.max(np.abs(w_true))] = 0
w_bdry = np.mean(w_bdry,axis=0)

# boundary of basal vertical velocity inversion (estimated lake boundary)
inv_bdry = 1+0*x
inv_bdry[np.abs(w_inv)<0.02*np.max(np.abs(w_inv))] = 0
inv_bdry = np.mean(inv_bdry,axis=0)

# calculate volume change time series:
dV_true = calc_dV_w(w_true,w_bdry)      # true volume change (from w_true)
dV_alt = calc_dV_h(h_obs,h_bdry)        # volume change estimate from h_obs alone
dV_inv = calc_dV_w(w_inv,inv_bdry)        # volume change from inversion

# calculate the estimated vs true highstand time for illustration
hs_true = np.argmax((dV_true))
hs_inv = np.argmax((dV_inv))
hs_alt = np.argmax((dV_alt))

# plot everything
xy_str = H/1e3
for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    plt.figure(figsize=(12,7))

    plt.subplot(121)
    plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='forestgreen',linewidth=3,label=r'true sol.')
    plt.plot(t0[0:i],dV_inv[0:i]*(H**2)/1e9,color='k',linestyle='--',linewidth=3,label=r'inversion')
    plt.plot(t0[0:i],dV_alt[0:i]*(H**2)/1e9,color='crimson',linestyle='-.',linewidth=3,label=r'altimetry')


    if i >= hs_true:
        plt.plot([t0[hs_true]],[dV_true[hs_true]*(H**2)/1e9],color='forestgreen',marker='o',markersize=10)
    if i >= hs_inv:
        plt.plot([t0[hs_inv]],[dV_inv[hs_inv]*(H**2)/1e9],color='k',marker='o',markersize=10)
    if i >= hs_alt:
        plt.plot([t0[hs_alt]],[dV_alt[hs_alt]*(H**2)/1e9],color='crimson',marker='o',markersize=10)

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(-0.3,0.3)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc='upper left')


    plt.subplot(222)
    plt.annotate(r'$\Delta h^\mathrm{obs}$',fontsize=20,xy=(27,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,h_obs[i,:,:].T,cmap='coolwarm',levels=np.arange(-0.5,0.55,0.05),extend='both')
    plt.contour(xy_str*x0,xy_str*y0,h_bdry[:,:].T,colors='crimson',linestyles='-.',linewidths=3,levels=[1e-2])
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.tick_right()
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.gca().yaxis.set_label_position("right")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(224)
    plt.annotate(r'$w_b^\mathrm{inv}$',fontsize=20,xy=(30,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,w_inv[i,:,:].T,cmap='coolwarm',extend='both',levels=np.arange(-5,5.5,0.5))
#    plt.contour(xy_str*x0,xy_str*y0,inv_bdry[:,:].T,colors='k',linestyles='--',linewidths=3,levels=[1e-1])
#    plt.contour(xy_str*x0,xy_str*y0,w_bdry[:,:].T,colors='forestgreen',linewidths=3,levels=[1e-1])
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('../pngs/'+str(i))
    plt.close()
