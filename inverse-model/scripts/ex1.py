# example script running the inversion on some synthetic data
# and plotting the results

import sys
sys.path.insert(0, '../source')

from inversion import invert
from params import x,y,t,Nt,data_dir,H,x0,y0,t0
import numpy as np
from post_process import calc_dV_w,calc_dV_h
import os
if os.path.isdir('../pngs')==False:
    os.mkdir('../pngs')    # make a directory for the results.


import matplotlib.pyplot as plt

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h_obs = np.load(data_dir+'/h.npy')
w_true = np.load(data_dir+'/w_true.npy')

w_inv,h_fwd,mis = invert(h_obs,eps_1=5e-2,eps_2=5e0)

print('misfit norm = '+str(mis))

# lake boundary estimate (h_bdry) = 1 inside lake, 0 outside lake
h_bdry = 1+0*x
h_bdry[np.abs(h_obs)<0.1] = 0
h_bdry = np.mean(h_bdry,axis=0)

# boundary of true basal vertical velocity ("true" lake boundary)
w_bdry = 1+0*x
w_bdry[np.sqrt(x**2+y**2)>1e4/H] = 0

# boundary of basal vertical velocity inversion (estimated lake boundary)
inv_bdry = 1+0*x
inv_bdry[np.abs(w_inv)<0.01*np.max(np.abs(w_inv))] = 0
inv_bdry = np.mean(inv_bdry,axis=0)

# calculate volume change time series:
dV_true = calc_dV_w(w_true,w_bdry)      # true volume change (from w_true)
dV_alt = calc_dV_h(h_obs,h_bdry)        # volume change estimate from h_obs alone
dV_inv = calc_dV_w(w_inv,inv_bdry)        # volume change from inversion

# calculate the estimated vs true highstand time for illustration
hs_true = np.argmax((dV_true[0:int(Nt/2.)+10]))
hs_inv = np.argmax((dV_inv[0:int(Nt/2.)+10]))
hs_alt = np.argmax((dV_alt[0:int(Nt/2.)+10]))

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
    plt.ylim(-0.2,0.4)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc='lower right')


    plt.subplot(222)
    plt.annotate(r'$h_\mathrm{obs}$',fontsize=20,xy=(30,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,h_obs[i,:,:].T,cmap='coolwarm',levels=np.arange(-1,1,0.2),extend='both')
    plt.contour(xy_str*x0,xy_str*y0,h_bdry[:,:].T,colors='crimson',linestyles='-.',linewidths=3,levels=[1e-10])
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.tick_right()
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.gca().yaxis.set_label_position("right")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(224)
    plt.annotate(r'$w_\mathrm{inv}$',fontsize=20,xy=(30,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,w_inv[i,:,:].T,cmap='coolwarm',extend='both',levels=np.arange(-5,5,1))
    plt.contour(xy_str*x0,xy_str*y0,inv_bdry[:,:].T,colors='k',linestyles='--',linewidths=3,levels=[1e-1])
    plt.contour(xy_str*x0,xy_str*y0,w_bdry[0,:,:].T,colors='forestgreen',linewidths=3,levels=[1e-10])
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
