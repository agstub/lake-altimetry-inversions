import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny,x,y,results_dir
from error_model import noise_var

if os.path.isdir(results_dir+'/post_pngs')==False:
    os.mkdir(results_dir+'/post_pngs')

w_map = np.load(results_dir+'/w_map.npy')
sample = np.load(results_dir+'/post_samples.npy')

h_obs = np.load(data_dir+'/h_obs.npy')

num = np.shape(sample)[-1]

xc = -1
yc = -1
bdry = 0*x+1
bdry[np.sqrt((x-xc)**2+(y-yc)**2)>6] = 0
bdry = np.mean(bdry,axis=0)

if os.path.isfile(data_dir+'/w_true.npy')==True:
    w_true = np.load(data_dir+'/w_true.npy')
    dV_true = calc_dV_w(w_true,bdry)
else:
    dV_true = np.zeros(Nt)

# calculate volume change time series:
dV_map = calc_dV_w(w_map,bdry)           # volume change from inversion

dV_h = calc_dV_h(h_obs,bdry)

dV_samp = np.zeros((Nt,num))

for l in range(num):
    print('processing sample '+str(l)+' out of '+str(num))
    dV_samp[:,l] = calc_dV_w(sample[...,l],bdry)        # volume change from inversion

### sample variance of volume change estimate
dV_var = np.sum((1./(num-1.))*(dV_map - dV_samp.T)**2,axis=0)
dV_sigma = np.sqrt(dV_var)

### sample variance:
## w_var = np.sum((1./(num-1.))*(w_map - np.transpose(sample,axes=(-1,0,1,2)))**2,axis=0)

# plot everything
xy_str = H/1e3

xp = xy_str*x0
yp = xy_str*y0

dV_max = np.max(dV_samp*(H**2)/1e9)
dV_min = np.min(dV_samp*(H**2)/1e9)

for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    fig = plt.figure(figsize=(12,12))
    plt.suptitle(r'$t=$'+format(t0[i],'.2f')+' yr',y=1.05,fontsize=28,bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
    plt.subplot(212)
    plt.plot(t0[0:i],dV_samp[0:i,:]*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.1)

    dV_mean = dV_map[0:i]*(H**2)/1e9

    plt.plot(t0[0:i],dV_mean,color='forestgreen',linewidth=3,label=r'MAP point')

    if np.max(np.abs(dV_true))>0:
        plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linestyle='-.',linewidth=3,label=r'true sol.')

    dV_high = (dV_map[0:i] + 3*dV_sigma[0:i] )*(H**2)/1e9
    dV_low = (dV_map[0:i] - 3*dV_sigma[0:i] )*(H**2)/1e9
    plt.plot(t0[0:i],10+(dV_samp[0:i,-1])*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.5,label='posterior sample')
    plt.plot(t0[0:i],dV_high,color='forestgreen',linestyle='--',linewidth=1.5,label=r'$3\sigma$')
    plt.plot(t0[0:i],dV_low,color='forestgreen',linestyle='--',linewidth=1.5)

    dV_alt = dV_h[0:i]*(H**2)/1e9
    plt.plot(t0[0:i],dV_alt,color='crimson',linestyle='-.',linewidth=3,label=r'altimetry')

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(dV_min-0.1,dV_max+0.1)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20,ncol=4,bbox_to_anchor=(1.05,-0.15))

    plt.subplot(221)
    sc = np.around(np.max(np.abs(h_obs))/5.,decimals=0)*5
    p=plt.contourf(xp,yp,h_obs[i,:,:].T,cmap='coolwarm',levels=sc*np.arange(-1.0,1.05,0.2),extend='both')
    plt.contour(xp,yp,bdry.T,colors='k',linestyles='--',linewidths=2,levels=[1e-5])

    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xlim(xp.min(),xp.max())
    plt.ylim(yp.min(),yp.max())
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.13, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$\Delta h^\mathrm{obs}$ (m)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.subplot(222)
    sc = np.around(np.max(np.abs(w_map))/5.,decimals=0)*5
    p=plt.contourf(xp,yp,w_map[i,:,:].T,cmap='coolwarm',extend='both',levels=sc*np.arange(-1.0,1.05,0.2))

    plt.contour(xp,yp,bdry.T,colors='k',linestyles='--',linewidths=2,levels=[1e-5])
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xp.min(),xp.max())
    plt.ylim(yp.min(),yp.max())
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.55, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$w^\mathrm{MAP}$ (m/yr)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.savefig(results_dir+'/post_pngs/'+str(i),bbox_inches='tight')
    plt.close()
