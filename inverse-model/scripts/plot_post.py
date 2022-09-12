import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h,calc_bdry_w,calc_bdry_h
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny
from error_model import noise_var

if os.path.isdir('../post_pngs')==False:
    os.mkdir('../post_pngs')

if os.path.isfile(data_dir+'/w_true.npy')==True:
    w_true = np.load(data_dir+'/w_true.npy')

w_map = np.load('../results/w_map.npy')
sample = np.load('../results/post_samples.npy')
h_fwd = np.load('../results/h_fwd.npy')


h_obs = np.load(data_dir+'/h_obs.npy')

num = np.shape(sample)[-1]

thresh = 0.1

# lake boundary estimates
if os.path.isfile(data_dir+'/w_true.npy')==True:
    true_bdry = calc_bdry_w(w_true,thresh)
    dV_true = calc_dV_w(w_true,true_bdry)        # "true" volume change

map_bdry = calc_bdry_w(w_map,thresh)

# calculate volume change time series:
dV_map = calc_dV_w(w_map,map_bdry)           # volume change from inversion


samp_bdry = np.zeros((Ny,Nx,num))
dV_samp = np.zeros((Nt,num))

for l in range(num):
    print('sample '+str(l)+' out of '+str(num))
    samp_bdry[...,l] = calc_bdry_w(sample[...,l],thresh)
    dV_samp[:,l] = calc_dV_w(sample[...,l],samp_bdry[...,l])        # volume change from inversion

dV_var = np.sum((1./(num-1.))*(dV_map - dV_samp.T)**2,axis=0)
dV_sigma = np.sqrt(dV_var)

# plot everything
xy_str = H/1e3

xp = xy_str*x0
yp = xy_str*y0

dV_max = np.max(dV_map*(H**2)/1e9)
dV_min = np.min(dV_map*(H**2)/1e9)

for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    fig = plt.figure(figsize=(12,12))
    plt.suptitle(r'$t=$'+format(t0[i],'.2f')+' yr',y=1.05,fontsize=28,bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
    plt.subplot(212)
    plt.plot(t0[0:i],dV_samp[0:i,:]*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.1)
    plt.plot(t0[0:i],dV_map[0:i]*(H**2)/1e9,color='forestgreen',linewidth=3,label=r'MAP point')

    if os.path.isfile(data_dir+'/w_true.npy')==True:
        plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linestyle='-.',linewidth=3,label=r'true sol.')

    plt.plot(t0[0:i],10+(dV_samp[0:i,-1])*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.5,label='posterior sample')
    plt.plot(t0[0:i],(dV_map[0:i] + 3*dV_sigma[0:i] )*(H**2)/1e9,color='forestgreen',linestyle='--',linewidth=1.5,label=r'$3\sigma$')
    plt.plot(t0[0:i],(dV_map[0:i] - 3*dV_sigma[0:i] )*(H**2)/1e9,color='forestgreen',linestyle='--',linewidth=1.5)

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(dV_min-0.1,dV_max+0.1)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20,ncol=3,bbox_to_anchor=(0.93,-0.15))


    plt.subplot(221)
    sc = np.around(np.max(np.abs(h_obs))/1.,decimals=0)*1
    p=plt.contourf(xp,yp,h_obs[i,:,:].T,cmap='coolwarm',levels=sc*np.arange(-1.0,1.05,0.2),extend='both')
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
    sc = np.around(np.max(np.abs(w_map))/1.,decimals=0)*1
    p=plt.contourf(xp,yp,w_map[i,:,:].T,cmap='coolwarm',extend='both',levels=sc*np.arange(-1.0,1.05,0.2))
    #for l in range(num):
     #  plt.contour(xy_str*x0,xy_str*y0,samp_bdry[...,l].T,colors='k',linewidths=2,levels=[1e-10],alpha=0.1)
    plt.contour(xp,yp,map_bdry[:,:].T,colors='forestgreen',linestyles='-',linewidths=3,levels=[1e-5])
    plt.xlabel(r'$x$ (km)',fontsize=20)
    # plt.gca().yaxis.tick_right()
    # plt.gca().yaxis.set_label_position("right")
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


    plt.savefig('../post_pngs/'+str(i),bbox_inches='tight')
    plt.close()
