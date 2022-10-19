import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny,x,y,results_dir,t,lake_name
from matplotlib.transforms import Bbox
from shapely.geometry import Point

if lake_name != 'synth' and lake_name != 'nonlinear':
    from load_lakes import gdf

if os.path.isdir(results_dir+'/results_pngs')==False:
    os.mkdir(results_dir+'/results_pngs')

w_inv = np.load(results_dir+'/w_inv.npy')

h_obs = np.load(data_dir+'/h_obs.npy')
x_d = np.load(data_dir+'/x_d.npy')
y_d = np.load(data_dir+'/y_d.npy')
xp,yp = np.meshgrid(x_d,y_d)

if lake_name != 'synth' and lake_name != 'nonlinear':
    outline = gdf.loc[gdf['name']==lake_name]
    alt_bdry = 0*xp
    for i in range(x_d.size):
        for j in range(y_d.size):
            point = Point(x_d[i],y_d[j])
            alt_bdry[i,j] = outline.contains(point)
else:
    xc,yc = 0,0
    alt_bdry = 0*xp+1
    alt_bdry[np.sqrt((xp-xc)**2+(yp-yc)**2)>15] = 0

if os.path.isfile(data_dir+'/w_true.npy')==True:
    w_true = np.load(data_dir+'/w_true.npy')
    dV_true = calc_dV_w(w_true,alt_bdry)
else:
    dV_true = np.zeros(Nt)

# calculate volume change time series:
dV_inv = calc_dV_w(w_inv,alt_bdry)           # volume change from inversion

dV_h = calc_dV_h(h_obs,alt_bdry)

# plot everything

dV_max = np.max(dV_inv*(H**2)/1e9)+0.1
dV_min = np.min(dV_inv*(H**2)/1e9)-0.1

for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    fig = plt.figure(figsize=(12,12))
    plt.suptitle(r'(a) inversion and sensitivity at $t=$'+format(t0[i],'.1f')+' yr',y=1.04,x=0.4,fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    plt.subplot(212)
    plt.title(r'(b) volume change time series', fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1),y=0.91,x=0.3075)

    dV_mean = dV_inv[0:i]*(H**2)/1e9
    dV_alt = dV_h[0:i]*(H**2)/1e9

    if lake_name != 'synth' and lake_name != 'nonlinear':
        label = r'ICESat-2'
    else:
        label = r'altimetry'
        dV_t = dV_true[0:i]*(H**2)/1e9

    plt.plot(t0[0:i],dV_alt,color='seagreen',linewidth=4,linestyle='-.',label=label)
    plt.plot(t0[0:i],dV_mean,color='indigo',linewidth=4,label=r'inversion')
    plt.plot(t0[0:i],dV_t,color='k',linestyle='--',linewidth=4,label=r'true solution')

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,3)
    plt.ylim(dV_min-0.1,dV_max+0.1)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if lake_name != 'synth' and lake_name != 'nonlinear':
        plt.legend(fontsize=20,ncol=3,bbox_to_anchor=(0.985,-0.15))
    else:
        plt.legend(fontsize=20,ncol=3,bbox_to_anchor=(0.95,-0.15))

    plt.subplot(221)
    p=plt.contourf(xp,yp,h_obs[i,:,:],cmap='coolwarm',extend='both',levels=0.5*np.linspace(-1,1,6))

    if lake_name != 'synth' and lake_name != 'nonlinear':
        outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)
    else:
        plt.contour(xp,yp,alt_bdry,colors='k',linewidths=3,levels=[1e-10])

    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xlim(x_d.min(),x_d.max())
    plt.ylim(y_d.min(),y_d.max())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.13, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$\Delta h$ (m)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.subplot(222)
    p=plt.contourf(xp,yp,w_inv[i,:,:],cmap='coolwarm',levels=2.5*np.linspace(-1,1,6),extend='both')

    if lake_name != 'synth' and lake_name != 'nonlinear':
        outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)
    else:
        plt.contour(xp,yp,alt_bdry,colors='k',linewidths=3,levels=[1e-10])
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(x_d.min(),x_d.max())
    plt.ylim(y_d.min(),y_d.max())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.55, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$w_b$ (m/yr)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    plt.savefig(results_dir+'/results_pngs/'+str(i),bbox_inches=Bbox([[0.15,-0.25],[11.8,13]]))
    plt.close()
