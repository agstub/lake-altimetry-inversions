import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h,calc_bdry_w
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny,x,y,results_dir,t,ind
from error_model import noise_var
from matplotlib.transforms import Bbox
from load_lakes import gdf
from shapely.geometry import Point

if os.path.isdir(results_dir+'/post_pngs')==False:
    os.mkdir(results_dir+'/post_pngs')

w_map = np.nan_to_num(np.load(results_dir+'/w_map.npy'))
sample = np.nan_to_num(np.load(results_dir+'/post_samples.npy'))

h_obs = np.load(data_dir+'/h_obs.npy')
x_d = np.load(data_dir+'/x_d.npy')
y_d = np.load(data_dir+'/y_d.npy')


num = np.shape(sample)[-1]

outline = gdf.loc[gdf['name']=='Cook_E2']

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Define boundary for volume change integration !!!!
xp,yp = np.meshgrid(x_d,y_d)

xc,yc = xp.mean()+1,yp.mean()+1         # Nimrod-2
bdry = 0*xp+1
bdry[np.sqrt((xp-xc)**2+(yp-yc)**2)>7] = 0

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

alt_bdry = 0*xp

for i in range(x_d.size):
    for j in range(y_d.size):
        point = Point(x_d[i],y_d[j])
        alt_bdry[i,j] = outline.contains(point)


if os.path.isfile(data_dir+'/w_true.npy')==True:
    w_true = np.load(data_dir+'/w_true.npy')
    dV_true = calc_dV_w(w_true,bdry)
else:
    dV_true = np.zeros(Nt)

# calculate volume change time series:
dV_map = calc_dV_w(w_map,alt_bdry)           # volume change from inversion


dV_h = calc_dV_h(h_obs,alt_bdry)

dV_samp = np.zeros((Nt,num))

for l in range(num):
    print('processing sample '+str(l)+' out of '+str(num))
    dV_samp[:,l] = calc_dV_w(sample[...,l],alt_bdry)        # volume change from inversion

### sample variance of volume change estimate
dV_var = np.sum((1./(num-1.))*(dV_map - dV_samp.T)**2,axis=0)
dV_sigma = np.sqrt(dV_var)

### sample variance:
#w_var = np.sum((1./(num-1.))*(w_map - np.transpose(sample,axes=(-1,0,1,2)))**2,axis=0)

# plot everything

dV_max = np.max(dV_map*(H**2)/1e9)+0.25
dV_min = np.min(dV_map*(H**2)/1e9)-0.25

ind_0 = np.argmax(ind[:,0,0])


for i in np.arange(ind_0,Nt-ind_0,1,dtype='int'):
    print('image '+str(i+1)+' out of '+str(Nt))

    fig = plt.figure(figsize=(12,12))
    plt.suptitle(r'(a) data and inversion at $t=$'+format(t0[i],'.1f')+' yr',y=1.04,x=0.4,fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    plt.subplot(212)
    plt.title(r'(b) volume change time series', fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1),y=0.91,x=0.3075)
    if num>2:
        plt.plot(t0[0:i],dV_samp[0:i,:]*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.1)

    dV_mean = dV_map[0:i]*(H**2)/1e9
    dV_alt = dV_h[0:i]*(H**2)/1e9
    dV_high = (dV_map[0:i] + 3*dV_sigma[0:i] )*(H**2)/1e9
    dV_low = (dV_map[0:i] - 3*dV_sigma[0:i] )*(H**2)/1e9

    plt.plot(t0[0:i],dV_alt,color='indigo',linewidth=4,linestyle='-',label=r'ICESat-2')
    plt.plot(t0[0:i],dV_mean,color='mediumseagreen',linewidth=4,linestyle='-.',label=r'posterior mean')

    if np.max(np.abs(dV_true))>0:
        plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linestyle='-.',linewidth=3,label=r'true sol.')

    plt.plot(t0[0:i],10+(dV_samp[0:i,-1])*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.5,label='posterior sample')
    plt.plot(t0[0:i],dV_high,color='mediumseagreen',linestyle='--',linewidth=2,label=r'$3\sigma$')
    plt.plot(t0[0:i],dV_low,color='mediumseagreen',linestyle='--',linewidth=2)

    dV_alt = dV_h[0:i]*(H**2)/1e9

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,2.5)
    plt.ylim(dV_min-0.1,dV_max+0.1)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if num > 2:
        plt.legend(fontsize=20,ncol=4,bbox_to_anchor=(1.1,-0.15))
    else:
        plt.legend(fontsize=20,ncol=2,bbox_to_anchor=(0.75,-0.15))

    plt.subplot(221)
    sc = np.around(np.max(bdry*np.abs(h_obs))/2.,decimals=0)*2
    p=plt.contourf(xp,yp,h_obs[i,:,:],cmap='coolwarm',levels=sc*np.arange(-1.0,1.05,0.2),extend='both')
    outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)

    plt.annotate(r'ICESat-2 ATL15',xy=(xp.mean()-15,yp.min()+2),fontsize=22,fontweight=625)
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xlim(x_d.min(),x_d.max())
    plt.ylim(y_d.min(),y_d.max())
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.13, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$\Delta h^\mathrm{anom}$ (m)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.subplot(222)
    sc = np.around(np.max(bdry*np.abs(w_map))/5.,decimals=0)*5
    p=plt.contourf(xp,yp,w_map[i,:,:],cmap='coolwarm',extend='both',levels=sc*np.arange(-1.0,1.05,0.2))
    plt.annotate(r'inversion',xy=(xp.mean()-10,yp.min()+2),fontsize=22,fontweight=625)
    outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(x_d.min(),x_d.max())
    plt.ylim(y_d.min(),y_d.max())
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect('equal', 'box')
    cbar_ax = fig.add_axes([0.55, 0.9, 0.35, 0.02])
    cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
    cbar.set_label(r'$w_b$ (m/yr)',fontsize=24,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    plt.savefig(results_dir+'/post_pngs/'+str(i),bbox_inches=Bbox([[0.15,-0.25],[11.8,13]]))
    plt.close()



plt.plot(dV_alt[5:51]/dV_mean[5:51])
plt.show()
plt.close()
