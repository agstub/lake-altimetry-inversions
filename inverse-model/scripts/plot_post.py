import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h,calc_bdry_w,calc_bdry_h
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny
from error_model import noise_var

if os.path.isdir('../post_pngs')==False:
    os.mkdir('../post_pngs')

w_true = np.load(data_dir+'/w_true.npy')
w_map = np.load('../results/w_map.npy')
sample = np.load('../results/post_samples.npy')

h = np.load(data_dir+'/h.npy')
noise_h = np.random.normal(size=(Nt,Nx,Ny),scale=np.sqrt(noise_var))
h_obs = h + noise_h
h_obs = localize(h_obs)

num = np.shape(sample)[-1]

# lake boundary estimates
true_bdry = calc_bdry_w(w_true,0.025)
map_bdry = calc_bdry_w(w_map,0.025)

# calculate volume change time series:
dV_map = calc_dV_w(w_map,map_bdry)           # volume change from inversion
dV_true = calc_dV_w(w_true,true_bdry)        # volume change from inversion

samp_bdry = np.zeros((Ny,Nx,num))
dV_samp = np.zeros((Nt,num))

for l in range(num):
    print('sample '+str(l)+' out of '+str(num))
    samp_bdry[...,l] = calc_bdry_w(sample[...,l],0.025)
    dV_samp[:,l] = calc_dV_w(sample[...,l],samp_bdry[...,l])        # volume change from inversion

dV_var = np.sum((1./(num-1.))*(dV_map - dV_samp.T)**2,axis=0)
dV_sigma = np.sqrt(dV_var)

# plot everything
xy_str = H/1e3
for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    plt.figure(figsize=(12,7))

    plt.subplot(121)
    plt.plot(t0[0:i],dV_samp[0:i,:]*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.1)
    plt.plot(t0[0:i],dV_map[0:i]*(H**2)/1e9,color='forestgreen',linewidth=3,label=r'MAP point')
    #plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linestyle='-.',linewidth=3,label=r'true sol.')
    plt.plot(t0[0:i],10+(dV_samp[0:i,-1])*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.5,label='posterior sample')
    plt.plot(t0[0:i],(dV_map[0:i] + 3*dV_sigma[0:i] )*(H**2)/1e9,color='forestgreen',linestyle='--',linewidth=1,label=r'$3\sigma$')
    plt.plot(t0[0:i],(dV_map[0:i] - 3*dV_sigma[0:i] )*(H**2)/1e9,color='forestgreen',linestyle='--',linewidth=1)

    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(-0.1,0.35)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc='lower left')


    plt.subplot(222)
    plt.annotate(r'$\Delta h^\mathrm{obs}$',fontsize=20,xy=(27,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,h_obs[i,:,:].T,cmap='coolwarm',levels=np.arange(-0.5,0.55,0.1),extend='both')
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
    #for l in range(num):
     #  plt.contour(xy_str*x0,xy_str*y0,samp_bdry[...,l].T,colors='k',linewidths=2,levels=[1e-10],alpha=0.1)
    plt.contour(xy_str*x0,xy_str*y0,map_bdry[:,:].T,colors='forestgreen',linestyles='-',linewidths=3,levels=[1e-10])
    #plt.contour(xy_str*x0,xy_str*y0,map_bdry[:,:].T+3*bdry_sigma,colors='forestgreen',linestyles='--',linewidths=2,levels=[1e-10])
    #plt.contour(xy_str*x0,xy_str*y0,map_bdry[:,:].T-3*bdry_sigma,colors='forestgreen',linestyles='--',linewidths=2,levels=[1e-10])
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
