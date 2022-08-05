import sys
sys.path.insert(0, '../source')

from inversion import invert
from operators import forward_w
from params import x,y,t,Nx,Ny,Nt,data_dir,H,x0,y0,t0
import numpy as np
from kernel_fcns import ifftd,fftd
from conj_grad import norm
from post_process import calc_dV_w,calc_dV_h
from pre_process import trim
import os
if os.path.isdir('../pngs')==False:
    os.mkdir('../pngs')    # make a directory for the results.


import matplotlib.pyplot as plt

L = 10
sigma = (1000/H)*L/3

w_true = 5*np.exp(-0.5*(sigma**(-2))*(x**2+y**2))*np.sin(4*np.pi*t/np.max(t))

h = ifftd(forward_w(w_true)).real

h_mean = np.copy(h)
h_mean[np.sqrt(x**2+y**2)>5] = 0
h_mean = np.sum(h_mean,axis=(-1,-2))/np.sum(h_mean!=0,axis=(-1,-2))
i_est = np.argmax(np.abs(h_mean))

h_temp = 1+0*h[i_est,:,:]
h_temp[np.abs(h[i_est,:,:])<0.1] = 0
h_bdry = np.multiply.outer(np.ones(Nt),h_temp)

w_bdry = 1+0*x
w_bdry[np.sqrt(x**2+y**2)>3*sigma] = 0

noise_h = np.random.normal(size=(Nt,Nx,Ny))
noise_level = 0.25
h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)


w_inv,h_fwd,mis = invert(h_obs,7e-3)

print('misfit norm = '+str(mis))
#print('(noise level = '+str(noise_level)+')')

dV_true = calc_dV_w(w_true,w_bdry)
dV_alt = calc_dV_h(h_obs,h_bdry)
dV_inv = calc_dV_w(w_inv,w_bdry)

hs_true = np.argmax(np.abs(dV_true[0:int(Nt/2.)+10]))
hs_alt = np.argmax(np.abs(dV_alt[0:int(Nt/2.)+10]))


xy_str = H/1e3

for i in range(Nt):
    print('image '+str(i+1)+' out of '+str(Nt))

    plt.figure(figsize=(12,7))

    plt.subplot(121)
    plt.plot(t0[0:i],dV_true[0:i]*(H**2)/1e9,color='royalblue',linewidth=3,label=r'true sol.')
    plt.plot(t0[0:i],dV_alt[0:i]*(H**2)/1e9,color='crimson',linestyle='-.',linewidth=3,label=r'altimetry')
    plt.plot(t0[0:i],dV_inv[0:i]*(H**2)/1e9,color='k',linestyle='--',linewidth=3,label=r'inversion')

    if i >= hs_true:
        plt.plot([t0[hs_true]],[dV_true[hs_true]*(H**2)/1e9],color='royalblue',marker='o',markersize=10)
    if i >= hs_alt:
        plt.plot([t0[hs_alt]],[dV_alt[hs_alt]*(H**2)/1e9],color='crimson',marker='o',markersize=10)

    plt.ylabel(r'$\Delta V$ (km$^2$)',fontsize=20)
    plt.xlim(0,t0[-1])
    plt.ylim(-0.2,0.4)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc='lower right')


    plt.subplot(222)
    plt.annotate(r'$h_\mathrm{obs}$',fontsize=20,xy=(30,-37))
    p=plt.contourf(xy_str*x0,xy_str*y0,h_obs[i,:,:].T,cmap='coolwarm',levels=np.arange(-1,1,0.2),extend='both')
    plt.contour(xy_str*x0,xy_str*y0,h_bdry[0,:,:].T,colors='k',linewidths=2,levels=[1e-10])
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
    plt.contourf(xy_str*x0,xy_str*y0,w_inv[i,:,:].T,cmap='coolwarm',extend='both',levels=np.arange(-5,5,1))
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
