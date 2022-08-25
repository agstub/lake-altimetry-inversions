# example script running the inversion on some synthetic data
# and plotting the results
import sys
sys.path.insert(0, '../source')
from inversion import invert
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny,x,y
import numpy as np
from post_process import calc_dV_w,calc_bdry_w
import os
import matplotlib.pyplot as plt
from noise import noise_var
from localization import localize
from conj_grad import norm

# load synthetic elevation data (h_obs) and "true" basal vertical velocity (w_true)
h = np.load(data_dir+'/h.npy')
noise_h = np.random.normal(size=(Nt,Nx,Ny),scale=np.sqrt(noise_var))
h_obs = h + noise_h
h_obs = localize(h_obs)

B = 1+0*x
B[np.sqrt(x**2+y**2)>20]=0

w_true = np.load(data_dir+'/w_true.npy')

eps_1 = 3
eps_2 = 10
num = 200

batch = 200

N = int(num/batch)

plt.figure(figsize=(8,6))

for k in range(N):
    print('\n ***Drawing samples '+str(k*batch)+'-'+str((k+1)*batch)+' out of '+str(num)+'*** \n')

    w_map,sample,h_fwd,mis = invert(h_obs,kappa=0.05,tau=60,a=5,num=batch)

    print('rel. misfit norm = '+str(mis))
    print('rel. noise norm = '+str(norm(noise_h)/norm(h_obs)))

    # calculate volume change time series:

    for i in range(batch):
        dV_samp_i = calc_dV_w(sample[:,:,:,i],B)        # volume change from inversion

        plt.plot(t0,dV_samp_i*(H**2)/1e9,color='k',linestyle='-',linewidth=1,alpha=0.1)

dV_map = calc_dV_w(w_map,B)
dV_true = calc_dV_w(w_true,B)

dV_samp_i = calc_dV_w(sample[:,:,:,i],B)        # volume change from inversion
plt.plot(t0,1+dV_samp_i*(H**2)/1e9,color='k',linestyle='-',linewidth=1,label=r'posterior samples',alpha=0.5)
plt.plot(t0,dV_map*(H**2)/1e9,color='crimson',linestyle='-',linewidth=4,label=r'MAP point')
plt.plot(t0,dV_true*(H**2)/1e9,color='royalblue',linestyle='--',linewidth=4,label=r'true solution')
plt.ylim(-0.2,0.4)
plt.xlabel(r'$t$ (yr)',fontsize=20)
plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14,loc='lower right')

plt.tight_layout()
plt.savefig('test',bbox_inches='tight')
plt.close()
