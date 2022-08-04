import sys
sys.path.insert(0, '../source')

from inversion import invert
from operators import forward_w
from params import x,t,Nx,Nt,data_dir
import numpy as np
from kernel_fcns import ifftd,fftd
from conj_grad import norm
from post_process import calc_dV

L = 10
sigma = L/3

w_true = 5*np.exp(-0.5*(sigma**(-2))*x**2)*np.sin(4*np.pi*t/np.max(t))

h = ifftd(forward_w(w_true)).real

noise_h = np.random.normal(size=(Nt,Nx))
noise_level = 0.05
h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)

dV_true = calc_dV(w_true,L=20)

np.save(data_dir+'/wb.npy',w_true)
np.save(data_dir+'/h.npy',h_obs)
np.save(data_dir+'/dV.npy',dV_true)

w_inv,h_fwd,dV_inv,mis = invert(h_obs,5e-3)
