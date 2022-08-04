import sys
sys.path.insert(0, '../source')

from inversion import invert
from operators import forward_w
from params import x,t,Nx,Nt,data_dir,H,beta0
import numpy as np
from kernel_fcns import ifftd,fftd
from conj_grad import norm
from post_process import calc_dV

def ind(x):
    L = 10
    I = 0*x + 1
    I[np.abs(x)>L] = 0
    return I

beta_true = -beta0*ind(x)

# import matplotlib.pyplot as plt
# plt.contourf(t,x,beta_true)
# plt.show()
#
# h = ifftd(forward_w(w_true)).real
#
# noise_h = np.random.normal(size=(Nt,Nx))
# noise_level = 0.05
# h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)
#
# dV_true = calc_dV(w_true,L=20)*H  # dimensional (m^2)
#
# np.save(data_dir+'/wb.npy',w_true)
# np.save(data_dir+'/h.npy',h_obs)
# np.save(data_dir+'/dV.npy',dV_true)
#
# w_inv,h_fwd,dV_inv,mis = invert(h_obs,3e-3)
#
# print(mis)
