import sys
sys.path.insert(0, '../source')

import numpy as np
import os

if os.path.isdir('../data_synth_lin')==False:
    os.mkdir('../data_synth_lin')    # make a directory for the results.


H = np.array([1000])          # ice thickness over the lake
beta = np.array([1e8])        # basal drag coeff. near the lake
eta = np.array([1e13])        # viscosity
u = np.array([0])             # background flow speed


t0 = np.linspace(0,6,100)              # time
x0 = np.linspace(-40,40,101)*1000/H.mean()    # x coordinate
y0 = np.linspace(-40,40,101)*1000/H.mean()    # y coordinate

np.save('../data_synth_lin/t.npy',t0)
np.save('../data_synth_lin/x.npy',x0)
np.save('../data_synth_lin/y.npy',y0)
np.save('../data_synth_lin/eta.npy',eta)
np.save('../data_synth_lin/beta.npy',beta)
np.save('../data_synth_lin/H.npy',H)
np.save('../data_synth_lin/u.npy',u)

from params import t,x,y,Nt,Nx,Ny
from operators import forward_w
from localization import localize
from kernel_fcns import ifftd
from conj_grad import norm

L = 10
sigma = (1000/H)*L/3

w_true = 5*np.exp(-0.5*(sigma**(-2))*(x**2+y**2))*np.sin(4*np.pi*t/np.max(t))

h = localize(ifftd(forward_w(w_true)).real)

noise_h = np.random.normal(size=(Nt,Nx,Ny))
noise_level = 0.25
h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)

np.save('../data_synth_lin/h.npy',h_obs)
np.save('../data_synth_lin/w_true.npy',w_true)
