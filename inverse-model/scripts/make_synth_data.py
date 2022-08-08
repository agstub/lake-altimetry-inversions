# this script makes synthetic elevation data from a prescribed basal vertical
# velocity anomaly. the forward operator is applied to the velocity anomaly,
# producing the elevation anomaly, then the elevation anomaly is localized
# to remove off-lake deformation (in practice these are interpreted as regional trends
# but a small component may be due to the lake activity), and some noise is added
# to the synthetic elevation field

import sys
sys.path.insert(0, '../source')

import numpy as np
import os

# make a directory for the data
if os.path.isdir('../data_synth_lin')==False:
    os.mkdir('../data_synth_lin')

# define the scalar parameters in the problem
# in practice we might use means of some of these fields around the lake
H = np.array([1000])          # ice thickness over the lake
beta = np.array([1e8])        # basal drag coeff. near the lake
eta = np.array([1e13])        # viscosity near the lake
u = np.array([0])             # background flow speed near the lake

# define coordinate arrays
t0 = np.linspace(0,6,100)                     # time
x0 = np.linspace(-40,40,101)*1000/H.mean()    # x coordinate
y0 = np.linspace(-40,40,101)*1000/H.mean()    # y coordinate

# save everything so far in the data directory
np.save('../data_synth_lin/t.npy',t0)
np.save('../data_synth_lin/x.npy',x0)
np.save('../data_synth_lin/y.npy',y0)
np.save('../data_synth_lin/eta.npy',eta)
np.save('../data_synth_lin/beta.npy',beta)
np.save('../data_synth_lin/H.npy',H)
np.save('../data_synth_lin/u.npy',u)

# import some functions to produce the synthetic elevation anomaly
from params import t,x,y,Nt,Nx,Ny
from operators import fwd
from localization import localize
from kernel_fcns import ifftd
from conj_grad import norm

# set basal vertical velocity anomaly to an oscillating gaussian
sigma = (10000.0/H)/3.0
w_true = 5*np.exp(-0.5*(sigma**(-2))*(x**2+y**2))*np.sin(4*np.pi*t/np.max(t))

# produce synthetic elevation anomaly by applying the forward operator
# (returns fft of elevation), inverse fourier-transforming the results,
# and removing the off-lake component ("localize" function)
h = localize(ifftd(fwd(w_true)).real)

# add some noise
noise_h = np.random.normal(size=(Nt,Nx,Ny))
noise_level = 0.25
h_obs = h + noise_level*norm(h)*noise_h/norm(noise_h)

np.save('../data_synth_lin/h.npy',h_obs)
np.save('../data_synth_lin/w_true.npy',w_true)
