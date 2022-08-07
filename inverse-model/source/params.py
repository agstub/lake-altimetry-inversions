# this file sets the physical and numerical parameters

import numpy as np
from scipy.fft import fftfreq

# SET DATA DIRECTORY

data_dir = '../data_synth_lin'

#----------------------------regularization-------------------------------------
# set reguarization type

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'H1'

#----------------------import space and time arrays-----------------------------

# arrays from the data directory
t0 = np.load(data_dir+'/t.npy')       # time array (yr)
x0 = np.load(data_dir+'/x.npy')       # x array (m)
y0 = np.load(data_dir+'/y.npy')       # y array (m)

H = np.load(data_dir+'/H.npy').mean()       # ice thickness (m)
beta = np.load(data_dir+'/beta.npy').mean() # basal drag coeff (Pa s/m)
eta = np.load(data_dir+'/eta.npy').mean()     # Newtonian ice viscosity (Pa s)
u = np.load(data_dir+'/u.npy').mean()     # mean surface velocity (Pa s)

# initial elevation profile (used in RHS of normal equations)
#h0 = np.load(data_dir+'/h.npy')[0,:,:]+ np.zeros(np.shape(h_obs))

#-------------------------------------------------------------------------------

# number of grid points in each direction
Nx = np.size(x0)                    # number of grid points in x-direction
Ny = np.size(y0)                    # number of grid points in y-direction
Nt = np.size(t0)                    # number of time steps


t_final = t0[-1]

dt = np.abs(t0[1]-t0[0])           # grid spacing in t direction
dx = np.abs(x0[1]-x0[0])           # grid spacing in x direction'

dy = np.abs(y0[1]-y0[0])           # grid spacing in y direction


#---------------------- physical parameters ------------------------------------
# dimensional parameters
h_sc = 1                    # elevation anomaly scale (m)
t_sc = 3.154e7              # observational timescale (s) default 1 yr
rho_i = 917                 # ice density (kg/m^3)
g = 9.81                    # gravitational acceleration

#----------------------Derived parameters---------------------------------------
t_r = 2*eta/(rho_i*g*H)     # viscous relaxation time

# nondimensional parameters
lamda = t_sc/t_r           # process timescale relative to
                           # surface relaxation timescale

beta0 = beta*H/(2*eta)   # friction coefficient relative to ice viscosity

uh0 = u/H                  # advection

ub = u                     # background sliding speed (scaled)

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-5               # stopping tolerance for conjugate gradient solver

max_cg_iter =  1000         # maximum conjugate gradient iterations

# discretization parameters

# frequency
kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some of the integral kernels
# have integrable or removable singularities at the zero frequency

kx0[0] = 1e-15
ky0[0] = 1e-15

# mesh grids for physical space domain
t,y,x = np.meshgrid(t0,y0,x0,indexing='ij')

# mesh grids for frequency domain
t,ky,kx = np.meshgrid(t0,ky0,kx0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)


h0 = 0*x
