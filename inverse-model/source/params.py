# this file sets the physical and numerical parameters

import numpy as np
from scipy.fft import fftfreq

# SET DATA DIRECTORY

data_dir = '../data_nonlinear'

#----------------------------regularization-------------------------------------
# set reguarization type

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'L2'

#----------------------import space and time arrays-----------------------------

# arrays from the data directory
h_obs = np.load(data_dir+'/h.npy')

t_d = np.load(data_dir+'/t.npy')       # time array (yr)
x_d = np.load(data_dir+'/x.npy')       # x array (m)
y_d = np.load(data_dir+'/y.npy')       # y array (m)

H_d = np.load(data_dir+'/H.npy')       # ice thickness (m)
beta_d = np.load(data_dir+'/beta.npy') # basal drag coeff (Pa s/m)
eta_d = np.load(data_dir+'/eta.npy')     # Newtonian ice viscosity (Pa s)
u_d = np.load(data_dir+'/u.npy')     # mean surface velocity (Pa s)

# initial elevation profile (used in RHS of normal equations)
#h0 = np.load(data_dir+'/h.npy')[0,:,:]+ np.zeros(np.shape(h_obs))

#-------------------------------------------------------------------------------
H = 1000         #np.mean(H_d)         # ice thickness over the lake
beta_e = 1e8     #beta_d[0]            # basal drag coeff. near the lake
eta = 1e13       #eta_d[0]
u = 0            #u_d[0]

# print('{:.2e}'.format(eta))
# print('{:.2e}'.format(beta_e))
# print('{:.2e}'.format(u))
# print('{:.2e}'.format(H))

# scale and shift the data coordinates to be consistent with model formulation
t0 = np.linspace(0,6,100)#t_d          # time
x0 = x_d*1000/H #x_d                   # x coordinate
y0 = y_d*1000/H #x_d                   # y coordinate

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

beta0 = beta_e*H/(2*eta)   # friction coefficient relative to ice viscosity

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
