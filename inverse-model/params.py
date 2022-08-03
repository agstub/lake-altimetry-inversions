# this file sets the physical and numerical parameters

import numpy as np
from scipy.fft import fftfreq

#----------------------------regularization-------------------------------------
# set reguarization type

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'L2'

#----------------------import space and time arrays-----------------------------

# arrays from the data directory
t_d = np.load('../data/t.npy')       # time array (yr)
x_d = np.load('../data/x.npy')       # x array (m)
y_d = np.load('../data/y.npy')       # y array (m)
H_d = np.load('../data/H.npy')       # ice thickness (m)
beta_d = np.load('../data/beta.npy') # basal drag coeff (Pa s/m)

# number of grid points in each direction
Nx = np.size(x_d)                    # number of grid points in x-direction
Ny = np.size(y_d)                    # number of grid points in y-direction
Nt = np.size(t_d)                    # number of time steps

# initial elevation profile (used in RHS of normal equations)
h0 = np.load('../data/h_obs.npy')[0,:,:]+ np.zeros((Nt,Ny,Nx))

H = np.mean(H_d)                   # ice thickness over the lake
beta_e = np.mean(beta_d)           # basal drag coeff. near the lake

# scale and shift the data coordinates to be consistent with model formulation
t0 = (t_d-t_d[0])/365.0            # time in years, shifted so that t0[0]=0
x0 = (x_d-np.mean(x_d))/H          # x coordinate shifted and normalized by ice thickness
y0 = (y_d-np.mean(y_d))/H          # y coordinate shifted and normalized by ice thickness

t_final = t0[-1]

dt = np.abs(t0[1]-t0[0])           # grid spacing in t direction
dx = np.abs(x0[1]-x0[0])           # grid spacing in x direction'
dy = np.abs(y0[1]-y0[0])           # grid spacing in y direction


#---------------------- physical parameters ------------------------------------
# dimensional parameters
h_sc = 1                    # elevation anomaly scale (m)
t_sc = 3.154e7              # observational timescale (s) default 1 yr
eta = 1e14                  # Newtonian ice viscosity (Pa s)
rho_i = 917                 # ice density (kg/m^3)
g = 9.81                    # gravitational acceleration

#----------------------Derived parameters---------------------------------------
t_r = 2*eta/(rho_i*g*H)     # viscous relaxation time

# nondimensional parameters
lamda = t_sc/t_r           # process timescale relative to
                           # surface relaxation timescale

beta0 = beta_e*H/(2*eta)   # friction coefficient relative to ice viscosity

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-5               # stopping tolerance for conjugate gradient solver

max_cg_iter =  1000         # maximum conjugate gradient iterations

# discretization parameters

# frequency
kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some of the integral kernels
# have integrable or removable singularities at the zero frequency

kx0[0] = 1e-10
ky0[0] = 1e-10

# mesh grids for physical space domain
t,y,x = np.meshgrid(t0,y0,x0,indexing='ij')

# mesh grids for frequency domain
t,ky,kx = np.meshgrid(t0,ky0,kx0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
