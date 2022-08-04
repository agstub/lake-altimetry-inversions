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

dim = np.size(np.shape(h_obs))-1     # spatial dimension

t_d = np.load(data_dir+'/t.npy')       # time array (yr)
x_d = np.load(data_dir+'/x.npy')       # x array (m)

if dim>1:
    y_d = np.load(data_dir+'/y.npy')       # y array (m)

H_d = np.load(data_dir+'/H.npy')       # ice thickness (m)
beta_d = np.load(data_dir+'/beta.npy') # basal drag coeff (Pa s/m)
eta_d = np.load(data_dir+'/eta.npy')     # Newtonian ice viscosity (Pa s)
u_d = np.load(data_dir+'/u.npy')     # mean surface velocity (Pa s)

# number of grid points in each direction
Nx = np.size(x_d)                    # number of grid points in x-direction
if dim>1:
    Ny = np.size(y_d)                    # number of grid points in y-direction
Nt = np.size(t_d)                    # number of time steps

# initial elevation profile (used in RHS of normal equations)
if dim>1:
    h0 = np.load(data_dir+'/h.npy')[0,:,:]+ np.zeros((Nt,Ny,Nx))
else:
    h0 = np.load(data_dir+'/h.npy')[0,:]+ np.zeros((Nt,Nx))

H = np.mean(H_d)                   # ice thickness over the lake
beta_e = beta_d[0]                 # basal drag coeff. near the lake
eta = eta_d[0]
u = u_d[0]

# print('{:.2e}'.format(eta))
# print('{:.2e}'.format(beta_e))
# print('{:.2e}'.format(u))
# print('{:.2e}'.format(H))



# scale and shift the data coordinates to be consistent with model formulation
t0 = t_d                         # time
x0 = x_d                         # x coordinate

if dim>1:
    y0 = y_d                         # y coordinate

t_final = t0[-1]

dt = np.abs(t0[1]-t0[0])           # grid spacing in t direction
dx = np.abs(x0[1]-x0[0])           # grid spacing in x direction'

if dim>1:
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

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-5               # stopping tolerance for conjugate gradient solver

max_cg_iter =  1000         # maximum conjugate gradient iterations

# discretization parameters

# frequency
kx0 =  fftfreq(Nx,dx)
if dim>1:
    ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some of the integral kernels
# have integrable or removable singularities at the zero frequency

kx0[0] = 1e-20
if dim>1:
    ky0[0] = 1e-20

# mesh grids for physical space domain
if dim>1:
    t,y,x = np.meshgrid(t0,y0,x0,indexing='ij')
else:
    t,x = np.meshgrid(t0,x0,indexing='ij')

# mesh grids for frequency domain
if dim>1:
    t,ky,kx = np.meshgrid(t0,ky0,kx0,indexing='ij')
else:
    t,kx = np.meshgrid(t0,kx0,indexing='ij')
    ky = 0*kx
    dy = dx
    y=x

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
