# this file sets the physical and numerical parameters

import numpy as np
from scipy.fft import fftfreq
import sys
from metavars import lake_name
# SET DATA DIRECTORY

data_dir = '../data_'+lake_name
results_dir = '../results_'+lake_name


#----------------------import space and time arrays-----------------------------

# arrays from the data directory
t0 = np.load(data_dir+'/t.npy')       # time array (yr)
x0 = np.load(data_dir+'/x.npy')       # x array (scaled by H)
y0 = np.load(data_dir+'/y.npy')       # y array (scaled  by H)

H = np.load(data_dir+'/H.npy').mean()       # ice thickness (m)
beta_d = np.load(data_dir+'/beta.npy').mean() # basal drag coeff (Pa s/m)
eta_d = np.load(data_dir+'/eta.npy').mean()     # Newtonian ice viscosity (Pa s)
u_d = np.load(data_dir+'/u.npy').mean()     # horizontal x ice velocity (m / yr)
v_d = np.load(data_dir+'/v.npy').mean()     # horizontal y ice velocity (m / yr)

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
# dimensional parameters (elevation scale set to 1 meter)
t_sc = 3.154e7              # observational timescale (s) default 1 yr
rho_i = 917                 # ice density (kg/m^3)
g = 9.81                    # gravitational acceleration

#----------------------Derived parameters---------------------------------------
t_r = 2*eta_d/(rho_i*g*H)     # viscous relaxation time

# nondimensional parameters
lamda0 = t_sc/t_r           # process timescale relative to
                            # surface relaxation timescale

beta0 = beta_d*H/(2*eta_d) # drag coefficient relative to ice viscosity and thickness

u0 = u_d/H                 # advection param. (x direction)
v0 = v_d/H                 # advection param. (y direction)

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-6              # stopping tolerance (relative) for conjugate gradient solver

max_cg_iter =  10000       # maximum conjugate gradient iterations


# spatial frequencies
kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some of the integral kernels
# have removable singularities at the zero frequency

kx0[0] = 1e-20
ky0[0] = 1e-20

# mesh grids for physical space domain
t,y,x = np.meshgrid(t0,y0,x0,indexing='ij')

# mesh grids for frequency domain
t,ky,kx = np.meshgrid(t0,ky0,kx0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
