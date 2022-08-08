#-------------------------------------------------------------------------------
# This FEniCS program simulates ice flow over a subglacial lake undergoing water
# volume changes.
#
# This is the main file that calls the stoke solver and free surface evolution
# functions at each timestep, and saves the results.
#-------------------------------------------------------------------------------

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from stokes import stokes_solve,get_zero
from geometry import interface,bed
from meshfcns import mesh_routine,get_wb
import scipy.integrate as scpint
import os
from params import (rho_i,g,tol,t_final,Lngth,Hght,nt,dt,rho_w,dy,
                    print_convergence,X_fine,nx,Nx,Ny,save_vtk,i0)

#--------------------Initial conditions-----------------------------------------
# compute initial mean elevation of ice-water interface and initial lake volume.
s_mean0 = np.mean(interface(X_fine)[interface(X_fine)-bed(X_fine)>tol])
lake_vol_0 = 2*scpint.quad(lambda x: interface(x)-bed(x),0,0.5*Lngth,full_output=1)[0]
#-------------------------------------------------------------------------------

resultsname = 'results'

if os.path.isdir(resultsname)==False:
    os.mkdir(resultsname)    # make a directory for the results.

if print_convergence == 'off':
    set_log_level(40)    # suppress Newton convergence information if desired.

# create VTK files
if save_vtk == 'on':
    vtkfile_u = File(resultsname+'/stokes/u.pvd')
    vtkfile_p = File(resultsname+'/stokes/p.pvd')

# create mesh
p0 = Point((0.0,0.0))
p1 = Point((0.5*Lngth,Hght))
mesh = RectangleMesh(p0,p1, Nx, Ny,diagonal="left/right")

M = mesh.coordinates()
# make sure all vertices are bounded below by the bed elevation
M[:,1][M[:,1]<bed(M[:,0])] = bed(M[:,0])[M[:,1]<bed(M[:,0])]

# define arrays for saving surfaces, lake volume, basal vertical velocity, viscosity,
# horizontal surface velocity, and basal drag over time.
s = np.zeros((nx,nt))       # basal surface elevation
h = np.zeros((nx,nt))       # upper surface elevation
wb = np.zeros((nx,nt))      # basal surface normal (i.e. ~vertical) velocity
lake_vol = np.zeros(nt)     # lake volume

eta_mean  = np.zeros(nt)    # mean viscosity
beta_mean  = np.zeros(nt)   # mean basal drag
u_mean = np.zeros(nt)       # mean horizontal surface velocity

t = 0                             # time

# begin time stepping
for i in range(nt):

    print('-----------------------------------------------')
    print('Timestep '+str(i+1)+' out of '+str(nt))

    if t==0:
        # set initial conditions.
        s_mean_i = s_mean0                    # Mean ice-water elevation.
        w = get_zero(mesh)
        mesh,s_int,h_int,sx_fn = mesh_routine(w,mesh,dt)
        h_int = lambda x: Hght # Ice-air surface function
        s_int = lambda x: interface(x)            # Lower surface function

    # solve the Stoke problem, returns solution "w", along with the mean basal drag "beta",
    # the mean viscosity "eta", and the mean horizontal surface velocity "u"
    w,beta_i,eta_i,u_i = stokes_solve(mesh,lake_vol_0,s_mean_i,h_int,s_int,t)

    # solve the surface kinematic equations and move the mesh.
    mesh,s_int,h_int,sx_fn = mesh_routine(w,mesh,dt)

    # extract the basal vertical velocity "wb" from the solution
    wb_int = get_wb(w,mesh,sx_fn)

    # save quantities of interest
    s[:,i] = s_int(X_fine)
    h[:,i] = h_int(X_fine)
    wb[:,i] = wb_int(X_fine)
    eta_mean[i] = eta_i
    beta_mean[i] = beta_i
    u_mean[i] = u_i

    # compute lake volume: integral of lower surface minus the bed elevation
    lake_vol[i] = 2*scpint.quad(lambda x: s_int(x)-bed(x),0,0.5*Lngth,full_output=1)[0]


    # save Stokes solution if desired
    if save_vtk == 'on':
        _u, _p,_pw = w.split()
        _u.rename("vel", "U")
        _p.rename("press","P")
        vtkfile_u << (_u,t)
        vtkfile_p << (_p,t)


    # update time
    t += dt

# save quantities of interest.
t_arr = np.linspace(0,t_final,num=nt)

np.savetxt(resultsname+'/s',s)
np.savetxt(resultsname+'/h',h)
np.savetxt(resultsname+'/wb',wb)
np.savetxt(resultsname+'/eta_mean',eta_mean)
np.savetxt(resultsname+'/beta_mean',beta_mean)
np.savetxt(resultsname+'/u_mean',u_mean)
np.savetxt(resultsname+'/x',X_fine)           # x = spatial coordinate
np.savetxt(resultsname+'/t',t_arr)            # t = time coordinate
np.savetxt(resultsname+'/lake_vol',lake_vol)
