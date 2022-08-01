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
from meshfcns import mesh_routine
import scipy.integrate as scpint
import os
from params import (rho_i,g,tol,t_final,Lngth,Hght,nt,dt,rho_w,dy,alpha,
                    print_convergence,X_fine,nx,Nx,Ny,t_period,C,save_vtk)

#--------------------Initial conditions-----------------------------------------
# compute initial mean elevation of ice-water interface and initial lake volume.
s_mean0 = np.mean(interface(X_fine)[interface(X_fine)-bed(X_fine)>tol])
lake_vol_0 = scpint.quad(lambda x: interface(x)-bed(x),0,Lngth,full_output=1)[0]
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
p1 = Point((Lngth,Hght))
mesh = RectangleMesh(p0,p1, Nx, Ny,diagonal="left/right")

M = mesh.coordinates()
# make sure all vertices are bounded below by the bed elevation
M[:,1][M[:,1]<bed(M[:,0])] = bed(M[:,0])[M[:,1]<bed(M[:,0])]
M[:,1][np.abs(M[:,1]-Hght)<0.9*dy] = Hght - M[:,0][np.abs(M[:,1]-Hght)<0.9*dy]*np.arctan(alpha)



# define arrays for saving surfaces, lake volume, water pressure, and
# grounding line positions over time.
Gamma_s = np.zeros((nx,nt))       # basal surface
Gamma_h = np.zeros((nx,nt))       # upper surface
lake_vol = np.zeros(nt)           # lake volume

t = 0                             # time

# begin time stepping
for i in range(10):

    print('-----------------------------------------------')
    print('Timestep '+str(i+1)+' out of '+str(nt))

    if t==0:
        # set initial conditions.
        s_mean_i = s_mean0                    # Mean ice-water elevation.
        w = get_zero(mesh)
        mesh,F_s,F_h,s_mean_i,h_mean_i = mesh_routine(w,mesh,dt)
        F_h = lambda x: Hght                  # Ice-air surface function
        F_s = lambda x: interface(x)          # Lower surface function

    # solve the Stoke problem, returns solution "w"

    w = stokes_solve(mesh,lake_vol_0,s_mean_i,F_h,F_s,t)

    # solve the surface kinematic equations, move the mesh, and compute the
    # grounding line positions.

    mesh,F_s,F_h,s_mean_i,h_mean_i = mesh_routine(w,mesh,dt)

    #print('sigma_e = '+str(rho_i*g*(H_out-H_mean) +rho_w*g*(s_out-s_mean_i)))
    #print('rho_w/rho_i * (s - s_mean) = '+str((rho_w/rho_i)*(s_out-s_mean_i)))

    # save quantities of interest
    Gamma_s[:,i] = F_s(X_fine)
    Gamma_h[:,i] = F_h(X_fine)

    # compute lake volume: integral of lower surface minus the bed elevation
    lake_vol[i] = scpint.quad(lambda x: F_s(x)-bed(x),0,Lngth,full_output=1)[0]


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

np.savetxt(resultsname+'/Gamma_s',Gamma_s)
np.savetxt(resultsname+'/Gamma_h',Gamma_h)
np.savetxt(resultsname+'/X',X_fine)           # X = spatial coordinate
np.savetxt(resultsname+'/t',t_arr)            # t = time coordinate
np.savetxt(resultsname+'/lake_vol',lake_vol)
