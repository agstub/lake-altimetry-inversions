#------------------------------------------------------------------------------
# These functions are used to:
# (1) update the mesh at each timestep by solving the
#     surface kinematic equations, AND...
# (2) compute the basal surface normal velocity
#------------------------------------------------------------------------------

from params import tol,Lngth,Hght,realtime_plot,X_fine,Nx,dt,dy
from dolfin import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from geometry import bed,interface
from scipy.interpolate import interp1d
from boundaryconds import mark_boundary

# ------------------------------------------------------------------------------

def mesh_routine(w,mesh,dt):
    # This function solves the surface kinematic equations and moves the mesh.
    # The mean elevation of the ice-water and ice-air interfaces are also
    # computed and returned.

    # first, compute slopes of free surfaces:
    # Returns: (1) FEniCS functions sx_fn and hx_fn, AND...
    #          (2) Python functions (h_int, s_int) of the surface elevations.

    sx_fn,s_int = get_sx_fn(w,mesh)
    hx_fn, h_int = get_hx_fn(w,mesh)

    # next, move the mesh
    move_mesh(mesh,sx_fn,hx_fn,dt,s_int,h_int,w)

    # plot surfaces in real time if realtime_plot = "on"
    plot_surfaces(h_int,s_int)

    return mesh,s_int,h_int,sx_fn


def move_mesh(mesh,sx_fn,hx_fn,dt,s_int,h_int,w):
    # this function computes the surface displacements and moves the mesh.

    M = mesh.coordinates()                           # Get mesh coordinates.

    w0 = w.sub(0).sub(1).compute_vertex_values(mesh) # Get vertical velocity at nodes.
    u0 = w.sub(0).sub(0).compute_vertex_values(mesh) # Get horizontal velocity at nodes.

    sx = sx_fn.compute_vertex_values(mesh)       # Get lower surface slope at nodes.
    hx = hx_fn.compute_vertex_values(mesh)       # Get upper surface slope at nodes.

    # Compute vertical displacements via the kinematic equation:
    # dZ/dt = w - u * dZ/dx
    # for Z = s(x,t) and Z = h(x,t).

    disp0 = w0 - u0*sx                               # Compute lower surface displacement.
    disp1 = w0 - u0*hx                               # Compute upper surface displacement.

    # Mark all of the vertices on the boundary of the mesh
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1]
    vertices_on_boundary = np.sort(vertices_on_boundary)


    # Loop over nodes in the boundary and displace them vertically according to
    # the velocity solution and surface slope.

    for i in vertices_on_boundary:
        # BOTTOM surface: ice-water interface
        if np.abs(M[i,1]-s_int(M[i,0]))<tol:

            M[i,1] += dt*disp0[i]

            # If new y-value is below the bed, set equal to the bed elevation
            if M[i,1]-bed(M[i,0])<0:
                M[i,1] = bed(M[i,0])

        #TOP surface: ice-air interface
        elif np.abs(M[i,1]-h_int(M[i,0]))<tol:
            M[i,1] += dt*disp1[i]


    # smooth the interior nodes of the mesh
    mesh.smooth()

#------------------------------------------------------------------------------

def get_sx_fn(w,mesh):
    # This function computes the slope of the lower surface and returns it as a FEniCS function.
    # The interpolated surface elevation is also returned as a python function.

    # Get x and y values of boundary nodes
    bmesh = BoundaryMesh(mesh,'exterior')
    M = bmesh.coordinates()
    X =  M[:,0][(M[:,0]>+tol)&(M[:,0]<0.5*Lngth-tol)&(M[:,1]<Hght-50)]
    Y =  M[:,1][ (M[:,0]>+tol)&(M[:,0]<0.5*Lngth-tol)&(M[:,1]<Hght-50)]

    Y = Y[np.argsort(X)]
    X = np.sort(X)

    s_left = np.min(M[:,1][M[:,0]<tol])

    s_right = np.min(M[:,1][np.abs(M[:,0]-0.5*Lngth)<tol])

    # Append values at x=0 and x=Lngth.
    Y = np.append(Y,s_right)
    Y = np.insert(Y,0,s_left)

    X = np.append(X,0.5*Lngth)
    X = np.insert(X,0,0)

    # Use SciPy to interpolate the lower surface
    s_int = interp1d(X,Y,kind='cubic',fill_value='extrapolate',bounds_error=False)

    # Define a FEniCS expression for the lower surface elevation
    class base_expr(UserExpression):
        def eval(self,value,x):
            value[0] = s_int(x[0])

    # Compute the slope of the lower surface in FEniCS
    V = FunctionSpace(mesh,'CG',1)
    base = base_expr(element=V.ufl_element(),domain=mesh)

    base_x = Dx(base,0)
    sx_fn = Function(V)
    sx_fn.assign(project(base_x,V))

    return sx_fn, s_int

#------------------------------------------------------------------------------

def get_hx_fn(w,mesh):
    # This function computes the upper surface slope and returns at as a FEniCS function.
    # The interpolated surface elevation is also returned.

    # Get coordinates of mesh nodes on boundary.
    bmesh = BoundaryMesh(mesh,'exterior')
    M = bmesh.coordinates()

    X =  M[:,0][(M[:,0]>+tol)&(M[:,0]<0.5*Lngth-tol)&(M[:,1]>Hght/2.)]
    Y =  M[:,1][ (M[:,0]>+tol)&(M[:,0]<0.5*Lngth-tol)&(M[:,1]>Hght/2.)]

    Y = Y[np.argsort(X)]
    X = np.sort(X)

    h_left = np.max(M[:,1][M[:,0]<+tol])

    h_right = np.max(M[:,1][np.abs(M[:,0]-0.5*Lngth)<tol])

    # Append values at x=0 and x=Lngth.
    Y = np.append(Y,h_right)
    Y = np.insert(Y,0,h_left)

    X = np.append(X,0.5*Lngth)
    X = np.insert(X,0,0)

    # Interpolate the boundary points:
    h_int = interp1d(X,Y,kind='cubic',fill_value='extrapolate',bounds_error=False)

    # Define a FEniCS expression for the upper surface elevation
    class surf_expr(UserExpression):
        def eval(self,value,x):
            value[0] = h_int(x[0])

    # Compute slope of upper surface
    V = FunctionSpace(mesh,'CG',1)
    surf = surf_expr(element=V.ufl_element(),domain=mesh)

    surf_x = Dx(surf,0)
    hx_fn = Function(V)
    hx_fn.assign(project(surf_x,V))

    return hx_fn,h_int

#------------------------------------------------------------------------------

def plot_surfaces(h_int,s_int):
    # Plotting in real time if realtime_plot is turned 'on' in the params.py file:
    # Saves a .png figure called 'surfaces' of the free surface geometry!
    if realtime_plot == 'on':
        X = X_fine
        Gamma_h = h_int(X)
        Gamma_s = s_int(X)

        plt.figure(figsize=(8,5))

        # Plot upper surface
        plt.plot(X/1000,Gamma_h[:]-0.98*Hght,color='royalblue',linewidth=1,label=r'$h-0.99h_0$')
        plt.fill_between(X/1000,y1=Gamma_s[:], y2=Gamma_h[:]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
        plt.fill_between(X/1000,bed(X),Gamma_s[:],facecolor='slateblue',alpha=0.5)
        plt.fill_between(X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)
        plt.plot(X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
        plt.plot(X[Gamma_s[:]-bed(X)>1e-3*tol]/1000,Gamma_s[:][Gamma_s[:]-bed(X)>1e-3*tol],'o',color='crimson',markersize=1,label=r'$s>\beta$')

        plt.plot(-X/1000,Gamma_h[:]-0.98*Hght,color='royalblue',linewidth=1,label=r'$h-0.99h_0$')
        plt.fill_between(-X/1000,y1=Gamma_s[:], y2=Gamma_h[:]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
        plt.fill_between(-X/1000,bed(X),Gamma_s[:],facecolor='slateblue',alpha=0.5)
        plt.fill_between(-X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)
        plt.plot(-X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
        plt.plot(-X[Gamma_s[:]-bed(X)>1e-3*tol]/1000,Gamma_s[:][Gamma_s[:]-bed(X)>1e-3*tol],'o',color='crimson',markersize=1,label=r'$s>\beta$')


        # Label axes and save png:
        plt.xlabel(r'$r$ (km)',fontsize=20)

        plt.yticks([])
        plt.xticks(fontsize=16)

        plt.ylim(np.min(bed(X))-2.0,0.02*Hght+5,8)
        plt.xlim(-0.5*Lngth/1000.0,0.5*Lngth/1000.0)
        plt.tight_layout()
        plt.savefig('surfaces',bbox_inches='tight')
        plt.close()

#------------------------------------------------------------------------------

# get upper surface elevation, lower surface elevation, basal vertical velocity,
# and mesh coordinates
def get_wb(w,mesh,sx_fn):

    M = mesh.coordinates()

    w_vv = w.sub(0).sub(1).compute_vertex_values(mesh)
    u_vv = w.sub(0).sub(0).compute_vertex_values(mesh)
    sx_vv = sx_fn.compute_vertex_values(mesh)       # Get lower surface slope at nodes.

    wb_vv = (w_vv-u_vv*sx_vv)/np.sqrt(1+sx_vv**2)

    X = M[:,0][np.abs(M[:,1])<0.25*dy]

    wb = wb_vv[np.abs(M[:,1])<0.25*dy]*3.154e7   # save in meters per year

    wb_int = interp1d(X,wb,kind='cubic',fill_value='extrapolate',bounds_error=False)

    return wb_int
