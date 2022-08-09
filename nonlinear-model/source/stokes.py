# This file contains the functions needed for solving the Stokes system.

from params import rho_i,g,tol,B,rm2,rho_w,C,eps_p,eps_b,eps_v,dt,quad_degree,Lngth,t_period,Hght
from boundaryconds import mark_boundary,apply_bcs
from geometry import bed
from hydrology import Vdot,lake_vol_0
import numpy as np
from dolfin import *

def dPi(un):
        # derivative of penalty functional for enforcing impenetrability
        # on the ice-bed boundary.
        return un+abs(un)

def Pi(u,nu):
        # penalty functional for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return 0.5*(un**2.0+un*abs(un))

def eta(u,x):
        return 0.5*B*((inner(D(u,x),D(u,x))+Constant(eps_v))**(rm2/2.0))

def beta(Tu):
        return Constant(C)*((Constant(eps_b)+inner(Tu,Tu))**(rm2/2.0))

def D(u,x):
    # strain rate in cylindrical coordinates
    return sym(as_tensor([[u[0].dx(0), 0, u[0].dx(1)],
                          [0, u[0]/x[0], 0],
                          [u[1].dx(0), 0, u[1].dx(1)]]))
def sigma(u,p,x):
    # stress tensor
        return -p*Identity(3) + 2*eta(u,x)*D(u,x)

def shear_bdry(u,v,nu,x):
        n_ext = as_tensor([nu[0],0,nu[1]])
        v_ext = as_tensor([v[0],0,v[1]])
        T_ext = Identity(3) - outer(n_ext,n_ext)
        TDn = dot(T_ext, dot(D(u,x),n_ext))
        Tv = dot(T_ext, v_ext)
        return inner(TDn,Tv)

def div_c(u,x):
    # divergence in cylindrical coordinates
        return u[0].dx(0) + u[0]/x[0] + u[1].dx(1)

def weak_form(u,p,pw,v,q,qw,f,g_lake,g_out,ds,nu,T,t,x):
    # define weak form of the subglacial lake problem

    # measures of the extended boundary (L0) and ice-water boundary (L1)
    L0 = Constant(assemble(1*ds(4))+assemble(1*ds(3)))
    L1 = Constant(assemble(1*ds(4)))

    # Nonlinear residual
    un = dot(u,nu)
    vn = dot(v,nu)

    Fw =  inner(sigma(u,p,x),D(v,x))*x[0]*dx + q*div_c(u,x)*x[0]*dx - inner(f, v)*x[0]*dx\
         + (g_lake+pw+Constant(rho_w*g*dt)*(un+Constant(Vdot(lake_vol_0,t)/(np.pi*(L1**2)))))*vn*x[0]*ds(4)\
         + qw*(un+Constant(Vdot(lake_vol_0,t)/(np.pi*L0**2)))*x[0]*ds(4)\
         + (g_lake+pw+Constant(rho_w*g*dt)*(un+Constant(Vdot(lake_vol_0,t)/(np.pi*(L1**2)))))*vn*x[0]*ds(3)\
         + qw*(un+Constant(Vdot(lake_vol_0,t)/(np.pi*L0**2)))*x[0]*ds(3)\
         + Constant(1/eps_p)*dPi(un)*vn*x[0]*ds(3)\
         + beta(dot(T,u))*inner(dot(T,u),dot(T,v))*x[0]*ds(3)\
         - shear_bdry(u,v,nu,x)*x[0]*ds(2) + g_out*vn*x[0]*ds(2)
    return Fw

def stokes_solve(mesh,s_mean,F_h,F_s,t):
        # stokes solver using Taylor-Hood elements and a Lagrange multiplier
        # for the water pressure.

        # define function spaces
        P1 = FiniteElement('P',triangle,1)     # pressure
        P2 = FiniteElement('P',triangle,2)     # velocity
        R  = FiniteElement("R", triangle,0)    # mean water pressure
        element = MixedElement([[P2,P2],P1,R])

        W = FunctionSpace(mesh,element)

        #---------------------define variational problem------------------------
        w = Function(W)
        (u,p,pw) = split(w)             # (velocity,pressure,mean water pressure)
        (v,q,qw) = TestFunctions(W)     # test functions corresponding to (u,p,pw)

        h_out = float(F_h(0.5*Lngth))   # upper surface elevation at outflow

        # Define Neumann condition at ice-water interface
        g_lake = Expression('rho_w*g*(s_mean-x[1])',rho_w=rho_w,g=g,s_mean=s_mean,degree=1)

        # Define cryostatic normal stress conditions for inflow/outflow boundaries
        g_out = Expression('rho_i*g*(h_out-x[1])',rho_i=rho_i,g=g,h_out=h_out,degree=1)

        f = Constant((0,-rho_i*g))        # Body force
        nu = FacetNormal(mesh)            # Outward-pointing unit normal to the boundary
        I = Identity(2)                   # Identity tensor
        T = I - outer(nu,nu)              # Orthogonal projection (onto boundary)

        # mark the boundary and define a measure for integration
        boundary_markers = mark_boundary(mesh)
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        cell_markers = MeshFunction('size_t', mesh,dim=2)
        dx = Measure('dx', domain=mesh, subdomain_data=cell_markers)

        x = SpatialCoordinate(mesh)

        # define weak form
        Fw = weak_form(u,p,pw,v,q,qw,f,g_lake,g_out,ds,nu,T,t,x)

        bcs = apply_bcs(W,boundary_markers)

        # solve for (u,p,pw).
        solve(Fw == 0, w, bcs=bcs,solver_parameters={"newton_solver":{"relative_tolerance": 1e-14,"maximum_iterations":50}},form_compiler_parameters={"quadrature_degree":quad_degree,"optimize":True})

        beta_i = assemble(beta(dot(T,w.sub(0)))*ds(3))/assemble(Constant(1)*ds(3))
        eta_i = assemble(eta(w.sub(0),x)*dx)/assemble(Constant(1)*dx)
        u_i = assemble(sqrt(dot(w.sub(0),w.sub(0)))*dx)/assemble(Constant(1)*dx)*3.154e7

        print('mean u [m/yr] = '+str(u_i))
        print('mean drag [Pa s/m] = '+"{:.2e}".format(beta_i))
        print('mean viscosity [Pa s] = '+"{:.2e}".format(eta_i))

        # return solution w
        return w,beta_i,eta_i,u_i
