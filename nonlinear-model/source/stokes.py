# This file contains the functions needed for solving the Stokes system.

from params import rho_i,g,tol,B,rm2,rho_w,C,eps_p,eps_v,dt,quad_degree,Lngth,t_period,Hght,alpha
from boundaryconds import mark_boundary
from geometry import bed
from hydrology import Vdot
import numpy as np
from dolfin import *

def dPi(u,nu,t):
        # derivative of penalty functional for enforcing impenetrability
        # on the ice-bed boundary.
        un = dot(u,nu)
        return un+abs(un)

def Pi(u,nu):
        # penalty functional for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return 0.5*(un**2.0+un*abs(un))

def eta(u):
        return 0.5*B*((inner(sym(grad(u)),sym(grad(u)))+Constant(eps_v))**(rm2/2.0))

def beta(Tu):
        return Constant(C)*((Constant(eps_v)+inner(Tu,Tu))**(rm2/2.0))

def sigma(u,p):
        return -p*Identity(2) + 2*eta(u)*sym(grad(u))

def weak_form(u,p,pw,v,q,qw,f,g_lake,g_in,g_out,ds,nu,T,lake_vol_0,t):
    # define weak form of the subglacial lake problem

    # measures of the extended boundary (L0) and ice-water boundary (L1)
    L0 = Constant(assemble(1*ds(4))+assemble(1*ds(5)))
    L1 = Constant(assemble(1*ds(4)))

    # Nonlinear residual
    Fw =  2*eta(u)*inner(sym(grad(u)),sym(grad(v)))*dx + (- div(v)*p + q*div(u))*dx - inner(f, v)*dx\
         + (g_lake+pw+Constant(rho_w*g*dt)*(dot(u,nu)+Constant(Vdot(lake_vol_0,t)/L1)))*inner(nu, v)*ds(4)\
         + qw*(inner(u,nu)+Constant(Vdot(lake_vol_0,t))/(L0))*ds(4)\
         + (g_lake+pw+Constant(rho_w*g*dt)*(dot(u,nu)+Constant(Vdot(lake_vol_0,t)/L1)))*inner(nu, v)*ds(5)\
         + qw*(inner(u,nu)+Constant(Vdot(lake_vol_0,t))/(L0) )*ds(5)\
         + Constant(1e-4/eps_p)*dPi(u,nu,t)*dot(v,nu)*ds(5)\
         + Constant(1/eps_p)*dot(u,nu)*dot(v,nu)*ds(3)\
         + beta(dot(T,u))*inner(dot(T,u),dot(T,v))*ds(3)\
         - inner( dot(T, dot(sigma(u,p),nu)),dot(T,v) )*ds(1) - inner( dot(T, dot(sigma(u,p),nu)),dot(T,v) )*ds(2)\
         + g_out*inner(nu,v)*ds(2) + g_in*inner(nu,v)*ds(1)
    return Fw


def stokes_solve(mesh,lake_vol_0,s_mean,F_h,F_s,t):
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

        h_out = float(F_h(Lngth))       # upper surface elevation at outflow
        h_in = float(F_h(0.0))          # upper surface elevation at inflow

        # Define Neumann condition at ice-water interface
        g_lake = Expression('rho_w*g*(s_mean-x[1])',rho_w=rho_w,g=g,s_mean=s_mean,degree=1)

        # Define cryostatic normal stress conditions for inflow/outflow boundaries
        g_out = Expression('rho_i*g*(h_out-x[1])',rho_i=rho_i,g=g,h_out=h_out,degree=1)
        g_in = Expression('rho_i*g*(h_in-x[1])',rho_i=rho_i,g=g,h_in=h_in,degree=1)

        f = Constant((0,-rho_i*g))        # Body force
        nu = FacetNormal(mesh)            # Outward-pointing unit normal to the boundary
        I = Identity(2)                   # Identity tensor
        T = I - outer(nu,nu)              # Orthogonal projection (onto boundary)

        # mark the boundary and define a measure for integration
        boundary_markers = mark_boundary(mesh)
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        cell_markers = MeshFunction('size_t', mesh,dim=2)
        dx = Measure('dx', domain=mesh, subdomain_data=cell_markers)


        # define weak form
        Fw = weak_form(u,p,pw,v,q,qw,f,g_lake,g_in,g_out,ds,nu,T,lake_vol_0,t)

        # solve for (u,p,pw).
        solve(Fw == 0, w, bcs=[],solver_parameters={"newton_solver":{"relative_tolerance": 1e-14,"maximum_iterations":50}},form_compiler_parameters={"quadrature_degree":quad_degree,"optimize":True,"eliminate_zeros":False})

        beta_mean = assemble(beta(w.sub(0))*ds(3))/assemble(Constant(1)*ds(3))
        eta_mean = assemble(eta(w.sub(0))*dx)/assemble(Constant(1)*dx)

        print('mean drag = '+"{:.2E}".format(beta_mean)+' Pa s/m')
        print('mean viscosity = '+"{:.2E}".format(eta_mean)+' Pa s')


        # return solution w
        return w

def get_zero(mesh):
        # get zero element of function space.
        # only used for setting initial conditions; see main.py.

        # define function spaces
        P1 = FiniteElement('P',triangle,1)     # pressure
        P2 = FiniteElement('P',triangle,2)     # velocity
        R = FiniteElement("R", triangle,0)     # mean water pressure
        element = MixedElement([[P2,P2],P1,R])
        W = FunctionSpace(mesh,element)        # function space for (u,p,pw)

        w = Function(W)                        # zero by default

        return w
