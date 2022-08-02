#-------------------------------------------------------------------------------
# This file contains functions that:
# (1) define the boundaries (ice-air,ice-water,ice-bed) of the mesh, AND...
# (2) mark the boundaries of the mesh
#-------------------------------------------------------------------------------
from params import tol,Lngth,Hght
from geometry import bed
import numpy as np
from dolfin import *

#-------------------------------------------------------------------------------
# Define SubDomains for ice-water boundary, ice-bed boundary, inflow (x=0) and
# outflow (x=Length of domain). The parameter 'tol' is a minimal water depth
# used to distinguish the ice-water and ice-bed surfaces.

class ShoreBoundary(SubDomain):
    # Shore boundary.
    # This boundary is marked first and all of the irrelevant portions are
    # overwritten by the other boundary markers.
    def inside(self, x, on_boundary):
        return (on_boundary and (x[1]<0.5*Hght) )

class TopBoundary(SubDomain):
    # Shore boundary.
    # This boundary is marked first and all of the irrelevant portions are
    # overwritten by the other boundary markers.
    def inside(self, x, on_boundary):
        return (on_boundary and (x[1]>0.5*Hght) )

class WaterBoundary(SubDomain):
    # Ice-water boundary.
    # Lifting of ice from the bed *is not* allowed on this boundary.
    def inside(self, x, on_boundary):
        return (on_boundary and (x[1]<0.5*Hght) and  ((x[1]-bed(x[0]))>tol) )

class BedBoundary(SubDomain):
    # Ice-bed boundary away from the lake; the portions near the lake are overwritten
    # by BasinBoundary.
    # Lifting of ice from the bed *is not* allowed on this boundary.
    def inside(self, x, on_boundary):
        return (on_boundary and (x[1]<0.5*Hght) and ((x[1]-bed(x[0]))<=tol ) )

class LeftBoundary(SubDomain):
    # Left boundary
    def inside(self, x, on_boundary):
        return (on_boundary and np.abs(x[0])<tol  )

class RightBoundary(SubDomain):
    # Right boundary
    def inside(self, x, on_boundary):
        return (on_boundary and np.abs(x[0]-Lngth)<tol)

#-------------------------------------------------------------------------------

def mark_boundary(mesh):
    # Assign markers to each boundary segment (except the upper surface).
    # This is used at each time step to update the markers.
    #
    # Boundary marker numbering convention:
    # 1 - Left boundary
    # 2 - Right boundary
    # 3 - Ice-bed boundary
    # 4 - Ice-water boundary
    # 5 - "Shore" boundary
    # This function returns these markers, which are used to define the
    # boundary integrals and dirichlet conditions.

    boundary_markers = MeshFunction('size_t', mesh,dim=1)
    boundary_markers.set_all(0)

    # Mark shoreline
    bdryShore = ShoreBoundary()
    bdryShore.mark(boundary_markers, 5)

    # Mark upper surface (only needed for post-processing)
    bdryTop = TopBoundary()
    bdryTop.mark(boundary_markers, 6)

    # Mark ice-water boundary
    bdryWater = WaterBoundary()
    bdryWater.mark(boundary_markers, 4)

    # Mark ice-bed boundary away from lake
    bdryBed = BedBoundary()
    bdryBed.mark(boundary_markers, 3)

    # Mark inflow boundary
    bdryLeft = LeftBoundary()
    bdryLeft.mark(boundary_markers, 1)

    # Mark outflow boundary
    bdryRight = RightBoundary()
    bdryRight.mark(boundary_markers, 2)

    # # uncomment to check if bounadires are marked correctly
    # # (by viewing markers.pvd in paraview):
    # markers = File('markers.pvd')
    # markers << boundary_markers

    return boundary_markers
