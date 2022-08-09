#-------------------------------------------------------------------------------
# this file defines the
# * (1) forward operator,
# * (2) adjoint operator, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)
#
# To avoid redundant applications of the fftd/ifftd in the inversion for computational efficiency:
# * the forward operator takes physical-space fields as input and return Fourier-space fields
# * the adjoint operato takes Fourier-space fields as input and physical-space fields
#-------------------------------------------------------------------------------


import numpy as np
from kernel_fcns import Rg_,Tw_,ker_w_,conv,xcor
from params import t,k,kx,ky,dx,Nx
from kernel_fcns import ifftd,fftd
from regularizations import Cpri_inv

#-------------------------------------------------------------------------------
def adj_fwd(X,eps_1,eps_2):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    A = adj(fwd(X)) + Cpri_inv(X,eps_1,eps_2)
    return A

def fwd(w):
    # forward operator for basal vertical velocity w
    # returns the fourier-transformed data (elevation) h minus the initial elevation profile
    w_ft = fftd(w)
    S_ft = conv(ker_w_,w_ft)
    return S_ft

def adj(f_ft):
    # adjoint of the basal vertical velocity forward operator
    S = ifftd(xcor(ker_w_,f_ft)).real
    return S
#-------------------------------------------------------------------------------
