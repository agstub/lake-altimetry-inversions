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
from kernel_fcns import ker,ker0,conv,xcor,Rg
from params import t,k,kx,ky,dx,Nx,lamda0,beta0
from kernel_fcns import ifftd,fftd
from regularization import reg

#-------------------------------------------------------------------------------
def A(X,eps):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    # This is the inverse of the posterior covariance operator
    S = adj(fwd(X)) + reg(X,eps)
    return S

def fwd(w):
    # forward operator for basal vertical velocity w
    # returns the fourier-transformed data (elevation) h minus the initial elevation profile
    w_ft = fftd(w)
    S_ft = conv(ker0,w_ft)
    return S_ft

def adj(f_ft):
    # adjoint of the basal vertical velocity forward operator
    S = ifftd(xcor(ker0,f_ft)).real
    return S
