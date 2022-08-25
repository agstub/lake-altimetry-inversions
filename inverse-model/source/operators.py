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
from kernel_fcns import ker,ker0,conv,xcor
from params import t,k,kx,ky,dx,Nx,lamda0,beta0
from kernel_fcns import ifftd,fftd
from prior import Cpri_inv
from noise import Cnoise_inv
#-------------------------------------------------------------------------------
def Cpost_inv(X,kappa,tau,a):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    # This is the inverse of the posterior covariance operator
    A = adj(Cnoise_inv(fwd(X))) + Cpri_inv(X,kappa,tau,a)
    return A

def fwd_uq(w,lamda=lamda0,beta=beta0):
    # forward operator for basal vertical velocity w
    # returns the fourier-transformed data (elevation) h minus the initial elevation profile
    w_ft = fftd(w)
    S_ft = conv(ker(lamda,beta),w_ft)
    return ifftd(S_ft).real


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
