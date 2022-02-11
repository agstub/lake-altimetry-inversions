#-------------------------------------------------------------------------------
# this file defines the
# * (1) forward operator,
# * (2) adjoint operator, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)
#
# To avoid redundant applications of the fft/ifft in the inversion for computational efficiency:
# * the forward operator takes physical-space fields as input and return Fourier-space fields
# * the adjoint operato takes Fourier-space fields as input and physical-space fields
#-------------------------------------------------------------------------------


import numpy as np
from kernel_fcns import Rg_,Tw_,ker_w_,conv,xcor
from params import w_reg,t,k,kx,ky,dx
from scipy.fft import ifft2,fft2
from regularizations import reg

#-------------------------------------------------------------------------------
def adj_fwd(X,eps_w):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    A = adjoint_w(forward_w(X)) + eps_w*reg(X,w_reg)
    return A

def forward_w(w):
    # forward operator for basal vertical velocity w
    # returns the data (elevation) h minus the initial elevation profile
    w_ft = fft2(w)
    S_ft = conv(ker_w_,w_ft)
    return S_ft

def adjoint_w(f_ft):
    # adjoint of the basal vertical velocity forward operator
    S = ifft2(xcor(ker_w_,f_ft)).real
    return S
#-------------------------------------------------------------------------------
