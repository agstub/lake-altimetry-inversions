# this file contains the inverse prior covariance operator

from params import k,x,y,dt
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # Laplacian computed via Fourier transform
    return -ifftd((k**2)*fftd(f)).real

def A(f,kappa):
    # elliptic operator
    R = (kappa**2)*f-lap(f)
    return R

def dfdt(f):
    return np.gradient(f,dt,axis=0)

def Qs2(f,kappa):
    # spatial component of precision operator
    return A(A(f,kappa),kappa)

def Qt(f,tau,a):
    # square root of temporal component of precision operator
    return dfdt(f)+a*f

def Qt_a(f,tau,a):
    # adjoint square root of temporal component of precision operator
    return -dfdt(f)+a*f

def Cpri_inv(f,kappa,tau,a):
    # inverse of prior covariance operator: C^-1 = Qt* Qs* Qs Qt
    # where: Qs = spatial component of square-root precision operator
    #        Qt = temporal component of square-root precision operator
    #        * = adjoint (note Qs* = Qs)
    #
    # this covariance is Matern in space (i.e. spatially correlated / smooth-ish)
    # and follows an Ornstein-Uhlenbeck process in time (i.e. basically a random
    # walk that continually returns to the mean). For large values of the parameter
    # "a", the prior looks like white noise in time. For smaller values of "a",
    # the process looks more like a Brownian motion.
    return (tau**2/(2*a))*Qt_a(Qs2(Qt(f,tau,a),kappa),tau,a)
