# this defines the regularization

from params import k,x,y,dt,t
from kernel_fcns import ifftd,fftd
import numpy as np

def lap(f):
    # Negative Laplacian computed via Fourier transform
    return ifftd((k**2)*fftd(f)).real


def reg(f,eps):
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

    ## OR return tau*f == white noise prior.... faster convergence?!?!?
    #return eps*f #eps = 3
    return eps*lap(f) #
    #return 10*tau*Qt_a(Qs2(Qt(f,a),kappa),a)
