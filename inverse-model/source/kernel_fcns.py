# this file contains the integral kernel functions that are used for applying the
# forward and adjoint operators
import numpy as np
from params import beta0,k,kx,lamda,t,dt,Nt,uh0,ub
from scipy.signal import fftconvolve
from scipy.fft import ifft2,fft2

#---------------------convolution and cross-correlation operators---------------
def conv(a,b):
    C = dt*fftconvolve(a,b,mode='full',axes=0)[0:Nt,:,:]
    return C

def xcor(a,b):
    C = dt*fftconvolve(np.conjugate(np.flipud(a)),b,mode='full',axes=0)[(Nt-1):2*Nt,:,:]
    return C

def fftd(f):
    C = fft2(f)
    return C

def ifftd(f):
    C = ifft2(f)
    return C

#------------------------Functions in kernel------------------------------------
def Rg():
    # Ice surface relaxation function for grounded ice
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    c_a = 0 # future: if we know bed slope then... (1j*kx/k)*np.tan(slope)
    R1 =  (1/n)*((1+g)*np.exp(4*n) - (2+4*g*n-4*c_a*n*(1+g*n))*np.exp(2*n) + 1 -g)
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g

    return R1/D

def Tw():
    # Basal velocity transfer function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g
    T1 = 2*(1+g)*(n+1)*np.exp(3*n) + 2*(1-g)*(n-1)*np.exp(n)

    return T1/D

Rg_ = Rg()
Tw_ = Tw()

#------------------------------- Kernel-----------------------------------------

def ker_w():
    # kernel for w_b forward problem
    K_0 = np.exp(-(1j*(2*np.pi*kx)*uh0+lamda*Rg_)*t)
    K = K_0*Tw_

    return K

ker_w_ = ker_w()
