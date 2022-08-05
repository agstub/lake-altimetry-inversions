# perhaps remove any 'regional' trends away from the lake before inverting
import numpy as np
from params import x,y,Nt,Nx,Ny

def trim(f):
    f_far = np.copy(f)
    f_far[np.sqrt(x**2)<np.max(np.sqrt(x**2))-5] = 0
    f_far[np.sqrt(y**2)<np.min(np.sqrt(y**2))-5] = 0
    f_far = f_far.sum(axis=(1,2))/(f_far != 0).sum(axis=(1,2))
    f -= np.multiply.outer(f_far,np.ones((Nx,Ny)))
    return f
