# perhaps remove any 'regional' trends away from the lake before inverting
import numpy as np
from params import x,y,Nt,Nx,Ny,ind

def localize(f):
    f_far = np.copy(f)
    f_far[np.sqrt(x**2+y**2)<0.8*np.max(np.sqrt(x**2+y**2))] = 0
    f_far[ind<1e-10] = 1e-10
    f_far[ind<1e-10] = 1e-10
    f_far = f_far.sum(axis=(1,2))/(f_far != 0).sum(axis=(1,2))
    f_loc = f- np.multiply.outer(f_far,np.ones((Nx,Ny)))
    return f_loc
