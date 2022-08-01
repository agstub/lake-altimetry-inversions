#-------------------------------------------------------------------------------
# This file defines the bed topography and initial ice-water interface functions.
# Note: Bed and ice-water interface should be equal on margins of the domain!
#-------------------------------------------------------------------------------

import numpy as np
from params import Lngth,tol,alpha

def bed(x):
    # generate bed topography
    B = 1-x*np.arctan(alpha)-8*(np.exp((-(x-Lngth/2.0)**4)/(8000**4) ))+7
    return B

def interface(x):
    # generate initial ice-water/ice-bed interface
    Int = np.maximum(0*x,bed(x))
    return Int
