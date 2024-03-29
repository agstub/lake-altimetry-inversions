#-------------------------------------------------------------------------------
# This file defines the bed topography and initial ice-water interface functions.
# Note: Bed and ice-water interface should be equal on margins of the domain!
#-------------------------------------------------------------------------------

import numpy as np
from params import Lngth,tol,X_fine
from scipy.integrate import quad

def bed(x):
    # generate bed topography

    B = 4-8*(np.exp((-x**4)/(8000**4) ))

    return B

def interface(x):
    # generate initial ice-water/ice-bed interface
    Int = np.maximum(0*x,bed(x))
    return Int

s_mean_0 = (8/(Lngth**2))*quad(lambda x: interface(x)*x,0,0.5*Lngth,full_output=1)[0]
